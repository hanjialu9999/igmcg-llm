"""第二十八轮回归测试：死代码清理 + RoPE 向量化优化。

覆盖：
- R28-1: _cached_causal_mask 已删除（死代码），_build_causal_window_mask 始终返回 float mask
- R28-2: RoPE _rope_apply 向量化与原方案数值等价（含 Partial RoPE / dim_wise 回退）
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.rope import RotaryEmbedding
from models.transformer import TransformerModel


def _small(**over):
    kw = dict(vocab_size=200, embedding_dim=64, num_heads=4, num_layers=2,
              hidden_dim=128, max_seq_length=32)
    kw.update(over)
    return TransformerModel(**kw)


# ─── R28-1: 死代码清理 ──────────────────────────────────────────────────────

def test_cached_causal_mask_removed():
    """R28-1: _cached_causal_mask 方法已删除（R27 后死代码）。"""
    m = _small(num_layers=1)
    attn = m.blocks[0].attn
    assert not hasattr(attn, '_cached_causal_mask'), \
        "_cached_causal_mask 方法应已删除"
    assert not hasattr(attn, '_causal_key'), "_causal_key 属性应已删除"
    assert not hasattr(attn, '_causal_cache'), "_causal_cache 属性应已删除"


def test_build_causal_window_mask_always_float():
    """R28-1: _build_causal_window_mask 始终返回 float mask（不返回 None）。"""
    m = _small(num_layers=1)
    attn = m.blocks[0].attn
    dev = torch.device('cpu')
    # 纯因果路径（window=0, mem_cols=0）
    mask = attn._build_causal_window_mask(8, 8, 0, dev, 0)
    assert mask is not None, "纯因果路径也应返回 float mask"
    assert mask.dtype == torch.float32, f"mask dtype 应为 float，got {mask.dtype}"
    assert mask.shape == (1, 1, 8, 8), f"mask shape 错误：{mask.shape}"


def test_generate_determinism_after_cleanup():
    """R28-1: 清理死代码后生成仍确定性。"""
    m = _small(num_layers=2)
    m.eval()
    out1 = m.generate([1, 2, 3], max_length=8, device='cpu', temperature=1.0, top_k=0)
    out2 = m.generate([1, 2, 3], max_length=8, device='cpu', temperature=1.0, top_k=0)
    assert out1 == out2, "生成结果不确定"


# ─── R28-2: RoPE 向量化数值等价 ─────────────────────────────────────────────

def test_rope_vectorized_matches_original():
    """R28-2: 向量化 RoPE 与原方案（手写 x1*cos-x2*sin）数值等价。"""
    torch.manual_seed(42)
    rope = RotaryEmbedding(dim=32)
    q = torch.randn(2, 4, 8, 32)
    k = torch.randn(2, 4, 8, 32)
    q_out, k_out = rope(q, k, start_pos=0)

    # 手算原方案
    cos, sin = rope._get_cos_sin(0, 8, q.device, q.dtype)
    d = 16  # rot_dim // 2
    q1, q2 = q[..., :d], q[..., d:]
    cos_half, sin_half = cos[..., :d], sin[..., :d]
    expected = torch.cat([
        q1 * cos_half - q2 * sin_half,
        q1 * sin_half + q2 * cos_half,
    ], dim=-1)
    assert torch.allclose(q_out, expected, atol=1e-6), \
        f"向量化 RoPE 与原方案不一致: max_diff={(q_out - expected).abs().max().item():.2e}"


def test_rope_vectorized_partial_rope():
    """R28-2: Partial RoPE（dim_fraction<1）向量化路径正确。"""
    torch.manual_seed(0)
    rope = RotaryEmbedding(dim=8, dim_fraction=0.5)  # rot_dim=4, no_pe_dim=4
    x = torch.randn(1, 1, 4, 8)
    out = rope.apply_to_single(x, start_pos=0)
    assert out.shape == x.shape

    # 手算
    cos, sin = rope._get_cos_sin(0, 4, x.device, x.dtype)
    d = 2
    x_rot = x[..., :4]
    x_pass = x[..., 4:]
    x1, x2 = x_rot[..., :d], x_rot[..., d:]
    cos_half, sin_half = cos[..., :d], sin[..., :d]
    x_rotated = torch.cat([
        x1 * cos_half - x2 * sin_half,
        x1 * sin_half + x2 * cos_half,
    ], dim=-1)
    expected = torch.cat([x_rotated, x_pass], dim=-1)
    assert torch.allclose(out, expected, atol=1e-6), \
        f"Partial RoPE 向量化不一致: max_diff={(out - expected).abs().max().item():.2e}"


def test_rope_dim_wise_falls_back():
    """R28-2: dim_wise=True 时回退原方案（sign 向量化失效）。"""
    torch.manual_seed(123)
    rope = RotaryEmbedding(dim=32, dim_wise=True)
    # 初始 dim_wise_logit=0 → sigmoid=0.5，非对称修改 cos/sin
    q = torch.randn(1, 2, 4, 32)
    k = torch.randn(1, 2, 4, 32)
    q_out, k_out = rope(q, k, start_pos=0)

    # 手算原方案（dim_wise 路径）
    cos, sin = rope._get_cos_sin(0, 4, q.device, q.dtype)
    cos, sin = rope._apply_dim_wise_mask(cos, sin)
    d = 16
    q1, q2 = q[..., :d], q[..., d:]
    cos_half, sin_half = cos[..., :d], sin[..., :d]
    expected = torch.cat([
        q1 * cos_half - q2 * sin_half,
        q1 * sin_half + q2 * cos_half,
    ], dim=-1)
    assert torch.allclose(q_out, expected, atol=1e-6), \
        f"dim_wise 回退路径不一致: max_diff={(q_out - expected).abs().max().item():.2e}"


def test_rope_identity_when_sin_zero():
    """R28-2: sin=0 时 RoPE 为恒等变换（cos=1）。"""
    rope = RotaryEmbedding(dim=16)
    x = torch.randn(1, 1, 4, 16)
    # 手工传 cos=1, sin=0
    cos = torch.ones(1, 1, 4, 16)
    sin = torch.zeros(1, 1, 4, 16)
    out = rope._rope_apply(x, cos, sin)
    assert torch.allclose(out, x, atol=1e-6), \
        f"sin=0 时应恒等，max_diff={(out - x).abs().max().item():.2e}"


def test_rope_sign_buffer_exists():
    """R28-2: _rope_sign buffer 存在且形状正确。"""
    rope = RotaryEmbedding(dim=32)
    assert hasattr(rope, '_rope_sign'), "_rope_sign buffer 应存在"
    assert rope._rope_sign.shape == (32,), \
        f"_rope_sign shape 错误：{rope._rope_sign.shape}"
    # 前半 -1，后半 +1
    d = 16
    assert torch.allclose(rope._rope_sign[:d], -torch.ones(d)), "前半应为 -1"
    assert torch.allclose(rope._rope_sign[d:], torch.ones(d)), "后半应为 +1"


# ─── 端到端：训练 + 增量解码 parity ──────────────────────────────────────────

def test_train_forward_not_crash():
    """R28-2: 训练前向不崩溃，梯度正常回流。"""
    torch.manual_seed(0)
    m = _small(num_layers=2)
    m.train()
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    loss = out.sum()
    loss.backward()
    assert out.shape == (2, 8, 200)
    assert torch.isfinite(out).all(), "输出含 nan/inf"
    # 梯度应回流到 RoPE 相关参数
    assert m.blocks[0].attn.qkv.weight.grad is not None, "梯度未回流"


def test_cache_parity_after_rope_optimization():
    """R28-2: RoPE 优化后 train/infer cache parity 保持。"""
    torch.manual_seed(77)
    m = _small(num_layers=2, alibi=True)
    m.eval()
    x = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        full = m(x, use_cache=False)
        out, past = m(x[:, :1], past_key_values=None, use_cache=True)
        for t in range(1, 8):
            out, past = m(x[:, t:t + 1], past_key_values=past, use_cache=True)
    diff = (full[:, -1, :] - out[:, -1, :]).abs().max().item()
    assert diff < 1e-4, f"RoPE 优化后 cache parity 差 {diff:.2e} 超过 1e-4"
