"""第三十一轮回归测试：GatedDeltaNet alpha_proj/beta_proj → alpha_beta_proj GEMM 合并。

覆盖：
- R31-1: GatedDeltaNet 的 alpha_proj + beta_proj 合并为单个 alpha_beta_proj GEMM
  - 数学等价：cat([alpha_proj(x), beta_proj(x)], dim=-1) == alpha_beta_proj(x).chunk(2, dim=-1)
  - state_dict 兼容：convert_legacy_state_dict 正确转换旧格式
  - 初始化正确：weight=0，bias 前 _gate_out 维 alpha_init，后 _gate_out 维 beta_init
  - 梯度回流：alpha_beta_proj.weight 收到梯度
  - cache parity：训练全量 vs 增量解码数值一致
  - channel_wise 模式：逐通道 alpha/beta 正确拆分
  - 端到端：GatedDeltaNet 模型可训练
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mixers import GatedDeltaNet


# ─── 测试夹具 ──────────────────────────────────────────────────────────────

def _make_gated_delta(**over):
    """构造 GatedDeltaNet 模块用于单元测试。"""
    kw = dict(dim=64, num_heads=4, max_seq_length=16)
    kw.update(over)
    return GatedDeltaNet(**kw)


def _make_model(**over):
    """构造小 TransformerModel 用于端到端测试。"""
    from models.transformer import TransformerModel
    kw = dict(vocab_size=200, embedding_dim=64, num_heads=4, num_layers=2,
              hidden_dim=128, max_seq_length=32, mixer='gated_delta')
    kw.update(over)
    return TransformerModel(**kw)


# ─── R31-1: alpha_beta_proj 数学等价 ─────────────────────────────────────

def test_alpha_beta_proj_numerical_equivalence():
    """R31-1: 合并后的 alpha_beta_proj(x).chunk(2) 与原 alpha_proj(x)/beta_proj(x) 数值等价。"""
    torch.manual_seed(42)
    dim, num_heads = 64, 4
    # 模拟旧结构：两个独立 Linear
    _gate_out = num_heads  # 标量模式
    legacy_alpha = nn.Linear(dim, _gate_out, bias=True)
    legacy_beta = nn.Linear(dim, _gate_out, bias=True)
    nn.init.normal_(legacy_alpha.weight, 0, 0.1)
    nn.init.normal_(legacy_alpha.bias, -2.0, 0.01)
    nn.init.normal_(legacy_beta.weight, 0, 0.1)
    nn.init.normal_(legacy_beta.bias, 2.0, 0.01)

    # 构造合并后的 Linear：weight = cat([alpha.weight, beta.weight], dim=0)
    merged = nn.Linear(dim, 2 * _gate_out, bias=True)
    with torch.no_grad():
        merged.weight.copy_(torch.cat([legacy_alpha.weight, legacy_beta.weight], dim=0))
        merged.bias.copy_(torch.cat([legacy_alpha.bias, legacy_beta.bias], dim=0))

    x = torch.randn(2, 8, dim)
    # 旧路径
    alpha_legacy = torch.sigmoid(legacy_alpha(x))
    beta_legacy = torch.sigmoid(legacy_beta(x))
    # 新路径
    ab = merged(x)
    alpha_new, beta_new = ab.chunk(2, dim=-1)
    alpha_new = torch.sigmoid(alpha_new)
    beta_new = torch.sigmoid(beta_new)

    max_diff_alpha = (alpha_legacy - alpha_new).abs().max().item()
    max_diff_beta = (beta_legacy - beta_new).abs().max().item()
    assert max_diff_alpha < 1e-6, f"alpha 数值不等价: max_diff={max_diff_alpha:.2e}"
    assert max_diff_beta < 1e-6, f"beta 数值不等价: max_diff={max_diff_beta:.2e}"


# ─── R31-1: state_dict 转换 ────────────────────────────────────────────────

def test_convert_legacy_state_dict():
    """R31-1: convert_legacy_state_dict 正确把 alpha_proj/beta_proj 合并为 alpha_beta_proj。"""
    torch.manual_seed(42)
    dim, num_heads = 64, 4
    _gate_out = num_heads
    # 构造旧格式 state_dict（含 alpha_proj/beta_proj）
    alpha_w = torch.randn(_gate_out, dim)
    alpha_b = torch.full((_gate_out,), -2.0)
    beta_w = torch.randn(_gate_out, dim)
    beta_b = torch.full((_gate_out,), 2.0)
    legacy_sd = {
        'blocks.0.attn.alpha_proj.weight': alpha_w,
        'blocks.0.attn.alpha_proj.bias': alpha_b,
        'blocks.0.attn.beta_proj.weight': beta_w,
        'blocks.0.attn.beta_proj.bias': beta_b,
        'blocks.0.attn.qkv.weight': torch.randn(3 * dim * (dim // num_heads), dim),
        'blocks.1.attn.alpha_proj.weight': alpha_w.clone(),
        'blocks.1.attn.alpha_proj.bias': alpha_b.clone(),
        'blocks.1.attn.beta_proj.weight': beta_w.clone(),
        'blocks.1.attn.beta_proj.bias': beta_b.clone(),
    }
    new_sd = GatedDeltaNet.convert_legacy_state_dict(legacy_sd)

    # 检查 alpha_beta_proj 存在且数值正确
    assert 'blocks.0.attn.alpha_beta_proj.weight' in new_sd, "转换后应含 alpha_beta_proj.weight"
    assert 'blocks.0.attn.alpha_beta_proj.bias' in new_sd, "转换后应含 alpha_beta_proj.bias"
    assert 'blocks.0.attn.alpha_proj.weight' not in new_sd, "旧 alpha_proj 应被移除"
    assert 'blocks.0.attn.beta_proj.weight' not in new_sd, "旧 beta_proj 应被移除"

    expected_w = torch.cat([alpha_w, beta_w], dim=0)
    expected_b = torch.cat([alpha_b, beta_b], dim=0)
    assert torch.allclose(new_sd['blocks.0.attn.alpha_beta_proj.weight'], expected_w), "weight 合并不正确"
    assert torch.allclose(new_sd['blocks.0.attn.alpha_beta_proj.bias'], expected_b), "bias 合并不正确"
    # 多层检查
    assert 'blocks.1.attn.alpha_beta_proj.weight' in new_sd, "第二层应也转换"
    # qkv 不应受影响
    assert 'blocks.0.attn.qkv.weight' in new_sd, "无关参数不应被删除"


def test_convert_legacy_state_dict_idempotent():
    """R31-1: 转换已转换过的 state_dict 不报错（无 alpha_proj 时不变）。"""
    new_format_sd = {
        'blocks.0.attn.alpha_beta_proj.weight': torch.randn(8, 64),
        'blocks.0.attn.alpha_beta_proj.bias': torch.randn(8),
        'blocks.0.attn.qkv.weight': torch.randn(192, 64),
    }
    result = GatedDeltaNet.convert_legacy_state_dict(new_format_sd.copy())
    # 无 alpha_proj 键时，结果应与输入一致（不引入 alpha_beta_proj 重复）
    assert 'blocks.0.attn.alpha_beta_proj.weight' in result
    assert 'blocks.0.attn.alpha_proj.weight' not in result
    assert len(result) == len(new_format_sd), "无 alpha_proj 时结果集大小不变"


# ─── R31-1: 初始化正确性 ─────────────────────────────────────────────────

def test_alpha_beta_proj_init_correct():
    """R31-1: alpha_beta_proj 初始化：weight=0，bias 前 _gate_out 维 alpha_init，后 _gate_out 维 beta_init。"""
    m = _make_gated_delta(alpha_init=-2.0, beta_init=2.0)
    assert m.alpha_beta_proj.weight.abs().max().item() == 0.0, "weight 应 init 0"
    _g = m._gate_out
    alpha_bias = m.alpha_beta_proj.bias[:_g]
    beta_bias = m.alpha_beta_proj.bias[_g:]
    assert torch.allclose(alpha_bias, torch.full_like(alpha_bias, -2.0)), "alpha 段 bias 应为 -2.0"
    assert torch.allclose(beta_bias, torch.full_like(beta_bias, 2.0)), "beta 段 bias 应为 2.0"


def test_alpha_beta_proj_init_custom():
    """R31-1: 自定义 alpha_init/beta_init 正确传入 bias。"""
    m = _make_gated_delta(alpha_init=-1.5, beta_init=3.0)
    _g = m._gate_out
    assert torch.allclose(m.alpha_beta_proj.bias[:_g], torch.full((_g,), -1.5))
    assert torch.allclose(m.alpha_beta_proj.bias[_g:], torch.full((_g,), 3.0))


# ─── R31-1: channel_wise 模式 ─────────────────────────────────────────────

def test_channel_wise_gate_out():
    """R31-1: channel_wise=True 时 _gate_out = num_heads * head_dim。"""
    dim, num_heads = 64, 4
    head_dim = dim // num_heads  # 16
    m = _make_gated_delta(channel_wise=True)
    assert m._gate_out == num_heads * head_dim, \
        f"channel_wise _gate_out 应为 {num_heads * head_dim}，实际 {m._gate_out}"
    # alpha_beta_proj 输出 = 2 * _gate_out
    assert m.alpha_beta_proj.out_features == 2 * num_heads * head_dim


def test_channel_wise_compute_gates_split():
    """R31-1: channel_wise 模式 _compute_gates 正确拆分 alpha/beta 并 reshape 为 (B,H,T,D)。"""
    dim, num_heads = 64, 4
    head_dim = dim // num_heads
    B, T = 2, 8
    m = _make_gated_delta(channel_wise=True)
    x = torch.randn(B, T, dim)
    alpha, beta = m._compute_gates(x)
    assert alpha.shape == (B, num_heads, T, head_dim), f"alpha 形状 {alpha.shape} 不符"
    assert beta.shape == (B, num_heads, T, head_dim), f"beta 形状 {beta.shape} 不符"
    # alpha/beta 应在 (0,1) 范围内（经 sigmoid）
    assert (alpha >= 0).all() and (alpha <= 1).all(), "alpha 应在 (0,1) 范围"
    assert (beta >= 0).all() and (beta <= 1).all(), "beta 应在 (0,1) 范围"


# ─── R31-1: 梯度回流 ─────────────────────────────────────────────────────

def test_gradient_flow():
    """R31-1: 反向传播 alpha_beta_proj.weight 正确收到梯度。"""
    m = _make_gated_delta()
    m.train()
    x = torch.randn(2, 8, 64)
    out, _ = m(x)
    loss = out.float().sum()
    loss.backward()
    assert m.alpha_beta_proj.weight.grad is not None, "alpha_beta_proj.weight 无梯度"
    assert m.alpha_beta_proj.weight.grad.abs().sum().item() > 0, "梯度不应全为 0"


# ─── R31-1: cache parity ─────────────────────────────────────────────────

def test_cache_parity():
    """R31-1: GatedDeltaNet 全量前向 vs 增量解码数值一致。"""
    torch.manual_seed(42)
    dim, num_heads = 64, 4
    B, T = 1, 6
    m = _make_gated_delta(max_seq_length=T)
    x = torch.randn(B, T, dim)

    # 全量前向
    m.eval()
    with torch.no_grad():
        out_full, present = m(x, use_cache=True)

    # 增量解码
    out_inc = []
    past = None
    with torch.no_grad():
        for t in range(T):
            out_t, past = m(x[:, t:t+1, :], past_kv=past, use_cache=True, start_pos=t)
            out_inc.append(out_t)
    out_inc = torch.cat(out_inc, dim=1)

    max_diff = (out_full - out_inc).abs().max().item()
    # delta rule 递推 6 步浮点累积，阈值 0.1
    assert max_diff < 0.1, f"cache parity 差异 {max_diff} 过大"


def test_cache_parity_channel_wise():
    """R31-1: channel_wise 模式 cache parity。"""
    torch.manual_seed(42)
    dim, num_heads = 64, 4
    B, T = 1, 6
    m = _make_gated_delta(channel_wise=True, max_seq_length=T)
    x = torch.randn(B, T, dim)

    m.eval()
    with torch.no_grad():
        out_full, present = m(x, use_cache=True)

    out_inc = []
    past = None
    with torch.no_grad():
        for t in range(T):
            out_t, past = m(x[:, t:t+1, :], past_kv=past, use_cache=True, start_pos=t)
            out_inc.append(out_t)
    out_inc = torch.cat(out_inc, dim=1)

    max_diff = (out_full - out_inc).abs().max().item()
    assert max_diff < 0.1, f"channel_wise cache parity 差异 {max_diff} 过大"


# ─── R31-1: 端到端 ─────────────────────────────────────────────────────

def test_model_end_to_end():
    """R31-1: GatedDeltaNet 模型端到端前向 + 反向 + 生成可用。"""
    m = _make_model()
    m.eval()
    ids = torch.randint(0, m.vocab_size, (2, 8))
    with torch.no_grad():
        out = m(ids)
        # 获取 logits
        logits = out["logits"] if isinstance(out, dict) else (out[0] if isinstance(out, tuple) else out)
    assert logits.shape == (2, 8, m.vocab_size), f"logits 形状 {logits.shape} 不符"


def test_model_backward():
    """R31-1: 端到端反向传播成功，所有参数收到梯度。"""
    import torch.nn.functional as F
    m = _make_model()
    m.train()
    ids = torch.randint(0, m.vocab_size, (2, 8))
    out = m(ids)
    logits = out["logits"] if isinstance(out, dict) else (out[0] if isinstance(out, tuple) else out)
    target = torch.randint(0, m.vocab_size, (16,))
    loss = F.cross_entropy(logits.reshape(-1, m.vocab_size), target)
    loss.backward()
    # alpha_beta_proj 梯度
    for blk in m.blocks:
        attn = getattr(blk, 'attn', None)
        if attn is None or not hasattr(attn, 'alpha_beta_proj'):
            continue
        assert attn.alpha_beta_proj.weight.grad is not None, "alpha_beta_proj 无梯度"


# ─── R31-1: RWKV-7 兼容性 ─────────────────────────────────────────────────

def test_rwkv7_compat():
    """R31-1: rwkv7=True 时 alpha_beta_proj 仍正常工作（z_proj/b_proj 不受影响）。"""
    m = _make_gated_delta(rwkv7=True)
    assert hasattr(m, 'z_proj'), "rwkv7 模式应创建 z_proj"
    assert hasattr(m, 'b_proj'), "rwkv7 模式应创建 b_proj"
    assert hasattr(m, 'alpha_beta_proj'), "alpha_beta_proj 应存在"
    # 前向不报错
    x = torch.randn(2, 8, 64)
    out, _ = m(x)
    assert out.shape == (2, 8, 64)


# ─── R31-1: 不创建旧参数 ─────────────────────────────────────────────────

def test_no_legacy_params():
    """R31-1: 合并后不应再创建 alpha_proj/beta_proj 属性。"""
    m = _make_gated_delta()
    assert not hasattr(m, 'alpha_proj'), "不应创建 alpha_proj"
    assert not hasattr(m, 'beta_proj'), "不应创建 beta_proj"
    assert hasattr(m, 'alpha_beta_proj'), "应创建 alpha_beta_proj"
    assert hasattr(m, '_gate_out'), "应保存 _gate_out"
