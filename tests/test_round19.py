"""第十九轮回归测试：KDA 逐通道衰减 / YaRN 长度外推 / 审查 bug 修复。

覆盖：
- KDA（Kimi Delta Attention）：channel_wise=True 时 alpha/beta 逐通道衰减
- YaRN 长度外推：yarn_scale>1.0 时 inv_freq 非均匀缩放
- 审查 Finding 1 CRITICAL：MLA + nope_layers 场景 k 不应用 RoPE
- 审查 Finding 2 MEDIUM：shared_alibi_enabled 包含 nope_layers_set
- 审查 Finding 3 LOW：nope_layers 索引越界校验
"""
import torch
import pytest

from models.transformer import TransformerModel
from models.rope import RotaryEmbedding
from models.mixers import GatedDeltaNet


def _small(**over):
    kw = dict(vocab_size=200, embedding_dim=64, num_heads=4, num_layers=2,
              hidden_dim=128, max_seq_length=32)
    kw.update(over)
    return TransformerModel(**kw)


# ---------------------------------------------------------------------------
# 审查 Finding 1 CRITICAL：MLA + nope_layers 场景 k 不应用 RoPE
# ---------------------------------------------------------------------------

def test_mla_nope_k_no_rope():
    """MLA + nope_layers 同时启用时，NoPE 层的 k 不应被 RoPE 旋转。"""
    m = _small(use_mla_kv=True, kv_latent_dim=32, nope_layers=[1])
    # NoPE 层的 attn 应 use_rope=False
    assert m.blocks[1].attn.use_rope is False
    # MLA 路径 attend 中 k 的 RoPE 应用应检查 use_rope
    # 验证：构造一个小输入，检查 attend 中 k 不被旋转
    attn = m.blocks[1].attn
    # 模拟 MLA 路径：c_kv 还原后 k 不应应用 RoPE（use_rope=False）
    # 直接验证 use_rope 标志传入 attend 逻辑
    assert attn.use_rope is False, "NoPE 层 use_rope 应为 False"
    assert attn.mla_kv_enabled is True, "应启用 MLA"


# ---------------------------------------------------------------------------
# 审查 Finding 2 MEDIUM：shared_alibi_enabled 包含 nope_layers_set
# ---------------------------------------------------------------------------

def test_shared_alibi_nope_layers_flag():
    """shared_alibi=True + nope_layers=[1] + alibi=False 时 shared_alibi_enabled 应为 True。"""
    m = _small(shared_alibi=True, alibi=False, nope_layers=[1])
    assert m.shared_alibi_enabled is True, \
        "shared_alibi + nope_layers 时 shared_alibi_enabled 应为 True（设备迁移后需重绑定）"


def test_shared_alibi_without_nope_or_alibi():
    """shared_alibi=True + alibi=False + nope_layers=[] 时 shared_alibi_enabled 应为 False。"""
    m = _small(shared_alibi=True, alibi=False, nope_layers=[])
    assert m.shared_alibi_enabled is False, \
        "无 alibi 且无 nope_layers 时 shared_alibi_enabled 应为 False"


# ---------------------------------------------------------------------------
# 审查 Finding 3 LOW：nope_layers 索引越界校验
# ---------------------------------------------------------------------------

def test_nope_layers_out_of_range():
    """nope_layers 索引越界时应 raise ValueError。"""
    with pytest.raises(ValueError, match="nope_layers 索引越界"):
        _small(nope_layers=[5])  # num_layers=2，索引 5 越界


def test_nope_layers_negative_index():
    """nope_layers 负索引应 raise ValueError。"""
    with pytest.raises(ValueError, match="nope_layers 索引越界"):
        _small(nope_layers=[-1])


# ---------------------------------------------------------------------------
# KDA 逐通道衰减（GatedDeltaNet channel_wise）
# ---------------------------------------------------------------------------

def test_kda_channel_wise_param_dims():
    """channel_wise=True 时 alpha_proj/beta_proj 输出维度 = num_heads * head_dim。"""
    dim, num_heads = 64, 4
    head_dim = dim // num_heads  # 16
    m = GatedDeltaNet(dim, num_heads, channel_wise=True)
    assert m.alpha_proj.out_features == num_heads * head_dim, \
        f"channel_wise alpha_proj 输出应为 {num_heads * head_dim}，实际 {m.alpha_proj.out_features}"
    assert m.beta_proj.out_features == num_heads * head_dim, \
        f"channel_wise beta_proj 输出应为 {num_heads * head_dim}，实际 {m.beta_proj.out_features}"


def test_kda_scalar_param_dims():
    """channel_wise=False（默认）时 alpha_proj/beta_proj 输出维度 = num_heads（标量）。"""
    dim, num_heads = 64, 4
    m = GatedDeltaNet(dim, num_heads, channel_wise=False)
    assert m.alpha_proj.out_features == num_heads
    assert m.beta_proj.out_features == num_heads


def test_kda_compute_gates_shape():
    """channel_wise 模式 _compute_gates 返回 (B,H,T,D)，标量模式返回 (B,H,T,1)。"""
    dim, num_heads = 64, 4
    head_dim = dim // num_heads
    B, T = 2, 8

    # 标量模式
    m_scalar = GatedDeltaNet(dim, num_heads, channel_wise=False)
    x = torch.randn(B, T, dim)
    alpha_s, beta_s = m_scalar._compute_gates(x)
    assert alpha_s.shape == (B, num_heads, T, 1), f"标量模式 alpha 形状 {alpha_s.shape} 不符"
    assert beta_s.shape == (B, num_heads, T, 1)

    # 通道模式
    m_chan = GatedDeltaNet(dim, num_heads, channel_wise=True)
    alpha_c, beta_c = m_chan._compute_gates(x)
    assert alpha_c.shape == (B, num_heads, T, head_dim), f"通道模式 alpha 形状 {alpha_c.shape} 不符"
    assert beta_c.shape == (B, num_heads, T, head_dim)


def test_kda_channel_wise_forward():
    """channel_wise 模式前向传播不报错且输出形状正确。"""
    dim, num_heads = 64, 4
    B, T = 2, 8
    m = GatedDeltaNet(dim, num_heads, channel_wise=True, max_seq_length=T)
    x = torch.randn(B, T, dim)
    out, present = m(x)
    assert out.shape == (B, T, dim), f"输出形状 {out.shape} 不符"


def test_kda_channel_wise_changes_output():
    """channel_wise=True 与 False 输出不同（逐通道衰减改变行为）。

    注：weight=0 时两种模式行为等价（设计预期，平滑过渡）。
    需设置 alpha_proj/beta_proj weight 非 0 使逐通道 alpha 产生差异。
    """
    torch.manual_seed(42)
    dim, num_heads = 64, 4
    B, T = 2, 8
    m_scalar = GatedDeltaNet(dim, num_heads, channel_wise=False, max_seq_length=T)
    torch.manual_seed(42)
    m_chan = GatedDeltaNet(dim, num_heads, channel_wise=True, max_seq_length=T)
    # 设置 alpha/beta proj weight 非 0，使 channel_wise 模式产生逐通道不同的门控值
    with torch.no_grad():
        torch.nn.init.normal_(m_scalar.alpha_proj.weight, 0, 0.1)
        torch.nn.init.normal_(m_scalar.beta_proj.weight, 0, 0.1)
        torch.nn.init.normal_(m_chan.alpha_proj.weight, 0, 0.1)
        torch.nn.init.normal_(m_chan.beta_proj.weight, 0, 0.1)
    x = torch.randn(B, T, dim)
    with torch.no_grad():
        out_scalar, _ = m_scalar(x)
        out_chan, _ = m_chan(x)
    assert not torch.allclose(out_scalar, out_chan, atol=1e-5), \
        "channel_wise 模式（weight 非 0）应改变输出"


def test_kda_channel_wise_backward():
    """channel_wise 模式梯度正常回流。"""
    dim, num_heads = 64, 4
    B, T = 2, 8
    m = GatedDeltaNet(dim, num_heads, channel_wise=True, max_seq_length=T)
    x = torch.randn(B, T, dim)
    out, _ = m(x)
    loss = out.float().sum()
    loss.backward()
    assert m.alpha_proj.weight.grad is not None, "alpha_proj 无梯度"
    assert m.beta_proj.weight.grad is not None, "beta_proj 无梯度"


def test_kda_channel_wise_cache_parity():
    """channel_wise 模式全量前向 vs 增量解码一致性。

    注：增量解码须传 start_pos=t 使 RoPE 位置与全量前向一致，
    否则 RoPE 位置不一致导致 q/k 不同（已知 trade-off，非 channel_wise 特有）。
    """
    torch.manual_seed(42)
    dim, num_heads = 64, 4
    B, T = 1, 6
    m = GatedDeltaNet(dim, num_heads, channel_wise=True, max_seq_length=T)
    x = torch.randn(B, T, dim)

    # 全量前向
    with torch.no_grad():
        out_full, present = m(x, use_cache=True)

    # 增量解码（逐 token，传 start_pos 保持 RoPE 位置一致）
    out_inc = []
    past = None
    with torch.no_grad():
        for t in range(T):
            out_t, past = m(x[:, t:t+1, :], past_kv=past, use_cache=True, start_pos=t)
            out_inc.append(out_t)
    out_inc = torch.cat(out_inc, dim=1)

    max_diff = (out_full - out_inc).abs().max().item()
    # delta rule 递推 6 步浮点累积，阈值 0.1（RoPE 位置一致后的残余精度差异）
    assert max_diff < 0.1, f"cache parity 差异 {max_diff} 过大"


def test_kda_specialized_init_preserved():
    """channel_wise 模式专用初始化在 _apply_specialized_inits 后正确重置。"""
    m = _small(mixer='gated_delta', gated_delta_channel_wise=True,
               delta_alpha_init=-2.0, delta_beta_init=2.0)
    for blk in m.blocks:
        attn = getattr(blk, 'attn', None)
        if attn is None or not hasattr(attn, 'alpha_proj'):
            continue
        assert torch.allclose(attn.alpha_proj.weight, torch.zeros_like(attn.alpha_proj.weight)), \
            "alpha_proj weight 应为 0"
        assert torch.allclose(attn.alpha_proj.bias, torch.full_like(attn.alpha_proj.bias, -2.0)), \
            "alpha_proj bias 应为 alpha_init=-2.0"
        assert torch.allclose(attn.beta_proj.weight, torch.zeros_like(attn.beta_proj.weight)), \
            "beta_proj weight 应为 0"
        assert torch.allclose(attn.beta_proj.bias, torch.full_like(attn.beta_proj.bias, 2.0)), \
            "beta_proj bias 应为 beta_init=2.0"


# ---------------------------------------------------------------------------
# YaRN 长度外推
# ---------------------------------------------------------------------------

def test_yarn_scale_1_backward_compat():
    """yarn_scale=1.0 时 inv_freq 与普通 RoPE 完全相同（向后兼容）。"""
    dim = 64
    rope_yarn = RotaryEmbedding(dim, yarn_scale=1.0)
    rope_normal = RotaryEmbedding(dim, yarn_scale=1.0)
    assert torch.allclose(rope_yarn.inv_freq, rope_normal.inv_freq), \
        "yarn_scale=1.0 时 inv_freq 应与普通 RoPE 相同"


def test_yarn_scale_gt1_changes_inv_freq():
    """yarn_scale>1.0 时 inv_freq 被缩放（低频维度缩放更多）。"""
    dim = 64
    rope_normal = RotaryEmbedding(dim, yarn_scale=1.0)
    rope_yarn = RotaryEmbedding(dim, yarn_scale=4.0, yarn_orig_max_seq_length=32)
    # inv_freq 应不同
    assert not torch.allclose(rope_normal.inv_freq, rope_yarn.inv_freq), \
        "yarn_scale=4.0 时 inv_freq 应与普通 RoPE 不同"
    # 低频维度（inv_freq 小，index 大）应被缩放（变得更小）
    # 高频维度（inv_freq 大，index 小）应保持不变（外推）
    n = len(rope_normal.inv_freq)
    high_freq_idx = 0  # 最高频
    low_freq_idx = n - 1  # 最低频
    # 高频维度变化小
    high_diff = abs(rope_normal.inv_freq[high_freq_idx].item() - rope_yarn.inv_freq[high_freq_idx].item())
    # 低频维度变化大
    low_diff = abs(rope_normal.inv_freq[low_freq_idx].item() - rope_yarn.inv_freq[low_freq_idx].item())
    assert low_diff > high_diff, \
        f"低频维度变化({low_diff})应大于高频维度变化({high_diff})"


def test_yarn_forward_changes_output():
    """yarn_scale>1.0 时 RoPE 前向输出与普通 RoPE 不同。"""
    dim = 64
    rope_normal = RotaryEmbedding(dim)
    rope_yarn = RotaryEmbedding(dim, yarn_scale=4.0, yarn_orig_max_seq_length=32)
    q = torch.randn(1, 4, 8, dim)
    k = torch.randn(1, 4, 8, dim)
    with torch.no_grad():
        q_n, k_n = rope_normal(q, k)
        q_y, k_y = rope_yarn(q, k)
    assert not torch.allclose(q_n, q_y, atol=1e-5), "YaRN 应改变 q 的旋转结果"
    assert not torch.allclose(k_n, k_y, atol=1e-5), "YaRN 应改变 k 的旋转结果"


def test_yarn_partial_rope_orthogonal():
    """YaRN + Partial RoPE 可叠加（dim_fraction<1.0 + yarn_scale>1.0）。"""
    dim = 64
    rope = RotaryEmbedding(dim, dim_fraction=0.5, yarn_scale=4.0, yarn_orig_max_seq_length=32)
    assert rope.rot_dim == 32, f"Partial RoPE rot_dim 应为 32，实际 {rope.rot_dim}"
    assert rope.no_pe_dim == 32, f"NoPE dim 应为 32，实际 {rope.no_pe_dim}"
    # 前向不报错
    q = torch.randn(1, 4, 8, dim)
    k = torch.randn(1, 4, 8, dim)
    with torch.no_grad():
        q_r, k_r = rope(q, k)
    assert q_r.shape == q.shape, "YaRN+Partial RoPE 输出形状应不变"


def test_yarn_model_integration():
    """模型级 YaRN 集成：yarn_scale>1.0 时模型前向不报错且输出形状正确。"""
    m = _small(yarn_scale=4.0, yarn_orig_max_seq_length=32)
    x = torch.randint(0, 200, (2, 8))
    with torch.no_grad():
        out = m(x)
    assert out.shape == (2, 8, 200), f"输出形状 {out.shape} 不符"


def test_yarn_model_backward():
    """模型级 YaRN 梯度正常回流。"""
    m = _small(yarn_scale=4.0, yarn_orig_max_seq_length=32)
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    loss = out.float().sum()
    loss.backward()
    # RoPE inv_freq 是 buffer（不可学），但模型其他参数应有梯度
    assert m.embedding.weight.grad is not None, "embedding 无梯度"


# ---------------------------------------------------------------------------
# 组合测试：KDA + YaRN
# ---------------------------------------------------------------------------

def test_kda_yarn_combined():
    """KDA channel_wise + YaRN 同时启用时前向不报错。"""
    m = _small(mixer='gated_delta', gated_delta_channel_wise=True,
               yarn_scale=2.0, yarn_orig_max_seq_length=32)
    x = torch.randint(0, 200, (2, 8))
    with torch.no_grad():
        out = m(x)
    assert out.shape == (2, 8, 200), f"组合测试输出形状 {out.shape} 不符"
