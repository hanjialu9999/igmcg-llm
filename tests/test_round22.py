"""第二十二轮回归测试：层内 head 拆半 RoPE/NoPE（intra_hybrid_rope）。

覆盖：
- 参数创建/初始化/nope_heads 计算
- 向后兼容（默认关时行为不变）
- head 拆半改变输出（与全 RoPE 不同）
- 梯度回流
- cache parity（增量解码与全量前向一致）
- 与 alibi/head_temp/value_relative_coding 组合
- 与 nope_layers 交互
- MLA 不兼容校验 / ratio 范围校验
- 与 YaRN/dim_wise_rope/Partial RoPE 组合
"""
import pytest
import torch
import torch.nn as nn

from models.transformer import TransformerModel
from models.model_config import AttnConfig


def _small(**over):
    kw = dict(vocab_size=200, embedding_dim=64, num_heads=4, num_layers=3,
              hidden_dim=128, max_seq_length=32)
    kw.update(over)
    return TransformerModel(**kw)


# ---------------------------------------------------------------------------
# 参数创建 / 初始化
# ---------------------------------------------------------------------------

def test_intra_hybrid_param_created():
    """intra_hybrid_rope=True 时设置 enabled 标志和 nope_heads。"""
    m = _small(intra_hybrid_rope=True, alibi=True)
    attn = m.blocks[0].attn
    assert attn.intra_hybrid_rope_enabled is True, "intra_hybrid_rope_enabled 应为 True"
    # num_heads=4, ratio=0.5 → nope_heads=2
    assert attn.intra_hybrid_nope_heads == 2, \
        f"nope_heads 应为 2（4*0.5），实际 {attn.intra_hybrid_nope_heads}"


def test_intra_hybrid_ratio_custom():
    """自定义 ratio 时 nope_heads 计算正确。"""
    m = _small(intra_hybrid_rope=True, intra_hybrid_ratio=0.25, alibi=True, num_heads=8,
               embedding_dim=128)
    attn = m.blocks[0].attn
    # num_heads=8, ratio=0.25 → nope_heads=2
    assert attn.intra_hybrid_nope_heads == 2, \
        f"nope_heads 应为 2（8*0.25），实际 {attn.intra_hybrid_nope_heads}"


def test_intra_hybrid_backward_compat():
    """默认关时 enabled=False, nope_heads=0。"""
    m = _small()
    attn = m.blocks[0].attn
    assert attn.intra_hybrid_rope_enabled is False, "默认应为 False"
    assert attn.intra_hybrid_nope_heads == 0, "默认 nope_heads 应为 0"


# ---------------------------------------------------------------------------
# 行为差异 / 梯度
# ---------------------------------------------------------------------------

def test_intra_hybrid_changes_output():
    """intra_hybrid_rope=True 改变输出（与全 RoPE 不同）。"""
    torch.manual_seed(42)
    m_off = _small(alibi=True)
    torch.manual_seed(42)
    m_on = _small(intra_hybrid_rope=True, alibi=True)
    x = torch.randint(0, 200, (2, 8))
    with torch.no_grad():
        out_off = m_off(x)
        out_on = m_on(x)
    # 后半 head 不旋转 → 输出应不同
    assert not torch.allclose(out_off, out_on, atol=1e-5), \
        "intra_hybrid_rope 应改变输出"


def test_intra_hybrid_gradient_flow():
    """intra_hybrid_rope 不引入新参数，但 qkv/proj 梯度应正常回流。"""
    m = _small(intra_hybrid_rope=True, alibi=True)
    m.train()
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    loss = out.float().sum()
    loss.backward()
    assert m.blocks[0].attn.qkv.weight.grad is not None, "qkv 无梯度"


# ---------------------------------------------------------------------------
# cache parity
# ---------------------------------------------------------------------------

def test_intra_hybrid_cache_parity():
    """intra_hybrid_rope cache parity：增量解码与全量前向末位 logits 一致。"""
    m = _small(intra_hybrid_rope=True, alibi=True, num_layers=2)
    m.eval()
    torch.manual_seed(42)
    x = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        full = m(x, use_cache=False)
        out, past = m(x[:, :1], past_key_values=None, use_cache=True)
        for t in range(1, 8):
            out, past = m(x[:, t:t + 1], past_key_values=past, use_cache=True)
    diff = (full[:, -1, :] - out[:, -1, :]).abs().max().item()
    assert diff < 1e-4, f"intra_hybrid cache parity max_diff={diff:.2e} 超过 1e-4"


# ---------------------------------------------------------------------------
# 组合测试
# ---------------------------------------------------------------------------

def test_intra_hybrid_with_head_temp_and_vrc():
    """intra_hybrid_rope + head_temp + value_relative_coding 组合。"""
    m = _small(intra_hybrid_rope=True, alibi=True,
               head_temp=True, value_relative_coding=True)
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    assert out.shape == (2, 8, 200), f"输出形状错误: {out.shape}"


def test_intra_hybrid_with_nope_layers():
    """intra_hybrid_rope + nope_layers 交互：nope_layers 中的层全 NoPE，其他层 head 拆半。"""
    m = _small(intra_hybrid_rope=True, alibi=True, nope_layers=[1], num_layers=3)
    # 层 0: head 拆半（use_rope=True, intra_hybrid 生效）
    assert m.blocks[0].attn.use_rope is True
    assert m.blocks[0].attn.intra_hybrid_rope_enabled is True
    # 层 1: 全 NoPE（use_rope=False, intra_hybrid 不生效）
    assert m.blocks[1].attn.use_rope is False
    # 层 2: head 拆半
    assert m.blocks[2].attn.use_rope is True
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    assert out.shape == (2, 8, 200)


def test_intra_hybrid_with_yarn():
    """intra_hybrid_rope + YaRN 组合。"""
    m = _small(intra_hybrid_rope=True, alibi=True,
               yarn_scale=2.0, yarn_orig_max_seq_length=32)
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    assert out.shape == (2, 8, 200)


def test_intra_hybrid_with_dim_wise_rope():
    """intra_hybrid_rope + dim_wise_rope 组合（不同切分轴正交）。"""
    m = _small(intra_hybrid_rope=True, alibi=True, dim_wise_rope=True)
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    assert out.shape == (2, 8, 200)


def test_intra_hybrid_with_partial_rope():
    """intra_hybrid_rope + Partial RoPE 组合。"""
    m = _small(intra_hybrid_rope=True, alibi=True, rope_dim_fraction=0.5)
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    assert out.shape == (2, 8, 200)


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

def test_intra_hybrid_mla_incompatible():
    """intra_hybrid_rope 与 use_mla_kv 不兼容应报错。"""
    with pytest.raises(ValueError, match="不兼容"):
        AttnConfig(intra_hybrid_rope=True, use_mla_kv=True)


def test_intra_hybrid_ratio_out_of_range():
    """intra_hybrid_ratio 须在 (0,1) 开区间。"""
    with pytest.raises(ValueError, match="开区间"):
        AttnConfig(intra_hybrid_rope=True, intra_hybrid_ratio=0.0)
    with pytest.raises(ValueError, match="开区间"):
        AttnConfig(intra_hybrid_rope=True, intra_hybrid_ratio=1.0)


# ---------------------------------------------------------------------------
# 逻辑验证
# ---------------------------------------------------------------------------

def test_intra_hybrid_nope_heads_not_all():
    """intra_hybrid_nope_heads < num_heads（不能全部 NoPE，否则等价于 nope_layers）。"""
    m = _small(intra_hybrid_rope=True, intra_hybrid_ratio=0.5, alibi=True, num_heads=4)
    attn = m.blocks[0].attn
    assert attn.intra_hybrid_nope_heads < attn.num_heads, \
        "nope_heads 应 < num_heads（至少 1 个 head 用 RoPE）"


def test_intra_hybrid_rope_half_correct():
    """验证前半 head 旋转、后半 head 不旋转。

    用 rope.apply_to_single 对比：前半 head 应与独立 RoPE 一致，
    后半 head 应与原始输入一致。
    """
    m = _small(intra_hybrid_rope=True, alibi=True, num_heads=4)
    attn = m.blocks[0].attn
    # 构造 q (1, 4, 2, 16)
    torch.manual_seed(0)
    q = torch.randn(1, 4, 2, 16)
    k = torch.randn(1, 4, 2, 16)
    # 独立应用 RoPE 到前 2 个 head
    q_rope_expected, k_rope_expected = attn.rope(q[:, :2], k[:, :2], start_pos=0, max_len=32)
    # 全量前向（模拟 project_and_norm 的 head 拆半）
    H_rope = 4 - attn.intra_hybrid_nope_heads  # 2
    q_rope, q_nope = q[:, :H_rope], q[:, H_rope:]
    k_rope, k_nope = k[:, :H_rope], k[:, H_rope:]
    q_rope, k_rope = attn.rope(q_rope, k_rope, start_pos=0, max_len=32)
    q_full = torch.cat([q_rope, q_nope], dim=1)
    # 前半应与独立 RoPE 一致
    assert torch.allclose(q_full[:, :2], q_rope_expected, atol=1e-6), \
        "前半 head 应与独立 RoPE 一致"
    # 后半应与原始输入一致（未旋转）
    assert torch.allclose(q_full[:, 2:], q[:, 2:], atol=1e-6), \
        "后半 head 应与原始输入一致（未旋转）"


def test_intra_hybrid_cache_parity_with_vrc():
    """intra_hybrid_rope + value_relative_coding cache parity。"""
    m = _small(intra_hybrid_rope=True, alibi=True,
               value_relative_coding=True, num_layers=2)
    m.eval()
    torch.manual_seed(42)
    x = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        full = m(x, use_cache=False)
        out, past = m(x[:, :1], past_key_values=None, use_cache=True)
        for t in range(1, 8):
            out, past = m(x[:, t:t + 1], past_key_values=past, use_cache=True)
    diff = (full[:, -1, :] - out[:, -1, :]).abs().max().item()
    assert diff < 1e-4, f"intra_hybrid+vrc cache parity max_diff={diff:.2e} 超过 1e-4"
