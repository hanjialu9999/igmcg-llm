"""第二十三轮回归测试：GPAS 梯度保留激活缩放 + 配置校验修复 + cache parity 扩展。

覆盖：
- GPASNorm 参数创建/初始化/前向/梯度
- 向后兼容（默认关时行为不变）
- GPAS 前向输出差异
- cache parity（增量解码与全量前向一致）
- 与各 block_type（attn/ssm/hybrid）组合
- 与 zero_centered_norm 组合
- alpha_init 不同值
- state_dict 键存在性
- 配置校验修复（intra_hybrid_rope 须 alibi/num_heads/mixer 兼容，ratio 不校验当 disabled）
- cache parity 扩展（长序列/多层/混合层）
"""
import pytest
import torch
import torch.nn as nn
import math

from models.transformer import TransformerModel
from models.model_config import AttnConfig
from models.norms import GPASNorm
from models.mixers import SlidingWindowCausalSelfAttention


def _small(**over):
    kw = dict(vocab_size=200, embedding_dim=64, num_heads=4, num_layers=3,
              hidden_dim=128, max_seq_length=32)
    kw.update(over)
    return TransformerModel(**kw)


# ---------------------------------------------------------------------------
# GPASNorm 单元测试
# ---------------------------------------------------------------------------

def test_gpas_norm_init_default():
    """GPASNorm 默认 init_alpha=0.5 → raw=0 → sigmoid=0.5。"""
    g = GPASNorm()
    assert abs(g.gpas_raw.item()) < 1e-6, f"raw 应为 0，实际 {g.gpas_raw.item()}"
    alpha = torch.sigmoid(g.gpas_raw).item()
    assert abs(alpha - 0.5) < 1e-6, f"alpha 应为 0.5，实际 {alpha}"


def test_gpas_norm_init_custom():
    """GPASNorm 自定义 init_alpha=0.9 → sigmoid(raw)≈0.9。"""
    g = GPASNorm(init_alpha=0.9)
    alpha = torch.sigmoid(g.gpas_raw).item()
    assert abs(alpha - 0.9) < 1e-4, f"alpha 应为 0.9，实际 {alpha}"


def test_gpas_norm_forward():
    """GPASNorm forward 缩放激活：out = sigmoid(raw) * x。"""
    g = GPASNorm(init_alpha=0.5)
    x = torch.randn(2, 3, 4)
    out = g(x)
    expected = 0.5 * x
    assert torch.allclose(out, expected, atol=1e-6), "forward 应为 alpha * x"


def test_gpas_norm_gradient():
    """GPASNorm gpas_raw 有梯度。"""
    g = GPASNorm(init_alpha=0.5)
    x = torch.randn(2, 3, 4, requires_grad=True)
    out = g(x)
    loss = out.sum()
    loss.backward()
    assert g.gpas_raw.grad is not None, "gpas_raw 应有梯度"
    assert g.gpas_raw.grad.abs() > 0, "gpas_raw 梯度应非零"


# ---------------------------------------------------------------------------
# GPAS 模型集成测试
# ---------------------------------------------------------------------------

def test_gpas_disabled_by_default():
    """默认 gpas=False 时无 gpas1/gpas2。"""
    m = _small()
    blk = m.blocks[0]
    assert not blk.gpas_enabled, "gpas_enabled 应为 False"
    assert not hasattr(blk, 'gpas1'), "不应有 gpas1"
    assert not hasattr(blk, 'gpas2'), "不应有 gpas2"


def test_gpas_enabled_creates_modules():
    """gpas=True 时创建 gpas1/gpas2。"""
    m = _small(gpas=True)
    blk = m.blocks[0]
    assert blk.gpas_enabled, "gpas_enabled 应为 True"
    assert hasattr(blk, 'gpas1'), "应有 gpas1"
    assert hasattr(blk, 'gpas2'), "应有 gpas2"
    assert isinstance(blk.gpas1, GPASNorm), "gpas1 应为 GPASNorm"
    assert isinstance(blk.gpas2, GPASNorm), "gpas2 应为 GPASNorm"


def test_gpas_output_differs():
    """gpas=True 时输出与 gpas=False 不同（α=0.5 缩放激活）。"""
    torch.manual_seed(42)
    m1 = _small(gpas=False)
    torch.manual_seed(42)
    m2 = _small(gpas=True)
    x = torch.randint(0, 200, (2, 8))
    with torch.no_grad():
        out1 = m1(x)
        out2 = m2(x)
    diff = (out1 - out2).abs().max().item()
    assert diff > 0.01, f"gpas 开启应改变输出，diff={diff}"


def test_gpas_backward_compatible():
    """gpas=False 时两个模型输出完全等价（向后兼容）。"""
    torch.manual_seed(42)
    m1 = _small(gpas=False)
    torch.manual_seed(42)
    m2 = _small(gpas=False)
    x = torch.randint(0, 200, (2, 8))
    with torch.no_grad():
        out1 = m1(x)
        out2 = m2(x)
    assert torch.allclose(out1, out2), "gpas=False 应完全等价"


def test_gpas_gradient():
    """gpas=True 时每层 gpas1/gpas2.gpas_raw 有梯度。"""
    m = _small(gpas=True)
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    loss = out.sum()
    loss.backward()
    for i, blk in enumerate(m.blocks):
        assert blk.gpas1.gpas_raw.grad is not None, f"block {i} gpas1.gpas_raw 应有梯度"
        assert blk.gpas2.gpas_raw.grad is not None, f"block {i} gpas2.gpas_raw 应有梯度"


def test_gpas_cache_parity():
    """gpas=True cache parity：增量解码与全量前向末位 logits 一致。"""
    m = _small(gpas=True, num_layers=2)
    m.eval()
    torch.manual_seed(42)
    x = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        full = m(x, use_cache=False)
        out, past = m(x[:, :1], past_key_values=None, use_cache=True)
        for t in range(1, 8):
            out, past = m(x[:, t:t + 1], past_key_values=past, use_cache=True)
    diff = (full[:, -1, :] - out[:, -1, :]).abs().max().item()
    assert diff < 1e-4, f"gpas cache parity max_diff={diff:.2e} 超过 1e-4"


def test_gpas_with_ssm_block():
    """gpas=True 与 ssm 块组合正常工作。"""
    m = _small(gpas=True, layer_plan=['attn', 'ssm', 'attn'])
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    assert out.shape == (2, 8, 200), f"输出 shape 错误: {out.shape}"


def test_gpas_with_hybrid_block():
    """gpas=True 与 hybrid 块组合正常工作。"""
    m = _small(gpas=True, layer_plan=['attn', 'hybrid', 'attn'], mixer='attn_linear')
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    assert out.shape == (2, 8, 200), f"输出 shape 错误: {out.shape}"


def test_gpas_with_zero_centered_norm():
    """gpas=True 与 zero_centered_norm 组合正常工作。"""
    m = _small(gpas=True, zero_centered_norm=True)
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    assert out.shape == (2, 8, 200), f"输出 shape 错误: {out.shape}"


def test_gpas_alpha_init_near_one():
    """gpas_alpha_init=0.999 时 α≈1.0（行为接近无 GPAS）。"""
    m = _small(gpas=True, gpas_alpha_init=0.999)
    blk = m.blocks[0]
    alpha = torch.sigmoid(blk.gpas1.gpas_raw).item()
    assert alpha > 0.99, f"alpha 应接近 1.0，实际 {alpha}"


def test_gpas_state_dict_keys():
    """gpas=True 时 state_dict 包含 gpas1/gpas2 键。"""
    m = _small(gpas=True, num_layers=2)
    sd = m.state_dict()
    assert 'blocks.0.gpas1.gpas_raw' in sd, "state_dict 应包含 gpas1.gpas_raw"
    assert 'blocks.0.gpas2.gpas_raw' in sd, "state_dict 应包含 gpas2.gpas_raw"
    assert 'blocks.1.gpas1.gpas_raw' in sd, "state_dict 应包含 block1 gpas1.gpas_raw"


def test_gpas_all_layers_have_gpas():
    """gpas=True 时所有层都有 gpas1/gpas2。"""
    m = _small(gpas=True, num_layers=4)
    for i, blk in enumerate(m.blocks):
        assert hasattr(blk, 'gpas1'), f"block {i} 应有 gpas1"
        assert hasattr(blk, 'gpas2'), f"block {i} 应有 gpas2"


# ---------------------------------------------------------------------------
# 配置校验修复测试（第二十二轮 finding 修复）
# ---------------------------------------------------------------------------

def test_intra_hybrid_requires_alibi():
    """intra_hybrid_rope=True 须 alibi=True（否则报错）。"""
    with pytest.raises(ValueError, match="alibi"):
        AttnConfig(intra_hybrid_rope=True, alibi=False)


def test_intra_hybrid_with_alibi_ok():
    """intra_hybrid_rope=True + alibi=True 通过校验。"""
    cfg = AttnConfig(intra_hybrid_rope=True, alibi=True)
    assert cfg.intra_hybrid_rope is True


def test_intra_hybrid_mixer_compatibility():
    """intra_hybrid_rope=True 须 mixer in {'attn','attn_linear'}。"""
    with pytest.raises(ValueError, match="mixer"):
        AttnConfig(intra_hybrid_rope=True, alibi=True, mixer='linear')
    with pytest.raises(ValueError, match="mixer"):
        AttnConfig(intra_hybrid_rope=True, alibi=True, mixer='gated_delta')


def test_intra_hybrid_ratio_not_checked_when_disabled():
    """intra_hybrid_rope=False 时 ratio 不校验（允许任意值）。"""
    cfg = AttnConfig(intra_hybrid_rope=False, intra_hybrid_ratio=0.0)
    assert cfg.intra_hybrid_ratio == 0.0
    cfg2 = AttnConfig(intra_hybrid_rope=False, intra_hybrid_ratio=2.0)
    assert cfg2.intra_hybrid_ratio == 2.0


def test_intra_hybrid_num_heads_one_model():
    """intra_hybrid_rope=True + num_heads=1 时模型构造报错。"""
    with pytest.raises(ValueError, match="num_heads"):
        _small(intra_hybrid_rope=True, alibi=True, num_heads=1)


def test_intra_hybrid_ratio_one_direct_mixer():
    """ratio=1.0 致 nope_heads=num_heads 时直接构造 mixer 报错。"""
    with pytest.raises(ValueError, match="nope_heads"):
        SlidingWindowCausalSelfAttention(
            dim=64, num_heads=2, max_seq_length=32,
            intra_hybrid_rope=True, intra_hybrid_ratio=1.0)


# ---------------------------------------------------------------------------
# cache parity 扩展测试（cache 审查子代理建议补充）
# ---------------------------------------------------------------------------

def test_intra_hybrid_cache_parity_long_seq():
    """intra_hybrid_rope 长序列（32 步增量）cache parity。"""
    m = _small(intra_hybrid_rope=True, alibi=True, num_layers=2, max_seq_length=40)
    m.eval()
    torch.manual_seed(42)
    x = torch.randint(0, 200, (1, 32))
    with torch.no_grad():
        full = m(x, use_cache=False)
        out, past = m(x[:, :1], past_key_values=None, use_cache=True)
        for t in range(1, 32):
            out, past = m(x[:, t:t + 1], past_key_values=past, use_cache=True)
    diff = (full[:, -1, :] - out[:, -1, :]).abs().max().item()
    assert diff < 1e-4, f"长序列 cache parity max_diff={diff:.2e}"


def test_intra_hybrid_cache_parity_multi_layer():
    """intra_hybrid_rope 多层（4 层）cache parity。"""
    m = _small(intra_hybrid_rope=True, alibi=True, num_layers=4, max_seq_length=20)
    m.eval()
    torch.manual_seed(42)
    x = torch.randint(0, 200, (1, 12))
    with torch.no_grad():
        full = m(x, use_cache=False)
        out, past = m(x[:, :1], past_key_values=None, use_cache=True)
        for t in range(1, 12):
            out, past = m(x[:, t:t + 1], past_key_values=past, use_cache=True)
    diff = (full[:, -1, :] - out[:, -1, :]).abs().max().item()
    assert diff < 1e-4, f"多层 cache parity max_diff={diff:.2e}"


def test_intra_hybrid_cache_parity_hybrid():
    """intra_hybrid_rope 混合层（attn+hybrid+attn）cache parity。"""
    m = _small(intra_hybrid_rope=True, alibi=True, num_layers=3, max_seq_length=20,
               layer_plan=['attn', 'hybrid', 'attn'], mixer='attn_linear')
    m.eval()
    torch.manual_seed(42)
    x = torch.randint(0, 200, (1, 12))
    with torch.no_grad():
        full = m(x, use_cache=False)
        out, past = m(x[:, :1], past_key_values=None, use_cache=True)
        for t in range(1, 12):
            out, past = m(x[:, t:t + 1], past_key_values=past, use_cache=True)
    diff = (full[:, -1, :] - out[:, -1, :]).abs().max().item()
    assert diff < 1e-4, f"混合层 cache parity max_diff={diff:.2e}"
