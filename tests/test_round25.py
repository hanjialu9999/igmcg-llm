"""第二十五回合回归测试：ALiBi 可学斜率（alibi_learnable）。

覆盖：
- 配置校验（alibi_learnable=True 须 alibi=True）
- 默认关时向后兼容（buffer，非 Parameter）
- 开启时 alibi_slopes 为 Parameter（参与 autograd）
- 初始值与原 ALiBi 几何级数一致（精确向后兼容）
- 梯度正确回流到斜率
- shared_alibi + alibi_learnable 共享同一 Parameter 对象
- cache parity（增量解码与全量前向一致）
- state_dict 键存在性（Parameter 进 state_dict，buffer 不进）
"""
import pytest
import torch
import torch.nn as nn
import math

from models.transformer import TransformerModel
from models.model_config import AttnConfig, ModelConfig
from models.mixers import SlidingWindowCausalSelfAttention


def _small(**over):
    kw = dict(vocab_size=200, embedding_dim=64, num_heads=4, num_layers=3,
              hidden_dim=128, max_seq_length=32)
    kw.update(over)
    return TransformerModel(**kw)


# ---------------------------------------------------------------------------
# 配置校验
# ---------------------------------------------------------------------------

def test_alibi_learnable_requires_alibi():
    """alibi_learnable=True 须 alibi=True（否则报错）。"""
    with pytest.raises(ValueError, match="alibi_learnable"):
        AttnConfig(alibi_learnable=True, alibi=False)


def test_alibi_learnable_disabled_by_default():
    """alibi_learnable 默认 False。"""
    cfg = AttnConfig()
    assert cfg.alibi_learnable is False


def test_alibi_learnable_with_alibi_ok():
    """alibi_learnable=True + alibi=True 通过校验。"""
    cfg = AttnConfig(alibi_learnable=True, alibi=True)
    assert cfg.alibi_learnable is True


# ---------------------------------------------------------------------------
# 模块层测试
# ---------------------------------------------------------------------------

def test_alibi_slopes_is_buffer_when_not_learnable():
    """alibi_learnable=False 时 alibi_slopes 是 buffer（非 Parameter）。"""
    attn = SlidingWindowCausalSelfAttention(dim=64, num_heads=4, alibi=True,
                                            alibi_learnable=False)
    assert isinstance(alibi := dict(attn.named_buffers()).get('alibi_slopes'),
                      torch.Tensor)
    assert 'alibi_slopes' not in dict(attn.named_parameters())


def test_alibi_slopes_is_parameter_when_learnable():
    """alibi_learnable=True 时 alibi_slopes 是 Parameter（参与 autograd）。"""
    attn = SlidingWindowCausalSelfAttention(dim=64, num_heads=4, alibi=True,
                                            alibi_learnable=True)
    params = dict(attn.named_parameters())
    assert 'alibi_slopes' in params, "alibi_slopes 应为 Parameter"
    assert params['alibi_slopes'].requires_grad


def test_alibi_slopes_initial_values_match_original():
    """初始斜率与原 ALiBi 几何级数 m_h = 2^(-(h+1)/H * 8) 精确一致（向后兼容）。"""
    num_heads = 4
    attn = SlidingWindowCausalSelfAttention(dim=64, num_heads=num_heads, alibi=True,
                                            alibi_learnable=True)
    expected = torch.tensor([2.0 ** (-(h + 1) / num_heads * 8.0)
                             for h in range(num_heads)])
    assert torch.allclose(attn.alibi_slopes.data, expected, atol=1e-6)


def test_alibi_slopes_gradient_flows():
    """alibi_learnable=True 时梯度正确回流到 alibi_slopes。"""
    attn = SlidingWindowCausalSelfAttention(dim=64, num_heads=4, alibi=True,
                                            alibi_learnable=True)
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    loss = out.sum()
    loss.backward()
    assert attn.alibi_slopes.grad is not None, "alibi_slopes 应有梯度"
    assert attn.alibi_slopes.grad.shape == (4,)
    # 梯度应非零（位置偏置确实参与前向）
    assert attn.alibi_slopes.grad.abs().sum() > 0


def test_alibi_learnable_forward_matches_fixed_at_init():
    """init 时 alibi_learnable=True 与 False 前向输出一致（相同初始斜率）。"""
    torch.manual_seed(42)
    attn_fixed = SlidingWindowCausalSelfAttention(dim=64, num_heads=4, alibi=True,
                                                  alibi_learnable=False)
    torch.manual_seed(42)
    attn_learnable = SlidingWindowCausalSelfAttention(dim=64, num_heads=4, alibi=True,
                                                     alibi_learnable=True)
    # 同步 qkv/proj 权重（两次构造的随机初始化独立）
    attn_learnable.qkv.weight.data.copy_(attn_fixed.qkv.weight.data)
    attn_learnable.proj.weight.data.copy_(attn_fixed.proj.weight.data)

    x = torch.randn(1, 8, 64)
    out_fixed, _ = attn_fixed(x)
    out_learnable, _ = attn_learnable(x)
    assert torch.allclose(out_fixed, out_learnable, atol=1e-6), (
        "init 时两者前向应一致（相同初始斜率）")


# ---------------------------------------------------------------------------
# 模型层测试
# ---------------------------------------------------------------------------

def test_model_alibi_learnable_creates_parameters():
    """模型级 alibi_learnable=True 时所有 attn 层的 alibi_slopes 为 Parameter。"""
    model = _small(alibi=True, alibi_learnable=True)
    param_names = set(dict(model.named_parameters()).keys())
    for i, blk in enumerate(model.blocks):
        if hasattr(blk, 'attn') and hasattr(blk.attn, 'alibi_slopes'):
            key = f'blocks.{i}.attn.alibi_slopes'
            assert key in param_names, f"{key} 应为 Parameter"


def test_model_alibi_learnable_disabled_no_parameter():
    """alibi_learnable=False 时 alibi_slopes 不在 named_parameters 中。"""
    model = _small(alibi=True, alibi_learnable=False)
    param_names = set(dict(model.named_parameters()).keys())
    for i, blk in enumerate(model.blocks):
        if hasattr(blk, 'attn') and hasattr(blk.attn, 'alibi_slopes'):
            key = f'blocks.{i}.attn.alibi_slopes'
            assert key not in param_names, f"{key} 不应为 Parameter"


def test_shared_alibi_with_learnable_shares_parameter():
    """shared_alibi=True + alibi_learnable=True 时所有层共用同一 Parameter 对象。"""
    model = _small(alibi=True, alibi_learnable=True, shared_alibi=True)
    slopes_objs = []
    for blk in model.blocks:
        if hasattr(blk, 'attn') and hasattr(blk.attn, 'alibi_slopes'):
            slopes_objs.append(blk.attn.alibi_slopes)
    # 至少有 2 层有 alibi
    assert len(slopes_objs) >= 2
    first = slopes_objs[0]
    for s in slopes_objs[1:]:
        assert s is first, "shared_alibi 应让所有层共用同一 Parameter 对象"


def test_shared_alibi_learnable_gradient_shared():
    """shared_alibi=True + alibi_learnable=True 时梯度累积到同一 Parameter。"""
    model = _small(alibi=True, alibi_learnable=True, shared_alibi=True)
    x = torch.randint(0, 200, (2, 8))
    out = model(x)
    out.sum().backward()
    # 找到第一个有 alibi_slopes 的层
    for blk in model.blocks:
        if hasattr(blk, 'attn') and hasattr(blk.attn, 'alibi_slopes'):
            assert blk.attn.alibi_slopes.grad is not None
            assert blk.attn.alibi_slopes.grad.abs().sum() > 0
            break


# ---------------------------------------------------------------------------
# cache parity 测试
# ---------------------------------------------------------------------------

def test_alibi_learnable_cache_parity():
    """alibi_learnable=True 时增量解码与全量前向一致。"""
    torch.manual_seed(0)
    model = _small(alibi=True, alibi_learnable=True)
    model.eval()
    x = torch.randint(0, 200, (1, 12))

    with torch.no_grad():
        out_full = model(x)  # use_cache=False 时只返回 logits
        # 增量解码
        presents = None
        outs = []
        for t in range(12):
            out_t, presents = model(x[:, t:t+1], past_key_values=presents, use_cache=True)
            outs.append(out_t)
        out_inc = torch.cat(outs, dim=1)
    assert torch.allclose(out_full, out_inc, atol=1e-4), (
        f"alibi_learnable cache parity 失败：max_diff={ (out_full - out_inc).abs().max() }")


# ---------------------------------------------------------------------------
# state_dict 测试
# ---------------------------------------------------------------------------

def test_alibi_learnable_in_state_dict():
    """alibi_learnable=True 时 alibi_slopes 出现在 state_dict（Parameter 持久化）。"""
    model = _small(alibi=True, alibi_learnable=True)
    sd = model.state_dict()
    keys = [k for k in sd if k.endswith('alibi_slopes')]
    assert len(keys) > 0, "alibi_learnable=True 时 alibi_slopes 应在 state_dict"


def test_alibi_fixed_not_in_state_dict():
    """alibi_learnable=False 时 alibi_slopes 不在 state_dict（buffer persistent=False）。"""
    model = _small(alibi=True, alibi_learnable=False)
    sd = model.state_dict()
    keys = [k for k in sd if k.endswith('alibi_slopes')]
    assert len(keys) == 0, "alibi_learnable=False 时 alibi_slopes 不应在 state_dict"
