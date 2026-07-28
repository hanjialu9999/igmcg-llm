"""第二十九轮回归测试：算子合并精简（低风险数值等价优化）。

覆盖：
- R29-1: convex_combine_scalar/linear 数学等价（g*h1+(1-g)*h2 == h2+g*(h1-h2)）
- R29-2: _apply_dim_wise_mask 数学等价（cos*mask+(1-mask) == (cos-1)*mask+1）
- R29-3: mask.repeat(2) 与 torch.cat([mask, mask]) 数值等价
- 边界情况：g=0/g=1/mask=0/mask=1/h1=h2 等极端值
"""
import torch
import torch.nn as nn
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.gates import (
    convex_combine_scalar, convex_combine_linear,
    apply_correction,
)
from models.rope import RotaryEmbedding


# ─── R29-1: convex_combine 数学等价 ────────────────────────────────────────

def test_convex_combine_scalar_equivalence():
    """R29-1: convex_combine_scalar 数学等价于原 g*h1+(1-g)*h2 公式。"""
    torch.manual_seed(42)
    param = nn.Parameter(torch.tensor([0.3]))
    h1 = torch.randn(2, 4, 8)
    h2 = torch.randn(2, 4, 8)

    out_new = convex_combine_scalar(param, h1, h2)

    # 手算原式
    g = torch.sigmoid(param)
    out_ref = g * h1 + (1.0 - g) * h2

    assert torch.allclose(out_new, out_ref, atol=1e-6), \
        f"convex_combine_scalar 不等价: max_diff={(out_new - out_ref).abs().max().item():.2e}"


def test_convex_combine_linear_equivalence():
    """R29-1: convex_combine_linear 数学等价于原公式。"""
    torch.manual_seed(42)
    linear = nn.Linear(8, 1)
    x = torch.randn(2, 4, 8)
    h1 = torch.randn(2, 4, 8)
    h2 = torch.randn(2, 4, 8)

    out_new = convex_combine_linear(linear, x, h1, h2)

    g = torch.sigmoid(linear(x))
    out_ref = g * h1 + (1.0 - g) * h2

    assert torch.allclose(out_new, out_ref, atol=1e-6), \
        f"convex_combine_linear 不等价: max_diff={(out_new - out_ref).abs().max().item():.2e}"


def test_convex_combine_scalar_boundary_g_zero():
    """R29-1: g=0 时（param=-inf 近似），输出应等于 h2。"""
    # 用大负数模拟 -inf
    param = torch.nn.Parameter(torch.tensor([-50.0]))
    h1 = torch.randn(2, 4)
    h2 = torch.randn(2, 4)
    out = convex_combine_scalar(param, h1, h2)
    assert torch.allclose(out, h2, atol=1e-5), \
        f"g≈0 时输出应等于 h2，max_diff={(out - h2).abs().max().item():.2e}"


def test_convex_combine_scalar_boundary_g_one():
    """R29-1: g=1 时（param=+inf 近似），输出应等于 h1。"""
    param = torch.nn.Parameter(torch.tensor([50.0]))
    h1 = torch.randn(2, 4)
    h2 = torch.randn(2, 4)
    out = convex_combine_scalar(param, h1, h2)
    assert torch.allclose(out, h1, atol=1e-5), \
        f"g≈1 时输出应等于 h1，max_diff={(out - h1).abs().max().item():.2e}"


def test_convex_combine_scalar_h1_equals_h2():
    """R29-1: h1==h2 时输出应等于 h1（数值稳定性）。"""
    param = torch.nn.Parameter(torch.tensor([0.0]))  # sigmoid=0.5
    h = torch.randn(2, 4)
    out = convex_combine_scalar(param, h, h.clone())
    assert torch.allclose(out, h, atol=1e-6), \
        f"h1==h2 时输出应等于 h，max_diff={(out - h).abs().max().item():.2e}"


def test_convex_combine_gradient_flow():
    """R29-1: 梯度回流正常（g 仍可学）。"""
    torch.manual_seed(42)
    param = torch.nn.Parameter(torch.tensor([0.0]))
    h1 = torch.randn(2, 4, requires_grad=True)
    h2 = torch.randn(2, 4, requires_grad=True)
    out = convex_combine_scalar(param, h1, h2)
    loss = out.sum()
    loss.backward()
    assert param.grad is not None, "param 应有梯度"
    assert param.grad.abs() > 0, "param 梯度应为非零"
    assert h1.grad is not None, "h1 应有梯度"
    assert h2.grad is not None, "h2 应有梯度"


def test_apply_correction_unchanged():
    """R29-1: apply_correction 未修改（已最简，作为基线验证）。"""
    torch.manual_seed(42)
    param = torch.nn.Parameter(torch.tensor([0.0]))
    h = torch.randn(2, 4)
    lh = torch.randn(2, 4)
    out = apply_correction(param, h, lh)

    cg = torch.sigmoid(param)
    expected = h + cg * (lh - h)
    assert torch.allclose(out, expected, atol=1e-7), \
        f"apply_correction 不正确: max_diff={(out - expected).abs().max().item():.2e}"


# ─── R29-2: _apply_dim_wise_mask 数学等价 ──────────────────────────────────

def test_apply_dim_wise_mask_equivalence():
    """R29-2: (cos-1)*mask+1 数学等价于原 cos*mask+(1-mask)。"""
    torch.manual_seed(42)
    rope = RotaryEmbedding(dim=16, dim_wise=True)
    # 直接调 _apply_dim_wise_mask
    cos = torch.randn(1, 1, 8, 16) * 0.5  # cos 范围 (-0.5, 0.5)
    sin = torch.randn(1, 1, 8, 16) * 0.5
    cos_new, sin_new = rope._apply_dim_wise_mask(cos.clone(), sin.clone())

    # 手算原式
    mask = torch.sigmoid(rope.dim_wise_logit)
    mask_full = torch.cat([mask, mask], dim=-1)
    cos_ref = cos * mask_full + (1.0 - mask_full)
    sin_ref = sin * mask_full

    assert torch.allclose(cos_new, cos_ref, atol=1e-6), \
        f"cos 不等价: max_diff={(cos_new - cos_ref).abs().max().item():.2e}"
    assert torch.allclose(sin_new, sin_ref, atol=1e-6), \
        f"sin 不等价: max_diff={(sin_new - sin_ref).abs().max().item():.2e}"


def test_apply_dim_wise_mask_mask_zero():
    """R29-2: mask=0 时（logit=-inf 近似），cos 应为 1（不旋转），sin 应为 0。"""
    rope = RotaryEmbedding(dim=16, dim_wise=True)
    # 手动设置 logit 为大负数模拟 mask=0
    with torch.no_grad():
        rope.dim_wise_logit.fill_(-50.0)
    cos = torch.randn(1, 1, 4, 16)
    sin = torch.randn(1, 1, 4, 16)
    cos_new, sin_new = rope._apply_dim_wise_mask(cos.clone(), sin.clone())
    assert torch.allclose(cos_new, torch.ones_like(cos), atol=1e-5), \
        f"mask=0 时 cos 应为 1，max_diff={(cos_new - 1).abs().max().item():.2e}"
    assert torch.allclose(sin_new, torch.zeros_like(sin), atol=1e-5), \
        f"mask=0 时 sin 应为 0，max_diff={sin_new.abs().max().item():.2e}"


def test_apply_dim_wise_mask_mask_one():
    """R29-2: mask=1 时（logit=+inf 近似），cos/sin 应不变。"""
    rope = RotaryEmbedding(dim=16, dim_wise=True)
    with torch.no_grad():
        rope.dim_wise_logit.fill_(50.0)
    cos = torch.randn(1, 1, 4, 16)
    sin = torch.randn(1, 1, 4, 16)
    cos_new, sin_new = rope._apply_dim_wise_mask(cos.clone(), sin.clone())
    assert torch.allclose(cos_new, cos, atol=1e-5), \
        f"mask=1 时 cos 应不变，max_diff={(cos_new - cos).abs().max().item():.2e}"
    assert torch.allclose(sin_new, sin, atol=1e-5), \
        f"mask=1 时 sin 应不变，max_diff={(sin_new - sin).abs().max().item():.2e}"


def test_apply_dim_wise_mask_gradient_flow():
    """R29-2: dim_wise_logit 梯度回流正常。"""
    torch.manual_seed(42)
    rope = RotaryEmbedding(dim=16, dim_wise=True)
    cos = torch.randn(1, 1, 4, 16, requires_grad=True)
    sin = torch.randn(1, 1, 4, 16, requires_grad=True)
    cos_new, sin_new = rope._apply_dim_wise_mask(cos, sin)
    loss = (cos_new.sum() + sin_new.sum())
    loss.backward()
    assert rope.dim_wise_logit.grad is not None, "dim_wise_logit 应有梯度"
    assert rope.dim_wise_logit.grad.abs().sum() > 0, "dim_wise_logit 梯度非零"


# ─── R29-3: repeat(2) 与 cat 数值等价 ──────────────────────────────────────

def test_repeat_cat_equivalence():
    """R29-3: mask.repeat(2) 与 torch.cat([mask, mask], dim=-1) 数值等价。"""
    torch.manual_seed(42)
    mask = torch.randn(8)
    via_repeat = mask.repeat(2)
    via_cat = torch.cat([mask, mask], dim=-1)
    assert torch.equal(via_repeat, via_cat), "repeat(2) 与 cat 数值不等"


def test_dim_wise_rope_forward_matches_legacy():
    """R29-2/3: dim_wise RoPE 端到端前向数值等价（init logit=0 → mask=0.5）。

    验证修改后 _apply_dim_wise_mask 路径在 forward 中行为正确。
    """
    torch.manual_seed(42)
    rope = RotaryEmbedding(dim=16, dim_wise=True)
    q = torch.randn(1, 2, 4, 16)
    k = torch.randn(1, 2, 4, 16)

    q_out, k_out = rope(q, k, start_pos=0)

    # 手算完整流程
    cos, sin = rope._get_cos_sin(0, 4, q.device, q.dtype)
    # dim_wise=True 时 logit init=0 → mask=0.5
    mask = torch.sigmoid(rope.dim_wise_logit)
    mask_full = torch.cat([mask, mask], dim=-1)
    cos_expected = cos * mask_full + (1.0 - mask_full)
    sin_expected = sin * mask_full

    # 应用 RoPE
    q_rot_dim = q[..., :16]
    d = 8
    q1, q2 = q_rot_dim[..., :d], q_rot_dim[..., d:]
    cos_half = cos_expected[..., :d]
    sin_half = sin_expected[..., :d]
    q_expected = torch.cat([
        q1 * cos_half - q2 * sin_half,
        q1 * sin_half + q2 * cos_half,
    ], dim=-1)

    assert torch.allclose(q_out, q_expected, atol=1e-6), \
        f"dim_wise RoPE 前向不等价: max_diff={(q_out - q_expected).abs().max().item():.2e}"


# ─── 整合：端到端前向不崩 ─────────────────────────────────────────────────

def test_model_forward_with_optimizations():
    """R29: 优化后模型前向仍正常。"""
    from models.transformer import TransformerModel
    m = TransformerModel(
        vocab_size=100, embedding_dim=64, num_heads=4, num_layers=2,
        hidden_dim=128, max_seq_length=32,
        mixer='attn_linear',  # 触发 convex_combine_scalar
        dim_wise_rope=True,    # 触发 _apply_dim_wise_mask
    )
    m.eval()
    x = torch.randint(0, 100, (2, 8))
    out = m(x)
    # forward(use_cache=False) 返回单 tensor (B, T, vocab)
    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == (2, 8, 100), f"输出 shape 错误: {out.shape}"


def test_model_generate_with_optimizations():
    """R29: 优化后生成仍确定性（需固定 torch.manual_seed）。"""
    torch.manual_seed(42)
    from models.transformer import TransformerModel
    m = TransformerModel(
        vocab_size=100, embedding_dim=64, num_heads=4, num_layers=2,
        hidden_dim=128, max_seq_length=32,
        mixer='attn_linear', dim_wise_rope=True,
    )
    m.eval()
    torch.manual_seed(0)
    out1 = m.generate([1, 2, 3], max_length=8, device='cpu', temperature=1.0, top_k=0)
    torch.manual_seed(0)
    out2 = m.generate([1, 2, 3], max_length=8, device='cpu', temperature=1.0, top_k=0)
    assert out1 == out2, "生成结果不确定"


def test_apply_correction_numerical_stability():
    """R29: apply_correction 在极端值下数值稳定。"""
    # 大值 h, lh
    param = torch.nn.Parameter(torch.tensor([0.0]))
    h = torch.randn(2, 4) * 100
    lh = torch.randn(2, 4) * 100
    out = apply_correction(param, h, lh)
    assert not torch.isnan(out).any(), "大值下出现 NaN"
    assert not torch.isinf(out).any(), "大值下出现 Inf"


def test_convex_combine_scalar_large_values():
    """R29: convex_combine_scalar 大值数值稳定。"""
    param = torch.nn.Parameter(torch.tensor([0.0]))
    h1 = torch.randn(2, 4) * 100
    h2 = torch.randn(2, 4) * 100
    out = convex_combine_scalar(param, h1, h2)
    assert not torch.isnan(out).any(), "大值下出现 NaN"
    assert not torch.isinf(out).any(), "大值下出现 Inf"
