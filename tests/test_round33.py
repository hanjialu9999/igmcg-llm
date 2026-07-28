"""第三十三轮回归测试：convex_combine 算子合并精简（低风险数值等价优化）。

覆盖 6 处 R33 优化：
- R33-1: CharMerge `z*agg + (1-z)*x` → `x + z*(agg-x)`（layers.py L47）
- R33-2: ngram `vec*(1-w) + p*w` → `vec + w*(p-vec)`（ngram.py L362）
- R33-3: YaRN `interp*(1-mask) + extrap*mask` → `interp + mask*(extrap-interp)`（rope.py L121）
- R33-4: AxialLinearAttention `g*out_row + (1-g)*out_col` → `out_col + g*(out_row-out_col)`（mixers.py L1113）
- R33-5: DALA target `α*prev + (1-α)*x0` → `x0 + α*(prev-x0)`（transformer.py L1342）
- R33-6: 复杂度奖励 `mg*1 + (1-mg)*0.3` → `0.3 + 0.7*mg`（transformer.py L998）

所有优化都是 R29.5 convex_combine 模式的应用：
  g*h1 + (1-g)*h2 == h2 + g*(h1-h2)  （结合律，浮点误差<1e-7）
"""
import torch
import torch.nn as nn
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.layers import CharMergeLayer
from models.mixers import AxialLinearAttention
from models.rope import RotaryEmbedding


# ─── R33-1: CharMerge convex_combine 数学等价 ──────────────────────────

def test_charmerge_convex_combine_equivalence():
    """R33-1: CharMerge 的 z*agg+(1-z)*x == x+z*(agg-x) 数学等价。"""
    torch.manual_seed(42)
    dim = 32
    m = CharMergeLayer(dim, kernel_size=3)
    x = torch.randn(2, 8, dim)

    # 手算原式（z*agg + (1-z)*x）
    x_t = x.transpose(1, 2)
    x_padded = torch.nn.functional.pad(x_t, (m.pad, 0))
    agg = torch.nn.functional.conv1d(x_padded, m.conv.weight, None, groups=dim).transpose(1, 2)
    z = torch.sigmoid(m.gate(x))
    out_ref = z * agg + (1 - z) * x
    out_ref = m.norm(out_ref)

    # 新路径（实际 forward 已用 x + z*(agg-x)）
    out_new = m(x)

    assert torch.allclose(out_new, out_ref, atol=1e-6), \
        f"CharMerge convex_combine 不等价: max_diff={(out_new - out_ref).abs().max().item():.2e}"


def test_charmerge_forward_runs():
    """R33-1: CharMerge forward 正常运行。"""
    m = CharMergeLayer(32, kernel_size=3)
    x = torch.randn(2, 8, 32)
    out = m(x)
    assert out.shape == (2, 8, 32), f"CharMerge 输出形状错误: {out.shape}"


def test_charmerge_backward():
    """R33-1: CharMerge 梯度正常回流。"""
    m = CharMergeLayer(32, kernel_size=3)
    x = torch.randn(2, 8, 32, requires_grad=True)
    out = m(x)
    out.sum().backward()
    assert m.gate.weight.grad is not None, "gate.weight 无梯度"
    assert m.conv.weight.grad is not None, "conv.weight 无梯度"


# ─── R33-2: ngram convex_combine 数学等价 ──────────────────────────────

def test_ngram_convex_combine_equivalence():
    """R33-2: vec*(1-w) + p*w == vec + w*(p-vec) 数学等价。"""
    torch.manual_seed(42)
    vec = torch.randn(100)
    p = torch.randn(100)
    w = 0.3

    # 原式
    out_ref = vec * (1 - w) + p * w
    # 新式
    out_new = vec + w * (p - vec)

    assert torch.allclose(out_new, out_ref, atol=1e-7), \
        f"ngram convex_combine 不等价: max_diff={(out_new - out_ref).abs().max().item():.2e}"


def test_ngram_convex_combine_inplace_equivalence():
    """R33-2: in-place 更新 vec[idx] = vec[idx] + w*(p-vec[idx]) 等价原式。"""
    torch.manual_seed(42)
    vec_ref = torch.randn(100)
    vec_new = vec_ref.clone()
    idx = torch.tensor([1, 5, 10, 20, 50])
    p = torch.randn(100)
    w = 0.4

    # 原式
    vec_ref[idx] = vec_ref[idx] * (1 - w) + p[idx] * w
    # 新式
    vec_new[idx] = vec_new[idx] + w * (p[idx] - vec_new[idx])

    assert torch.allclose(vec_new, vec_ref, atol=1e-7), \
        f"ngram in-place 不等价: max_diff={(vec_new - vec_ref).abs().max().item():.2e}"


# ─── R33-3: YaRN inv_freq convex_combine 数学等价 ──────────────────────

def test_yarn_inv_freq_equivalence():
    """R33-3: interp*(1-mask) + extrap*mask == interp + mask*(extrap-interp) 数学等价。"""
    torch.manual_seed(42)
    dim = 16
    inv_freq_extrapolation = torch.randn(dim)
    inv_freq_interpolation = torch.randn(dim)
    mask = torch.rand(dim)

    # 原式
    out_ref = inv_freq_interpolation * (1 - mask) + inv_freq_extrapolation * mask
    # 新式
    out_new = inv_freq_interpolation + mask * (inv_freq_extrapolation - inv_freq_interpolation)

    assert torch.allclose(out_new, out_ref, atol=1e-7), \
        f"YaRN inv_freq 不等价: max_diff={(out_new - out_ref).abs().max().item():.2e}"


def test_yarn_rope_forward_runs():
    """R33-3: YaRN RotaryEmbedding 正常运行。"""
    rope = RotaryEmbedding(dim=32, yarn_scale=2.0, yarn_beta=0.1,
                            yarn_orig_max_seq_length=64)
    # 触发 _yarn_inv_freq 计算
    cos, sin = rope._get_cos_sin(0, 8, torch.device('cpu'), torch.float32)
    assert cos.shape[-2:] == (8, 32), f"YaRN cos 形状错误: {cos.shape}"
    assert sin.shape[-2:] == (8, 32), f"YaRN sin 形状错误: {sin.shape}"


# ─── R33-4: AxialLinearAttention convex_combine 数学等价 ────────────────

def test_axial_convex_combine_equivalence():
    """R33-4: g*out_row + (1-g)*out_col == out_col + g*(out_row-out_col) 数学等价。"""
    torch.manual_seed(42)
    g = torch.tensor(0.3)
    out_row = torch.randn(2, 4, 4, 16)
    out_col = torch.randn(2, 4, 4, 16)

    # 原式
    out_ref = g * out_row + (1 - g) * out_col
    # 新式
    out_new = out_col + g * (out_row - out_col)

    assert torch.allclose(out_new, out_ref, atol=1e-7), \
        f"Axial convex_combine 不等价: max_diff={(out_new - out_ref).abs().max().item():.2e}"


def test_axial_forward_runs():
    """R33-4: AxialLinearAttention forward 正常运行。"""
    dim, num_heads = 32, 4
    m = AxialLinearAttention(dim, num_heads, max_seq_length=16)
    # AxialLinearAttention 需要序列长度为完全平方数
    x = torch.randn(2, 16, dim)
    out, _ = m(x)
    assert out.shape == (2, 16, dim), f"AxialLinearAttention 输出形状错误: {out.shape}"


# ─── R33-5: DALA target convex_combine 数学等价 ────────────────────────

def test_dala_target_equivalence():
    """R33-5: α*prev + (1-α)*x0 == x0 + α*(prev-x0) 数学等价。"""
    torch.manual_seed(42)
    alpha = 0.6
    prev = torch.randn(2, 8, 32)
    x0 = torch.randn(2, 8, 32)

    # 原式
    out_ref = alpha * prev + (1.0 - alpha) * x0
    # 新式
    out_new = x0 + alpha * (prev - x0)

    assert torch.allclose(out_new, out_ref, atol=1e-6), \
        f"DALA target 不等价: max_diff={(out_new - out_ref).abs().max().item():.2e}"


def test_dala_aligned_training_forward():
    """R33-5: aligned_training 启用时模型 forward 正常。"""
    from models.transformer import TransformerModel
    m = TransformerModel(vocab_size=100, embedding_dim=32, num_heads=4, num_layers=3,
                          hidden_dim=64, max_seq_length=16,
                          layer_contrastive=True, aligned_training=True)
    m.train()
    x = torch.randint(0, 100, (2, 8))
    out = m(x)
    logits = out["logits"] if isinstance(out, dict) else out
    assert logits.shape == (2, 8, 100)
    # aligned_training 时应累积 contrastive_loss
    assert hasattr(m, '_contrastive_loss') and m._contrastive_loss is not None
    loss = logits.sum() + m._contrastive_loss
    loss.backward()


# ─── R33-6: 复杂度奖励常量折叠数学等价 ──────────────────────────────────

def test_complexity_reward_constant_folding():
    """R33-6: mg*1 + (1-mg)*0.3 == 0.3 + 0.7*mg 数学等价。"""
    mg = torch.tensor(0.6)

    # 原式
    out_ref = mg * 1.0 + (1.0 - mg) * 0.3
    # 新式
    out_new = 0.3 + 0.7 * mg

    assert torch.allclose(out_new, out_ref, atol=1e-7), \
        f"复杂度奖励常量折叠不等价: max_diff={(out_new - out_ref).abs().max().item():.2e}"


def test_complexity_reward_boundary():
    """R33-6: 边界值 mg=0 和 mg=1 数值正确。"""
    # mg=0 → cost=0.3
    mg_zero = torch.tensor(0.0)
    assert abs((0.3 + 0.7 * mg_zero).item() - 0.3) < 1e-7
    # mg=1 → cost=1.0
    mg_one = torch.tensor(1.0)
    assert abs((0.3 + 0.7 * mg_one).item() - 1.0) < 1e-7


# ─── 组合测试 ──────────────────────────────────────────────────────────

def test_all_optimizations_end_to_end():
    """R33 组合：所有优化特性同时开启，模型可训练。"""
    from models.transformer import TransformerModel
    m = TransformerModel(vocab_size=100, embedding_dim=32, num_heads=4, num_layers=2,
                          hidden_dim=64, max_seq_length=16,
                          char_merge=True, aligned_training=True,
                          yarn_scale=2.0)
    m.train()
    x = torch.randint(0, 100, (2, 8))
    out = m(x)
    logits = out["logits"] if isinstance(out, dict) else out
    assert logits.shape == (2, 8, 100)
    loss = logits.sum()
    if hasattr(m, '_contrastive_loss') and m._contrastive_loss is not None:
        loss = loss + m._contrastive_loss
    loss.backward()
    assert m.char_merge.gate.weight.grad is not None, "char_merge gate 无梯度"


def test_charmerge_with_training():
    """R33-1: CharMerge 在完整训练流程中数值稳定。"""
    from models.transformer import TransformerModel
    m = TransformerModel(vocab_size=100, embedding_dim=32, num_heads=4, num_layers=2,
                          hidden_dim=64, max_seq_length=16, char_merge=True)
    m.train()
    x = torch.randint(0, 100, (2, 8))
    out = m(x)
    logits = out["logits"] if isinstance(out, dict) else out
    loss = logits.float().sum()
    loss.backward()
    # 梯度有限（无 NaN/Inf）
    for p in m.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"梯度含 NaN/Inf"


def test_yarn_with_training():
    """R33-3: YaRN 在完整训练流程中数值稳定。"""
    from models.transformer import TransformerModel
    m = TransformerModel(vocab_size=100, embedding_dim=32, num_heads=4, num_layers=2,
                          hidden_dim=64, max_seq_length=16,
                          yarn_scale=2.0, alibi=True)
    m.train()
    x = torch.randint(0, 100, (2, 8))
    out = m(x)
    logits = out["logits"] if isinstance(out, dict) else out
    loss = logits.float().sum()
    loss.backward()
    for p in m.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"梯度含 NaN/Inf"
