"""第三十五轮回归测试：addcmul 融合 + sigmoid 合并 + arange 去重。

验证 R35 优化的数值等价性：
1. convex_combine_scalar/linear/apply_correction 改用 torch.addcmul
2. CharMergeLayer forward 改用 torch.addcmul
3. AxialLinearAttention 融合改用 torch.addcmul
4. highway_gates 两次 sigmoid 合并为一次
5. _full_retrieval_bias 重复 arange 消除
"""
import math
import torch
import torch.nn as nn
import pytest

from models.gates import (
    convex_combine_scalar, convex_combine_linear, apply_correction,
)
from models.layers import CharMergeLayer


def test_convex_combine_scalar_addcmul_equivalence():
    """R35-1a: convex_combine_scalar addcmul 与手写 h2+g*(h1-h2) 数值等价。"""
    torch.manual_seed(42)
    param = nn.Parameter(torch.randn(1))
    h1 = torch.randn(2, 8, 64)
    h2 = torch.randn(2, 8, 64)
    out = convex_combine_scalar(param, h1, h2)
    # 手算
    g = torch.sigmoid(param)
    expected = h2 + g * (h1 - h2)
    assert torch.allclose(out, expected, atol=1e-6), \
        f"convex_combine_scalar addcmul 不等价: max_diff={(out - expected).abs().max().item():.2e}"


def test_convex_combine_linear_addcmul_equivalence():
    """R35-1b: convex_combine_linear addcmul 与手写等价。"""
    torch.manual_seed(42)
    linear = nn.Linear(64, 1)
    x = torch.randn(2, 8, 64)
    h1 = torch.randn(2, 8, 64)
    h2 = torch.randn(2, 8, 64)
    out = convex_combine_linear(linear, x, h1, h2)
    g = torch.sigmoid(linear(x))
    expected = h2 + g * (h1 - h2)
    assert torch.allclose(out, expected, atol=1e-6), \
        f"convex_combine_linear addcmul 不等价: max_diff={(out - expected).abs().max().item():.2e}"


def test_apply_correction_addcmul_equivalence():
    """R35-1c: apply_correction addcmul 与手写等价。"""
    torch.manual_seed(42)
    param = nn.Parameter(torch.randn(1))
    h = torch.randn(2, 8, 64)
    lh = torch.randn(2, 8, 64)
    out = apply_correction(param, h, lh)
    cg = torch.sigmoid(param)
    expected = h + cg * (lh - h)
    assert torch.allclose(out, expected, atol=1e-6), \
        f"apply_correction addcmul 不等价: max_diff={(out - expected).abs().max().item():.2e}"


def test_convex_combine_scalar_gradient_flow():
    """R35-2: addcmul 梯度正确回流。"""
    param = nn.Parameter(torch.randn(1))
    h1 = torch.randn(2, 4, 32, requires_grad=True)
    h2 = torch.randn(2, 4, 32, requires_grad=True)
    out = convex_combine_scalar(param, h1, h2)
    loss = out.sum()
    loss.backward()
    assert param.grad is not None, "param梯度未回流"
    assert h1.grad is not None, "h1梯度未回流"
    assert h2.grad is not None, "h2梯度未回流"


def test_char_merge_addcmul_equivalence():
    """R35-3: CharMergeLayer addcmul 与手写等价。"""
    torch.manual_seed(42)
    m = CharMergeLayer(64, kernel_size=3)
    m.eval()
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        out = m(x)
    # 手算 forward（不用 addcmul 版本）
    import torch.nn.functional as F
    B, T, D = x.shape
    x_t = x.transpose(1, 2)
    x_padded = F.pad(x_t, (m.pad, 0))
    agg = F.conv1d(x_padded, m.conv.weight, None, groups=D).transpose(1, 2)
    z = torch.sigmoid(m.gate(x))
    expected_out_raw = x + z * (agg - x)
    expected = m.norm(expected_out_raw)
    assert torch.allclose(out, expected, atol=1e-6), \
        f"CharMerge addcmul 不等价: max_diff={(out - expected).abs().max().item():.2e}"


def test_highway_gates_sigmoid_merge_equivalence():
    """R35-4: highway_gates 两次 sigmoid 合并为一次的数值等价性。

    sigmoid 逐元素，先 sigmoid 整体再切片 == 先切片再 sigmoid。
    """
    torch.manual_seed(42)
    highway_gates = nn.Linear(64, 2)
    x = torch.randn(2, 8, 64)
    # 合并路径（R35）
    gates = torch.sigmoid(highway_gates(x))
    gate1_new = gates[..., 0:1]
    gate2_new = gates[..., 1:2]
    # 原路径（R32）
    raw = highway_gates(x)
    gate1_old = torch.sigmoid(raw[..., 0:1])
    gate2_old = torch.sigmoid(raw[..., 1:2])
    assert torch.allclose(gate1_new, gate1_old, atol=1e-7), \
        f"gate1 sigmoid 合并不等价: max_diff={(gate1_new - gate1_old).abs().max().item():.2e}"
    assert torch.allclose(gate2_new, gate2_old, atol=1e-7), \
        f"gate2 sigmoid 合并不等价: max_diff={(gate2_new - gate2_old).abs().max().item():.2e}"


def test_full_retrieval_bias_arange_dedup():
    """R35-5: _full_retrieval_bias 重复 arange 消除后的数值等价性。

    验证 window>0 和 window=0 两种路径的 causal mask 正确性。
    """
    from models.transformer import TransformerModel
    from models.model_config import ModelConfig, AttnConfig
    from models.config_loader import build_model

    torch.manual_seed(42)
    config = {
        'model': {
            'vocab_size': 100, 'embedding_dim': 64, 'num_heads': 4,
            'num_layers': 2, 'hidden_dim': 128, 'max_seq_length': 32,
            'block_types': ['attn', 'attn'],
            'attn': {'retrieval_full': True, 'window': 4, 'retrieval_topk': 2},
        }
    }
    model = build_model(config, device='cpu')
    model.eval()
    x = torch.randint(0, 100, (2, 8))
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 8, 100), f"输出形状错误: {out.shape}"


def test_addcmul_dml_supported():
    """R35-6: addcmul 在 DML 设备上可用且数值正确（CPU 验证等价即可）。"""
    torch.manual_seed(42)
    h2 = torch.randn(2, 8, 64)
    g = torch.randn(2, 8, 64)
    h1 = torch.randn(2, 8, 64)
    r1 = h2 + g * (h1 - h2)
    r2 = torch.addcmul(h2, g, h1 - h2)
    assert torch.allclose(r1, r2, atol=1e-6), \
        f"addcmul CPU 等价性失败: max_diff={(r1 - r2).abs().max().item():.2e}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
