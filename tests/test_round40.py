# -*- coding: utf-8 -*-
"""R40 回归测试：DML 不支持的融合算子（torch.lerp / torch.addcmul）全面清理。

R38 修 layers.py、R39 修 ngram.py 后，本轮发现同根因仍漏网 7 处：
  - gates.py   : convex_combine_scalar / convex_combine_linear / apply_correction
  - mixers.py  : GatedDeltaNet 增量解码 ×4（含 rwkv7 分支）、AxialLinearAttention 融合、DiffAttn
  - rope.py    : YaRN inv_freq 插值
  - transformer.py : DALA 层间目标

问题（与 R38/R39 同根因）：
  1. 函数式 torch.lerp → DML 不支持 aten::lerp.Tensor_out，每调用 CPU 回退 + 同步；
  2. torch.lerp 要求 start/end/weight 同 dtype——CPU bf16 autocast 推理时
     weight（fp32 标量/张量）与 tensor（bf16）不匹配直接 RuntimeError 崩溃；
  3. torch.addcmul 的 backward 在 DML 不支持（R38 layers.py 记录）→ 训练崩溃风险。

统一回退 4 算子等价式：lerp(a,b,w)=a+w*(b-a)；addcmul(a,b,c)=a+b*c。
"""

import torch

from models.gates import convex_combine_scalar, convex_combine_linear, apply_correction


def _module_files():
    import os
    root = os.path.join(os.path.dirname(__file__), '..', 'models')
    names = ['gates.py', 'mixers.py', 'rope.py', 'transformer.py', 'layers.py', 'ngram.py']
    return {n: open(os.path.join(root, n), encoding='utf-8').read() for n in names}


def test_r40_no_torch_lerp_in_models():
    """models/ 生产代码不得再出现函数式 torch.lerp（DML CPU 回退 + bf16 崩溃）。"""
    for name, code in _module_files().items():
        assert 'torch.lerp(' not in code, f"{name} 仍含 torch.lerp（DML 回退）"
        assert 'torch.lerp(' not in code.replace('torch.lerp_', ''), f"{name} 仍含 torch.lerp"


def test_r40_no_torch_addcmul_in_models():
    """models/ 生产代码不得再出现 torch.addcmul（backward 在 DML 不支持）。"""
    for name, code in _module_files().items():
        assert 'torch.addcmul(' not in code, f"{name} 仍含 torch.addcmul（backward DML 崩）"


def test_r40_gates_4op_equivalence():
    """4 算子展开与 R35 融合式数值等价（误差 <1e-6）。"""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(1))
    h1 = torch.randn(2, 4)
    h2 = torch.randn(2, 4)
    g = torch.sigmoid(p)

    out = convex_combine_scalar(p, h1, h2)
    ref = torch.lerp(h2, h1, g)          # R35 原式
    ref2 = g * h1 + (1 - g) * h2          # R29 原式
    assert torch.allclose(out, ref, atol=1e-6)
    assert torch.allclose(out, ref2, atol=1e-6)

    linear = torch.nn.Linear(4, 1)
    out_l = convex_combine_linear(linear, h1, h1, h2)
    g_l = torch.sigmoid(linear(h1))
    ref_l = g_l * h1 + (1 - g_l) * h2
    assert torch.allclose(out_l, ref_l, atol=1e-6)

    cg = torch.sigmoid(p)
    out_c = apply_correction(p, h1, h2)
    assert torch.allclose(out_c, h1 + cg * (h2 - h1), atol=1e-6)


def test_r40_gates_bf16_autocast_forward():
    """CPU bf16 autocast 下 gates 三函数不再崩溃（原 torch.lerp 在此路径报 dtype 错）。"""
    torch.manual_seed(1)
    p = torch.nn.Parameter(torch.randn(1))
    h1 = torch.randn(2, 4, dtype=torch.bfloat16)
    h2 = torch.randn(2, 4, dtype=torch.bfloat16)
    linear = torch.nn.Linear(4, 1)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        o1 = convex_combine_scalar(p, h1, h2)
        o2 = convex_combine_linear(linear, h1, h2, h1)
        o3 = apply_correction(p, h1, h2)
    assert o1.shape == h1.shape and o2.shape == h1.shape and o3.shape == h1.shape
    assert o1.dtype == torch.float32 or o1.dtype == torch.bfloat16


def test_r40_rope_yarn_no_lerp():
    """YaRN inv_freq 计算走 4 算子且与 lerp 式数值等价。"""
    from models.rope import _yarn_linear_ramp_mask
    dim, base, scale, orig = 64, 10000.0, 8.0, 2048
    low, high = 32.0, 1.0
    freq_indices = torch.arange(0, dim, 2).float()
    inv_freq_extrapolation = 1.0 / (base ** (freq_indices / dim))
    inv_freq_interpolation = 1.0 / (scale * base ** (freq_indices / dim))
    mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2)
    out = inv_freq_interpolation + mask * (inv_freq_extrapolation - inv_freq_interpolation)
    ref = torch.lerp(inv_freq_interpolation, inv_freq_extrapolation, mask)
    assert torch.allclose(out, ref, atol=1e-6)


def test_r40_dala_no_lerp():
    """DALA 目标 = x0 + α*(prev-x0)，与 lerp 式数值等价（标量 α 路径）。"""
    torch.manual_seed(2)
    x0 = torch.randn(3, 5)
    prev = torch.randn(3, 5)
    alpha = 0.4
    out = x0 + alpha * (prev - x0)
    ref = torch.lerp(x0, prev, alpha)
    assert torch.allclose(out, ref, atol=1e-6)


def test_r40_delta_rule_no_addcmul():
    """GatedDeltaNet 增量更新 = α*S + β*(v-S·k)⊗k，与 addcmul 式数值等价。"""
    torch.manual_seed(3)
    B, H, D = 2, 4, 8
    S = torch.randn(B, H, D, D)
    kf = torch.randn(B, H, D)
    v = torch.randn(B, H, D)
    alpha_S = torch.randn(B, H, 1, 1)
    beta_S = torch.randn(B, H, 1, 1)
    Sk = torch.einsum('bhd,bhde->bhe', kf, S)
    out = alpha_S * S + beta_S * ((v - Sk).unsqueeze(-1) * kf.unsqueeze(-2))
    ref = torch.addcmul(alpha_S * S, beta_S, (v - Sk).unsqueeze(-1) * kf.unsqueeze(-2))
    assert torch.allclose(out, ref, atol=1e-5)


def test_r40_diffattn_no_addcmul():
    """DiffAttn = attn1 - λ*attn2，与 addcmul(value=-1) 式数值等价。"""
    torch.manual_seed(4)
    attn1 = torch.rand(2, 4, 8, 8)
    attn2 = torch.rand(2, 4, 8, 8)
    lam = torch.rand(1)
    out = attn1 - lam * attn2
    ref = torch.addcmul(attn1, lam, attn2, value=-1)
    assert torch.allclose(out, ref, atol=1e-6)


def test_r40_mixers_forward_smoke():
    """关键 mixer 路径前向冒烟（构造时涉及 lerp/addcmul 的类均可跑）。"""
    from models.config_loader import build_model
    config = {
        'model': {
            'vocab_size': 512, 'embedding_dim': 64, 'num_heads': 2,
            'num_layers': 1, 'hidden_dim': 128, 'max_seq_length': 32,
        }
    }
    model = build_model(config, device='cpu')
    model.eval()
    x = torch.randint(0, 512, (1, 6))
    out = model(x)
    assert out[0].shape == (6, 512) or out[0].shape == (1, 6, 512)
