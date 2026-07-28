"""第三十四轮回归测试：特性优化 + 算子精简（低风险数值等价优化）。

覆盖 4 处 R34 优化：
- R34-1: GatedDeltaNet forward loop 消除 unsqueeze/squeeze 冗余（mixers.py）
        alpha_t/beta_t 保持原形状，仅 S 更新需 unsqueeze，z 更新免 squeeze。
- R34-2: CrossLayerRouter routed/k → routed*(1/k)（transformer.py，预计算倒数）
- R34-3a: DifferentialAttention /sqrt(D) → *_inv_sqrt_d（mixers.py，预计算倒数）
- R34-3b: MambaSSM _compute_dA_and_xb dt.unsqueeze(-1) 复用 + xb 中间张量减小（mixers.py）

所有优化数学等价，浮点误差 < 1e-6。
"""
import torch
import torch.nn as nn
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mixers import GatedDeltaNet, DifferentialAttention, MambaSSM, MambaSSMWithCAST
from models.transformer import CrossLayerRouter


# ─── R34-1: GatedDeltaNet unsqueeze/squeeze 消除 ──────────────────────────

def _make_gated_delta(channel_wise=False, rwkv7=False, dim=64, num_heads=4):
    """创建 GatedDeltaNet 模型用于测试。"""
    torch.manual_seed(42)
    return GatedDeltaNet(
        dim=dim, num_heads=num_heads, max_seq_length=64,
        qk_norm=True, attn_temp=True, feature='relu',
        alpha_init=-2.0, beta_init=2.0,
        channel_wise=channel_wise, rwkv7=rwkv7)


def test_gated_delta_scalar_mode_forward():
    """R34-1: GatedDeltaNet 标量模式前向输出正确（channel_wise=False）。"""
    torch.manual_seed(42)
    m = _make_gated_delta(channel_wise=False)
    m.eval()
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        out, _ = m(x)
    assert out.shape == (2, 8, 64), f"输出形状错误: {out.shape}"
    assert not torch.isnan(out).any(), "输出含 NaN"
    assert not torch.isinf(out).any(), "输出含 Inf"


def test_gated_delta_channel_wise_mode_forward():
    """R34-1: GatedDeltaNet 通道模式前向输出正确（channel_wise=True）。"""
    torch.manual_seed(42)
    m = _make_gated_delta(channel_wise=True)
    m.eval()
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        out, _ = m(x)
    assert out.shape == (2, 8, 64), f"输出形状错误: {out.shape}"
    assert not torch.isnan(out).any(), "输出含 NaN"


def test_gated_delta_rwkv7_forward():
    """R34-1: GatedDeltaNet RWKV-7 模式前向输出正确。"""
    torch.manual_seed(42)
    m = _make_gated_delta(rwkv7=True)
    m.eval()
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        out, _ = m(x)
    assert out.shape == (2, 8, 64), f"输出形状错误: {out.shape}"
    assert not torch.isnan(out).any(), "输出含 NaN"


def test_gated_delta_cache_parity():
    """R34-1: GatedDeltaNet 全量前向 vs 增量解码 cache parity。

    增量解码须传 start_pos=t 使 RoPE 位置与全量前向一致（全量 0..T-1 vs 增量每步 0）。
    delta rule 递推浮点累积阈值 0.1（与 R19/R31 一致）。
    """
    torch.manual_seed(42)
    m = _make_gated_delta()
    m.eval()
    x = torch.randn(1, 8, 64)

    # 全量前向
    with torch.no_grad():
        out_full, present = m(x, use_cache=True)

    # 逐 token 增量解码（传 start_pos 保持 RoPE 位置一致）
    outs_inc = []
    past_kv = None
    with torch.no_grad():
        for t in range(8):
            xt = x[:, t:t+1, :]
            out_t, past_kv = m(xt, past_kv=past_kv, use_cache=True, start_pos=t)
            outs_inc.append(out_t)
    out_inc = torch.cat(outs_inc, dim=1)

    max_diff = (out_full - out_inc).abs().max().item()
    assert max_diff < 0.1, f"cache parity 失败: max_diff={max_diff:.2e}"


def test_gated_delta_channel_wise_cache_parity():
    """R34-1: GatedDeltaNet channel_wise 模式 cache parity。

    增量解码须传 start_pos=t 使 RoPE 位置与全量前向一致。
    delta rule 递推浮点累积阈值 0.1（与 R19/R31 一致）。
    """
    torch.manual_seed(42)
    m = _make_gated_delta(channel_wise=True)
    m.eval()
    x = torch.randn(1, 8, 64)

    with torch.no_grad():
        out_full, present = m(x, use_cache=True)

    outs_inc = []
    past_kv = None
    with torch.no_grad():
        for t in range(8):
            xt = x[:, t:t+1, :]
            out_t, past_kv = m(xt, past_kv=past_kv, use_cache=True, start_pos=t)
            outs_inc.append(out_t)
    out_inc = torch.cat(outs_inc, dim=1)

    max_diff = (out_full - out_inc).abs().max().item()
    assert max_diff < 0.1, f"channel_wise cache parity 失败: max_diff={max_diff:.2e}"


def test_gated_delta_rwkv7_cache_parity():
    """R34-1: GatedDeltaNet RWKV-7 模式 cache parity。

    增量解码须传 start_pos=t 使 RoPE 位置与全量前向一致。
    delta rule 递推浮点累积阈值 0.1（与 R19/R31 一致）。
    """
    torch.manual_seed(42)
    m = _make_gated_delta(rwkv7=True)
    m.eval()
    x = torch.randn(1, 8, 64)

    with torch.no_grad():
        out_full, present = m(x, use_cache=True)

    outs_inc = []
    past_kv = None
    with torch.no_grad():
        for t in range(8):
            xt = x[:, t:t+1, :]
            out_t, past_kv = m(xt, past_kv=past_kv, use_cache=True, start_pos=t)
            outs_inc.append(out_t)
    out_inc = torch.cat(outs_inc, dim=1)

    max_diff = (out_full - out_inc).abs().max().item()
    assert max_diff < 0.1, f"RWKV-7 cache parity 失败: max_diff={max_diff:.2e}"


def test_gated_delta_gradient_flow():
    """R34-1: GatedDeltaNet 梯度正常回流（unsqueeze/squeeze 消除不破坏 autograd）。"""
    torch.manual_seed(42)
    m = _make_gated_delta()
    m.train()
    x = torch.randn(2, 8, 64, requires_grad=True)
    out, _ = m(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "梯度未回流到输入"
    assert not torch.isnan(x.grad).any(), "梯度含 NaN"
    # alpha_beta_proj 权重应有梯度
    assert m.alpha_beta_proj.weight.grad is not None, "alpha_beta_proj 权重无梯度"


def test_gated_delta_rwkv7_gradient_flow():
    """R34-1: GatedDeltaNet RWKV-7 梯度正常回流。"""
    torch.manual_seed(42)
    m = _make_gated_delta(rwkv7=True)
    m.train()
    x = torch.randn(2, 8, 64, requires_grad=True)
    out, _ = m(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "梯度未回流到输入"
    assert m.z_proj.weight.grad is not None, "z_proj 权重无梯度"
    assert m.b_proj.weight.grad is not None, "b_proj 权重无梯度"


# ─── R34-2: CrossLayerRouter division→multiplication ─────────────────────

def test_cross_layer_router_forward():
    """R34-2: CrossLayerRouter 前向输出正确。"""
    torch.manual_seed(42)
    dim = 64
    num_layers = 4
    router = CrossLayerRouter(dim, num_layers, topk=2)
    router.eval()
    x = torch.randn(2, 8, dim)
    prev_outputs = [torch.randn(2, 8, dim) for _ in range(3)]
    with torch.no_grad():
        out = router.route(3, x, prev_outputs)
    assert out.shape == (2, 8, dim), f"输出形状错误: {out.shape}"
    assert not torch.isnan(out).any(), "输出含 NaN"


def test_cross_layer_router_division_equivalence():
    """R34-2: routed*(1/k) == routed/k 数学等价。"""
    torch.manual_seed(42)
    dim = 64
    num_layers = 4
    k = 2
    router = CrossLayerRouter(dim, num_layers, topk=k)
    router.eval()
    x = torch.randn(2, 8, dim)
    prev_outputs = [torch.randn(2, 8, dim) for _ in range(3)]

    with torch.no_grad():
        out_new = router.route(3, x, prev_outputs)

    # 手算原式（routed / k）
    prev_stack = torch.stack(prev_outputs, dim=1)
    prev_mean = prev_stack.mean(dim=2)
    scores = router.routers[3](prev_mean).squeeze(-1)
    _, topk_idx = torch.topk(scores, k, dim=-1)
    pos = torch.arange(3, device=scores.device)
    mask = (topk_idx.unsqueeze(-1) == pos.view(1, 1, -1)).any(dim=1).to(dtype=scores.dtype)
    gates = torch.sigmoid(scores) * mask
    routed_ref = torch.einsum('bn,bntd->btd', gates, prev_stack)
    routed_ref = routed_ref / k  # 原式：除法
    out_ref = x + routed_ref

    assert torch.allclose(out_new, out_ref, atol=1e-7), \
        f"division→multiplication 不等价: max_diff={(out_new - out_ref).abs().max().item():.2e}"


def test_cross_layer_router_layer0_passthrough():
    """R34-2: 第 0 层直接返回 x（无前层可路由）。"""
    router = CrossLayerRouter(64, 4, topk=2)
    x = torch.randn(2, 8, 64)
    out = router.route(0, x, [])
    assert torch.equal(out, x), "第 0 层应直接返回 x"


def test_cross_layer_router_gradient_flow():
    """R34-2: CrossLayerRouter 梯度正常回流。"""
    torch.manual_seed(42)
    dim = 64
    router = CrossLayerRouter(dim, 4, topk=2)
    router.train()
    x = torch.randn(2, 8, dim, requires_grad=True)
    prev_outputs = [torch.randn(2, 8, dim, requires_grad=True) for _ in range(3)]
    out = router.route(3, x, prev_outputs)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "梯度未回流到 x"
    assert router.routers[3].weight.grad is not None, "梯度未回流到 router weight"


# ─── R34-3a: DifferentialAttention /sqrt(D) → *_inv_sqrt_d ───────────────

def test_differential_attention_forward():
    """R34-3a: DifferentialAttention 前向输出正确。"""
    torch.manual_seed(42)
    dim = 64
    num_heads = 4
    m = DifferentialAttention(dim, num_heads, max_seq_length=64)
    m.eval()
    x = torch.randn(2, 8, dim)
    with torch.no_grad():
        out, _ = m(x)
    assert out.shape == (2, 8, dim), f"输出形状错误: {out.shape}"
    assert not torch.isnan(out).any(), "输出含 NaN"


def test_differential_attention_inv_sqrt_d_equivalence():
    """R34-3a: scores * _inv_sqrt_d == scores / sqrt(D) 数学等价。

    禁用 QK-Norm 和温度以隔离测试 _inv_sqrt_d 等价性（否则手动计算须复现
    QK-Norm/温度缩放逻辑，与测试目的无关）。
    """
    torch.manual_seed(42)
    dim = 64
    num_heads = 4
    D = dim // num_heads
    m = DifferentialAttention(dim, num_heads, max_seq_length=64,
                               qk_norm=False, attn_temp=False)
    m.eval()

    # 验证 _inv_sqrt_d 值正确
    expected_inv = 1.0 / math.sqrt(D)
    assert abs(m._inv_sqrt_d - expected_inv) < 1e-10, \
        f"_inv_sqrt_d 值错误: {m._inv_sqrt_d} vs {expected_inv}"

    # 验证前向等价（手动计算 / sqrt(D) 路径）
    x = torch.randn(2, 8, dim)
    with torch.no_grad():
        out_new, _ = m(x)

    # 手算原式（/ sqrt(D)），无 QK-Norm/温度（已禁用）
    B, T, C = x.shape
    H, D_h = m.num_heads, m.head_dim
    qkv12 = m.qkv12(x)
    q1, k1, v, q2, k2 = qkv12.reshape(B, T, 5, H, D_h).unbind(dim=2)
    q1t = q1.transpose(1, 2)
    q2t = q2.transpose(1, 2)
    k1t = k1.transpose(1, 2)
    k2t = k2.transpose(1, 2)
    vt = v.transpose(1, 2)
    scores1_ref = torch.matmul(q1t, k1t.transpose(-2, -1)) / math.sqrt(D_h)
    scores2_ref = torch.matmul(q2t, k2t.transpose(-2, -1)) / math.sqrt(D_h)
    # softmax
    causal = torch.triu(torch.ones(1, 1, T, T, dtype=torch.bool), diagonal=1)
    scores1_ref = scores1_ref.masked_fill(causal, m.mask_fill_value)
    scores2_ref = scores2_ref.masked_fill(causal, m.mask_fill_value)
    attn1_ref = torch.softmax(scores1_ref, dim=-1)
    attn2_ref = torch.softmax(scores2_ref, dim=-1)
    lam = torch.sigmoid(m.diff_lambda)
    attn_diff_ref = attn1_ref - lam * attn2_ref
    out_ref = torch.matmul(attn_diff_ref, vt)
    out_ref = out_ref.transpose(1, 2).reshape(B, T, C)
    out_ref = m.proj(out_ref)

    assert torch.allclose(out_new, out_ref, atol=1e-6), \
        f"/sqrt(D) → *_inv_sqrt_d 不等价: max_diff={(out_new - out_ref).abs().max().item():.2e}"


def test_differential_attention_cache_parity():
    """R34-3a: DifferentialAttention cache parity。"""
    torch.manual_seed(42)
    dim = 64
    m = DifferentialAttention(dim, 4, max_seq_length=64)
    m.eval()
    x = torch.randn(1, 8, dim)

    with torch.no_grad():
        out_full, present = m(x, use_cache=True)

    outs_inc = []
    past_kv = None
    with torch.no_grad():
        for t in range(8):
            xt = x[:, t:t+1, :]
            out_t, past_kv = m(xt, past_kv=past_kv, use_cache=True)
            outs_inc.append(out_t)
    out_inc = torch.cat(outs_inc, dim=1)

    assert torch.allclose(out_full, out_inc, atol=1e-5), \
        f"cache parity 失败: max_diff={(out_full - out_inc).abs().max().item():.2e}"


def test_differential_attention_gradient_flow():
    """R34-3a: DifferentialAttention 梯度正常回流。"""
    torch.manual_seed(42)
    m = DifferentialAttention(64, 4, max_seq_length=64)
    m.train()
    x = torch.randn(2, 8, 64, requires_grad=True)
    out, _ = m(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "梯度未回流到输入"
    assert m.qkv12.weight.grad is not None, "qkv12 权重无梯度"
    assert m.diff_lambda.grad is not None, "diff_lambda 无梯度"


# ─── R34-3b: MambaSSM _compute_dA_and_xb 优化 ───────────────────────────

def test_mamba_ssm_compute_dA_and_xb_equivalence():
    """R34-3b: MambaSSM _compute_dA_and_xb 优化后数学等价。"""
    torch.manual_seed(42)
    dim = 64
    d_state = 16
    d_inner = dim
    m = MambaSSM(dim, d_state=d_state)
    m.eval()

    B, L = 2, 8
    x_conv = torch.randn(B, L, d_inner)
    dt = torch.randn(B, L, d_inner)
    Bp = torch.randn(B, L, d_state)

    # 新路径（优化后）
    with torch.no_grad():
        dA_new, xb_new = m._compute_dA_and_xb(x_conv, dt, Bp)

    # 手算原式
    A = -torch.exp(m.A_log)
    dA_ref = torch.exp(dt.unsqueeze(-1) * A)
    xb_ref = (dt.unsqueeze(-1) * Bp.unsqueeze(2)) * x_conv.unsqueeze(-1)

    assert torch.allclose(dA_new, dA_ref, atol=1e-7), \
        f"dA 不等价: max_diff={(dA_new - dA_ref).abs().max().item():.2e}"
    assert torch.allclose(xb_new, xb_ref, atol=1e-7), \
        f"xb 不等价: max_diff={(xb_new - xb_ref).abs().max().item():.2e}"


def test_mamba_ssm_forward():
    """R34-3b: MambaSSM 前向输出正确。"""
    torch.manual_seed(42)
    m = MambaSSM(64, d_state=16)
    m.eval()
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        y, _, _ = m(x)
    assert y.shape == (2, 8, 64), f"输出形状错误: {y.shape}"
    assert not torch.isnan(y).any(), "输出含 NaN"


def test_mamba_ssm_cache_parity():
    """R34-3b: MambaSSM 全量 vs 增量 cache parity。"""
    torch.manual_seed(42)
    m = MambaSSM(64, d_state=16)
    m.eval()
    x = torch.randn(1, 8, 64)

    with torch.no_grad():
        y_full, present_state, present_conv = m(x, use_cache=True)

    # 逐 token 增量
    outs = []
    past_state = None
    past_conv = None
    with torch.no_grad():
        for t in range(8):
            xt = x[:, t:t+1, :]
            yt, past_state, past_conv = m(xt, past_state, past_conv, use_cache=True)
            outs.append(yt)
    y_inc = torch.cat(outs, dim=1)

    assert torch.allclose(y_full, y_inc, atol=1e-4), \
        f"cache parity 失败: max_diff={(y_full - y_inc).abs().max().item():.2e}"


def test_mamba_ssm_gradient_flow():
    """R34-3b: MambaSSM 梯度正常回流。"""
    torch.manual_seed(42)
    m = MambaSSM(64, d_state=16)
    m.train()
    x = torch.randn(2, 8, 64, requires_grad=True)
    y, _, _ = m(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None, "梯度未回流到输入"
    assert m.in_proj.weight.grad is not None, "in_proj 权重无梯度"
    assert m.A_log.grad is not None, "A_log 无梯度"


def test_mamba_cast_compute_dA_and_xb_equivalence():
    """R34-3b: MambaSSMWithCAST _compute_dA_and_xb 优化后数学等价。"""
    torch.manual_seed(42)
    dim = 64
    d_state = 16
    m = MambaSSMWithCAST(dim, d_state=d_state)
    m.eval()

    B, L = 2, 8
    d_inner = dim
    x_conv = torch.randn(B, L, d_inner)
    dt = torch.randn(B, L, d_inner)
    Bp = torch.randn(B, L, d_state)

    # 新路径（优化后）
    with torch.no_grad():
        dA_new, xb_new = m._compute_dA_and_xb(x_conv, dt, Bp)

    # 手算原式
    A_base = -torch.exp(m.A_log)
    A_delta = m._compute_cast_delta(x_conv)
    A_effective = A_base.unsqueeze(0).unsqueeze(0) + A_delta
    dA_ref = torch.exp(dt.unsqueeze(-1) * A_effective)
    xb_ref = (dt.unsqueeze(-1) * Bp.unsqueeze(2)) * x_conv.unsqueeze(-1)

    assert torch.allclose(dA_new, dA_ref, atol=1e-6), \
        f"CAST dA 不等价: max_diff={(dA_new - dA_ref).abs().max().item():.2e}"
    assert torch.allclose(xb_new, xb_ref, atol=1e-6), \
        f"CAST xb 不等价: max_diff={(xb_new - xb_ref).abs().max().item():.2e}"


def test_mamba_cast_forward():
    """R34-3b: MambaSSMWithCAST 前向输出正确。"""
    torch.manual_seed(42)
    m = MambaSSMWithCAST(64, d_state=16)
    m.eval()
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        y, _, _ = m(x)
    assert y.shape == (2, 8, 64), f"输出形状错误: {y.shape}"
    assert not torch.isnan(y).any(), "输出含 NaN"


def test_mamba_cast_cache_parity():
    """R34-3b: MambaSSMWithCAST cache parity。"""
    torch.manual_seed(42)
    m = MambaSSMWithCAST(64, d_state=16)
    m.eval()
    x = torch.randn(1, 8, 64)

    with torch.no_grad():
        y_full, present_state, present_conv = m(x, use_cache=True)

    outs = []
    past_state = None
    past_conv = None
    with torch.no_grad():
        for t in range(8):
            xt = x[:, t:t+1, :]
            yt, past_state, past_conv = m(xt, past_state, past_conv, use_cache=True)
            outs.append(yt)
    y_inc = torch.cat(outs, dim=1)

    assert torch.allclose(y_full, y_inc, atol=1e-4), \
        f"CAST cache parity 失败: max_diff={(y_full - y_inc).abs().max().item():.2e}"


# ─── 端到端组合测试 ──────────────────────────────────────────────────────

def test_gated_delta_all_features_combined():
    """R34 端到端：GatedDeltaNet + channel_wise + RWKV-7 组合前向。

    增量解码须传 start_pos=t 使 RoPE 位置与全量前向一致。
    delta rule 递推浮点累积阈值 0.1（与 R19/R31 一致）。
    """
    torch.manual_seed(42)
    m = GatedDeltaNet(
        dim=64, num_heads=4, max_seq_length=64,
        qk_norm=True, attn_temp=True, feature='relu',
        alpha_init=-2.0, beta_init=2.0,
        channel_wise=True, rwkv7=True)
    m.eval()
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        out, _ = m(x)
    assert out.shape == (2, 8, 64), f"输出形状错误: {out.shape}"
    assert not torch.isnan(out).any(), "输出含 NaN"

    # cache parity
    x2 = torch.randn(1, 8, 64)
    with torch.no_grad():
        out_full, _ = m(x2, use_cache=True)
    outs = []
    past = None
    with torch.no_grad():
        for t in range(8):
            out_t, past = m(x2[:, t:t+1], past_kv=past, use_cache=True, start_pos=t)
            outs.append(out_t)
    out_inc = torch.cat(outs, dim=1)
    max_diff = (out_full - out_inc).abs().max().item()
    assert max_diff < 0.1, f"组合 cache parity 失败: max_diff={max_diff:.2e}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
