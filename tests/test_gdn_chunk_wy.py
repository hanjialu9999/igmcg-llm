"""④（10-05）：GatedDeltaNet chunk_wy——教科书 gated delta rule + 块内 WY/UT 分块，状态按 key 行存。

参考递推（fp64 逐步循环）：S_t = α_t S_{t-1} + β_t k_t ⊗ (v_t - α_t k_tᵀ S_{t-1})，o_t = q_tᵀ S_t。
"""

import pytest
import torch
import torch.nn as nn

try:
    import torch_directml  # noqa: F401  与 test_round42 同：提前注册 DML 后端
except Exception:
    pass

from models.mixers import GatedDeltaNet
from models.model_config import ModelConfig
from models.transformer import TransformerModel


def _ref_loop(q, k, v, a, b):
    B, H, T, D = q.shape
    S = q.new_zeros(B, H, D, D)
    outs = []
    for t in range(T):
        at, bt = a[:, :, t], b[:, :, t]  # (B,H,1)
        Sk = torch.einsum('bhd,bhde->bhe', k[:, :, t], S)
        S = at.unsqueeze(-1) * S + bt.unsqueeze(-1) * (k[:, :, t].unsqueeze(-1) * (v[:, :, t] - at * Sk).unsqueeze(-2))
        outs.append(torch.einsum('bhd,bhde->bhe', q[:, :, t], S))
    return torch.stack(outs, dim=2), S


def _inputs(T=37, B=2, H=3, D=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(B, H, T, D, generator=g, dtype=torch.float64) for _ in range(3))
    k = k / k.norm(dim=-1, keepdim=True)
    a = torch.sigmoid(torch.randn(B, H, T, 1, generator=g, dtype=torch.float64) * 3 + 2)  # 含 α→1 长记忆
    b = torch.sigmoid(torch.randn(B, H, T, 1, generator=g, dtype=torch.float64) * 2)
    return [t.requires_grad_() for t in (q, k, v, a, b)]


@pytest.mark.parametrize('T', [16, 37, 64])
def test_chunk_wy_scan_matches_loop_fwd_and_grad(T):
    m = GatedDeltaNet(dim=24, num_heads=3, chunk_wy=True)
    xs = _inputs(T=T)
    o1, S1, _ = m._chunk_wy_scan(*xs)
    o2, S2 = _ref_loop(*xs)
    assert (o1 - o2).abs().max().item() < 1e-9
    assert (S1 - S2).abs().max().item() < 1e-9
    w = torch.randn_like(o1)
    g1 = torch.autograd.grad((o1 * w).sum() + S1.sum(), xs)
    g2 = torch.autograd.grad((o2 * w).sum() + S2.sum(), xs)
    for a, b in zip(g1, g2):
        assert (a - b).abs().max().item() < 1e-8


def test_chunk_wy_full_vs_incremental_parity():
    torch.manual_seed(0)
    m = GatedDeltaNet(dim=64, num_heads=4, chunk_wy=True, max_seq_length=64).eval()
    with torch.no_grad():  # 门控权重放开，让 α/β 逐 token 变化
        m.alpha_beta_proj.weight.normal_(0, 0.3)
    x = torch.randn(2, 37, 64)
    with torch.no_grad():
        full, pres_full = m(x, use_cache=True)
        y, past = m(x[:, :5], use_cache=True)
        ys = [y]
        for t in range(5, 37):
            y, past = m(x[:, t:t + 1], past_kv=past, use_cache=True, start_pos=t)
            ys.append(y)
    assert (full - torch.cat(ys, dim=1)).abs().max().item() < 1e-4
    assert (pres_full[2] - past[2]).abs().max().item() < 1e-4


@pytest.mark.parametrize('kw', [dict(channel_wise=True), dict(rwkv7=True), dict(chunk_scan=True)])
def test_chunk_wy_rejects_unsupported(kw):
    with pytest.raises(ValueError):
        GatedDeltaNet(dim=32, num_heads=2, chunk_wy=True, **kw)


def test_controller_chunk_scan_model_parity():
    cfg = ModelConfig(vocab_size=50, embedding_dim=64, num_heads=4, num_layers=3, hidden_dim=128,
                      max_seq_length=32, controller=True, gradient_checkpointing=False,
                      controller_direction_causal=True, controller_chunk_scan=True)
    m = TransformerModel.from_config(cfg).eval()
    assert all(mx.chunk_wy for mx in m.controller.mixers)
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():  # 放开 direction / film 投影（零初始化下测不出 Controller 分量）
        c = m.controller
        c.direction_proj.weight.copy_(torch.randn(c.direction_proj.weight.shape, generator=g) * 0.1)
        for proj in c.film_projs:
            if isinstance(proj, nn.Linear):
                proj.weight.copy_(torch.randn(proj.weight.shape, generator=g) * 0.05)
    x = torch.randint(0, 50, (2, 20))
    with torch.no_grad():
        full = m(x)
        y, past = m(x[:, :3], use_cache=True)
        ys = [y]
        for t in range(3, 20):
            y, past = m(x[:, t:t + 1], past_key_values=past, use_cache=True)
            ys.append(y)
    assert (full - torch.cat(ys, dim=1)).abs().max().item() < 1e-4
