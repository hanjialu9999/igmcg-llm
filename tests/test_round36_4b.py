"""第三十六轮-4b 回归测试：GatedDeltaNet S 更新 chunk-wise 矩阵前缀扫描。

验证 R36-4b 优化：
1. 半群 (A1,B1)⊙(A2,B2)=(A2@A1, A2@B1+B2) 结合律（小规模 D=4）
2. _matrix_prefix_scan vs 标准 delta rule for 循环等价（atol=1e-5）
3. chunk_scan ON 全量前向 vs 增量解码 cache parity（标量/channel_wise/rwkv7/全开）
4. chunk_scan ON 前向输出有限/形状正确
5. 训练前向+反向+梯度回流 alpha_beta_proj/b_proj/z_proj
6. 谱半径稳定性（alpha 接近 1）
7. 边界 T=1/T=2/T=64

核心：chunk_scan 用「标准 delta rule」S_t = A_t·S_{t-1} + B_t（非原 for-loop 转置约定）。
开启 chunk_scan 后全量训练路径走 _matrix_prefix_scan，增量解码路径（T=1）也用标准形式
单步更新 S=A_t'@S+B_t'，二者约定一致故 cache parity 成立。
不测 chunk_scan ON vs OFF parity（不同 delta rule 约定，必然不等）。
"""
import pytest
import torch

from models.mixers import GatedDeltaNet, _matrix_prefix_scan


def _make_gdn(dim=128, heads=4, seq=64, **kw):
    """直接构造 GatedDeltaNet。"""
    defaults = dict(qk_norm=True, attn_temp=True, max_seq_length=seq,
                    alpha_init=-2.0, beta_init=2.0)
    defaults.update(kw)
    return GatedDeltaNet(dim, heads, **defaults)


def _compose(A1, B1, A2, B2):
    """半群运算 (A2,B2)⊙(A1,B1) = (A2@A1, A2@B1+B2)。

    语义：先应用 (A1,B1) 再应用 (A2,B2)，即 S' = A1@S+B1, S'' = A2@S'+B2。
    """
    A_new = torch.bmm(A2, A1)
    B_new = torch.bmm(A2, B1) + B2
    return A_new, B_new


def _reference_std_forloop(A, B, past_state=None):
    """参考实现：标准 delta rule for 循环 S_t = A_t @ S_{t-1} + B_t。

    Args:
        A: (Bb, L, D, D)
        B: (Bb, L, D, D)
        past_state: (Bb, D, D) 或 None（零矩阵）

    Returns:
        S: (Bb, L, D, D)
    """
    Bb, L, D, _ = A.shape
    if past_state is None:
        past_state = torch.zeros(Bb, D, D, device=A.device, dtype=A.dtype)
    S = past_state
    S_all = []
    for t in range(L):
        S = torch.bmm(A[:, t], S) + B[:, t]
        S_all.append(S)
    return torch.stack(S_all, dim=1)


def _build_AB_mats(alpha, beta, kf, v, rwkv7=False, z_gate=None, b_dir=None):
    """构建标准 delta rule 的 A_t/B_t 矩阵（与 mixers.py 全量路径一致）。

    Args:
        alpha/beta: (B,H,T,1) 标量 或 (B,H,T,D) channel_wise
        kf/v: (B,H,T,D)
        rwkv7: 是否应用 rank-1 修正
        z_gate: (B,H,T) per-head 标量门
        b_dir: (B,H,T,D) 扰动方向

    Returns:
        A_mats, B_mats: (B,H,T,D,D)
    """
    B, H, T, D = kf.shape
    I_D = torch.eye(D, device=kf.device, dtype=kf.dtype).view(1, 1, 1, D, D)
    kf_kfT = kf.unsqueeze(-1) * kf.unsqueeze(-2)   # (B,H,T,D,D)
    v_k = v.unsqueeze(-1) * kf.unsqueeze(-2)       # (B,H,T,D,D)
    A_mats = alpha.unsqueeze(-1) * I_D - beta.unsqueeze(-1) * kf_kfT
    B_mats = beta.unsqueeze(-1) * v_k
    if rwkv7:
        bT_A = torch.einsum('bhtd,bhtde->bhte', b_dir, A_mats)  # (B,H,T,D)
        bT_B = torch.einsum('bhtd,bhtde->bhte', b_dir, B_mats)
        z_exp = z_gate.unsqueeze(-1).unsqueeze(-1)              # (B,H,T,1,1)
        A_mats = A_mats + z_exp * (b_dir.unsqueeze(-1) * bT_A.unsqueeze(-2))
        B_mats = B_mats + z_exp * (b_dir.unsqueeze(-1) * bT_B.unsqueeze(-2))
    return A_mats, B_mats


# ===== 1. 半群结合律 =====

class TestSemigroupAssociativity:
    def test_semigroup_associativity_scalar(self):
        """R36-4b-1: 半群 (A2@A1, A2@B1+B2) 结合律（D=4，随机矩阵）。"""
        torch.manual_seed(42)
        D = 4
        # 三个随机 (A, B) 对，batch=3
        A1 = torch.randn(3, D, D)
        B1 = torch.randn(3, D, D)
        A2 = torch.randn(3, D, D)
        B2 = torch.randn(3, D, D)
        A3 = torch.randn(3, D, D)
        B3 = torch.randn(3, D, D)

        # ((A1,B1) ⊙ (A2,B2)) ⊙ (A3,B3)
        A_l, B_l = _compose(A1, B1, A2, B2)
        A_left, B_left = _compose(A_l, B_l, A3, B3)

        # (A1,B1) ⊙ ((A2,B2) ⊙ (A3,B3))
        A_r, B_r = _compose(A2, B2, A3, B3)
        A_right, B_right = _compose(A1, B1, A_r, B_r)

        assert torch.allclose(A_left, A_right, atol=1e-5), \
            f"半群 A 结合律失败: max_diff={(A_left - A_right).abs().max().item()}"
        assert torch.allclose(B_left, B_right, atol=1e-5), \
            f"半群 B 结合律失败: max_diff={(B_left - B_right).abs().max().item()}"

    def test_semigroup_associativity_near_identity(self):
        """R36-4b-2: 半群结合律（A 接近单位阵，模拟 delta rule 的 α·I - β·k⊗k^T）。"""
        torch.manual_seed(42)
        D = 4
        I = torch.eye(D).view(1, D, D)
        # A = 0.9·I - 0.1·k⊗k^T（谱半径 < 1，模拟稳定 delta rule）
        k1 = torch.randn(3, D); k1 = k1 / k1.norm(dim=-1, keepdim=True)
        k2 = torch.randn(3, D); k2 = k2 / k2.norm(dim=-1, keepdim=True)
        k3 = torch.randn(3, D); k3 = k3 / k3.norm(dim=-1, keepdim=True)
        A1 = 0.9 * I - 0.1 * (k1.unsqueeze(-1) * k1.unsqueeze(-2))
        A2 = 0.9 * I - 0.1 * (k2.unsqueeze(-1) * k2.unsqueeze(-2))
        A3 = 0.9 * I - 0.1 * (k3.unsqueeze(-1) * k3.unsqueeze(-2))
        B1 = 0.1 * torch.randn(3, D, D)
        B2 = 0.1 * torch.randn(3, D, D)
        B3 = 0.1 * torch.randn(3, D, D)

        A_l, B_l = _compose(A1, B1, A2, B2)
        A_left, B_left = _compose(A_l, B_l, A3, B3)

        A_r, B_r = _compose(A2, B2, A3, B3)
        A_right, B_right = _compose(A1, B1, A_r, B_r)

        assert torch.allclose(A_left, A_right, atol=1e-6), \
            f"近单位阵半群 A 结合律失败: max_diff={(A_left - A_right).abs().max().item()}"
        assert torch.allclose(B_left, B_right, atol=1e-6), \
            f"近单位阵半群 B 结合律失败: max_diff={(B_left - B_right).abs().max().item()}"


# ===== 2. _matrix_prefix_scan vs for 循环等价（标准 delta rule）=====

class TestMatrixPrefixScanEquivalence:
    def test_scan_vs_forloop_t16(self):
        """R36-4b-3: _matrix_prefix_scan vs for 循环等价（T=16, D=8, atol=1e-5）。"""
        torch.manual_seed(42)
        Bb, L, D = 2, 16, 8
        A = torch.randn(Bb, L, D, D) * 0.3
        B = torch.randn(Bb, L, D, D) * 0.3

        S_scan = _matrix_prefix_scan(A, B, past_state=None, chunk_size=16)
        S_ref = _reference_std_forloop(A, B, past_state=None)

        max_diff = (S_scan - S_ref).abs().max().item()
        assert max_diff < 1e-5, f"T=16 scan vs forloop 不等价: max_diff={max_diff}"

    def test_scan_vs_forloop_t64_multi_chunk(self):
        """R36-4b-4: 多 chunk 场景 scan vs forloop（T=64, chunk_size=16 → 4 chunks, atol=1e-5）。"""
        torch.manual_seed(42)
        Bb, L, D = 2, 64, 8
        A = torch.randn(Bb, L, D, D) * 0.3
        B = torch.randn(Bb, L, D, D) * 0.3

        S_scan = _matrix_prefix_scan(A, B, past_state=None, chunk_size=16)
        S_ref = _reference_std_forloop(A, B, past_state=None)

        max_diff = (S_scan - S_ref).abs().max().item()
        assert max_diff < 1e-5, f"T=64 多 chunk scan vs forloop 不等价: max_diff={max_diff}"

    def test_scan_vs_forloop_with_past_state(self):
        """R36-4b-5: 带 past_state 的 scan vs forloop（模拟增量解码续算, atol=1e-5）。"""
        torch.manual_seed(42)
        Bb, L, D = 2, 16, 8
        A = torch.randn(Bb, L, D, D) * 0.3
        B = torch.randn(Bb, L, D, D) * 0.3
        past_state = torch.randn(Bb, D, D) * 0.5

        S_scan = _matrix_prefix_scan(A, B, past_state=past_state, chunk_size=16)
        S_ref = _reference_std_forloop(A, B, past_state=past_state)

        max_diff = (S_scan - S_ref).abs().max().item()
        assert max_diff < 1e-5, f"带 past_state scan vs forloop 不等价: max_diff={max_diff}"

    def test_scan_t1_degenerate(self):
        """R36-4b-6: T=1 退化情况 S_0 = A_0 @ past + B_0。"""
        torch.manual_seed(42)
        Bb, D = 2, 4
        A = torch.randn(Bb, 1, D, D)
        B = torch.randn(Bb, 1, D, D)
        past_state = torch.randn(Bb, D, D)

        S_scan = _matrix_prefix_scan(A, B, past_state=past_state, chunk_size=16)
        S_ref = torch.bmm(A[:, 0], past_state).unsqueeze(1) + B

        max_diff = (S_scan - S_ref).abs().max().item()
        assert max_diff < 1e-6, f"T=1 退化 scan 不正确: max_diff={max_diff}"


# ===== 3. chunk_scan ON 前向输出 =====

class TestChunkScanForward:
    def test_forward_output_finite(self):
        """R36-4b-7: chunk_scan ON 前向输出有限（不发散）。"""
        torch.manual_seed(42)
        gdn = _make_gdn(chunk_scan=True, chunk_size=16)
        gdn.eval()
        B, T = 2, 8
        x = torch.randn(B, T, 128)
        out, present = gdn(x, use_cache=True)
        assert torch.isfinite(out).all()
        assert out.shape == (B, T, 128)

    def test_forward_output_shape(self):
        """R36-4b-8: chunk_scan ON 输出形状正确。"""
        torch.manual_seed(42)
        gdn = _make_gdn(chunk_scan=True, chunk_size=16)
        gdn.eval()
        B, T = 3, 16
        x = torch.randn(B, T, 128)
        out, _ = gdn(x, use_cache=False)
        assert out.shape == (B, T, 128)

    def test_forward_channel_wise(self):
        """R36-4b-9: chunk_scan ON + channel_wise 前向有限。"""
        torch.manual_seed(42)
        gdn = _make_gdn(channel_wise=True, chunk_scan=True, chunk_size=16)
        gdn.eval()
        B, T = 2, 8
        x = torch.randn(B, T, 128)
        out, _ = gdn(x, use_cache=True)
        assert torch.isfinite(out).all()
        assert out.shape == (B, T, 128)

    def test_forward_rwkv7(self):
        """R36-4b-10: chunk_scan ON + rwkv7 前向有限。"""
        torch.manual_seed(42)
        gdn = _make_gdn(rwkv7=True, chunk_scan=True, chunk_size=16)
        gdn.eval()
        B, T = 2, 8
        x = torch.randn(B, T, 128)
        out, _ = gdn(x, use_cache=True)
        assert torch.isfinite(out).all()

    def test_forward_all_features(self):
        """R36-4b-11: chunk_scan ON + channel_wise + rwkv7 全开前向有限。"""
        torch.manual_seed(42)
        gdn = _make_gdn(channel_wise=True, rwkv7=True, chunk_scan=True, chunk_size=16)
        gdn.eval()
        B, T = 2, 8
        x = torch.randn(B, T, 128)
        out, _ = gdn(x, use_cache=True)
        assert torch.isfinite(out).all()
        assert out.shape == (B, T, 128)


# ===== 4. chunk_scan ON cache parity（全量 vs 增量解码）=====

class TestChunkScanCacheParity:
    def test_cache_parity_scalar(self):
        """R36-4b-12: chunk_scan ON 标量模式 cache parity（全量 vs 增量，阈值 0.1）。"""
        torch.manual_seed(42)
        gdn = _make_gdn(chunk_scan=True, chunk_size=16)
        gdn.eval()
        B, T = 1, 8
        x = torch.randn(B, T, 128)

        out_full, _ = gdn(x, use_cache=True)
        outs = []
        past = None
        for t in range(T):
            o, past = gdn(x[:, t:t+1], past_kv=past, use_cache=True, start_pos=t)
            outs.append(o)
        out_inc = torch.cat(outs, dim=1)

        max_diff = (out_inc - out_full).abs().max().item()
        assert max_diff < 0.1, f"标量模式 chunk_scan cache parity 失败: max_diff={max_diff}"

    def test_cache_parity_channel_wise(self):
        """R36-4b-13: chunk_scan ON channel_wise 模式 cache parity（阈值 0.1）。"""
        torch.manual_seed(42)
        gdn = _make_gdn(channel_wise=True, chunk_scan=True, chunk_size=16)
        gdn.eval()
        B, T = 1, 8
        x = torch.randn(B, T, 128)

        out_full, _ = gdn(x, use_cache=True)
        outs = []
        past = None
        for t in range(T):
            o, past = gdn(x[:, t:t+1], past_kv=past, use_cache=True, start_pos=t)
            outs.append(o)
        out_inc = torch.cat(outs, dim=1)

        max_diff = (out_inc - out_full).abs().max().item()
        assert max_diff < 0.1, f"channel_wise chunk_scan cache parity 失败: max_diff={max_diff}"

    def test_cache_parity_rwkv7(self):
        """R36-4b-14: chunk_scan ON rwkv7 模式 cache parity（阈值 0.1）。"""
        torch.manual_seed(42)
        gdn = _make_gdn(rwkv7=True, chunk_scan=True, chunk_size=16)
        gdn.eval()
        B, T = 1, 8
        x = torch.randn(B, T, 128)

        out_full, _ = gdn(x, use_cache=True)
        outs = []
        past = None
        for t in range(T):
            o, past = gdn(x[:, t:t+1], past_kv=past, use_cache=True, start_pos=t)
            outs.append(o)
        out_inc = torch.cat(outs, dim=1)

        max_diff = (out_inc - out_full).abs().max().item()
        assert max_diff < 0.1, f"rwkv7 chunk_scan cache parity 失败: max_diff={max_diff}"

    def test_cache_parity_all_features(self):
        """R36-4b-15: chunk_scan ON 全开模式 cache parity（阈值 0.2）。"""
        torch.manual_seed(42)
        gdn = _make_gdn(channel_wise=True, rwkv7=True, chunk_scan=True, chunk_size=16)
        gdn.eval()
        B, T = 1, 8
        x = torch.randn(B, T, 128)

        out_full, _ = gdn(x, use_cache=True)
        outs = []
        past = None
        for t in range(T):
            o, past = gdn(x[:, t:t+1], past_kv=past, use_cache=True, start_pos=t)
            outs.append(o)
        out_inc = torch.cat(outs, dim=1)

        max_diff = (out_inc - out_full).abs().max().item()
        assert max_diff < 0.2, f"全开 chunk_scan cache parity 失败: max_diff={max_diff}"


# ===== 5. 训练 smoke test =====

class TestTrainingSmoke:
    def test_training_forward_backward(self):
        """R36-4b-16: chunk_scan ON 训练前向+反向不崩溃。"""
        torch.manual_seed(42)
        gdn = _make_gdn(chunk_scan=True, chunk_size=16)
        gdn.train()
        B, T = 2, 8
        x = torch.randn(B, T, 128, requires_grad=True)
        out, _ = gdn(x, use_cache=False)
        loss = out.sum()
        loss.backward()
        assert torch.isfinite(loss).item()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_training_gradient_flow_alpha_beta(self):
        """R36-4b-17: chunk_scan ON 梯度回流到 alpha_beta_proj。"""
        torch.manual_seed(42)
        gdn = _make_gdn(chunk_scan=True, chunk_size=16)
        gdn.train()
        B, T = 2, 8
        x = torch.randn(B, T, 128)
        out, _ = gdn(x, use_cache=False)
        loss = out.sum()
        loss.backward()
        assert gdn.alpha_beta_proj.weight.grad is not None
        assert torch.isfinite(gdn.alpha_beta_proj.weight.grad).all()
        assert gdn.alpha_beta_proj.bias.grad is not None

    def test_training_gradient_flow_rwkv7(self):
        """R36-4b-18: chunk_scan ON + rwkv7 梯度回流到 b_proj/z_proj。"""
        torch.manual_seed(42)
        gdn = _make_gdn(rwkv7=True, chunk_scan=True, chunk_size=16)
        gdn.train()
        B, T = 2, 8
        x = torch.randn(B, T, 128)
        out, _ = gdn(x, use_cache=False)
        loss = out.sum()
        loss.backward()
        assert gdn.b_proj.weight.grad is not None
        assert torch.isfinite(gdn.b_proj.weight.grad).all()
        assert gdn.z_proj.weight.grad is not None
        assert gdn.z_proj.bias.grad is not None

    def test_training_loss_decreases(self):
        """R36-4b-19: chunk_scan ON 训练 loss 正常下降。"""
        torch.manual_seed(42)
        gdn = _make_gdn(chunk_scan=True, chunk_size=16)
        gdn.train()
        opt = torch.optim.AdamW(gdn.parameters(), lr=1e-3)
        B, T = 2, 8
        x = torch.randn(B, T, 128)
        target = torch.randn(B, T, 128)
        losses = []
        for _ in range(5):
            opt.zero_grad()
            out, _ = gdn(x, use_cache=False)
            loss = ((out - target) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], f"loss 未下降: {losses[0]:.4f} → {losses[-1]:.4f}"


# ===== 6. 边界与稳定性 =====

class TestEdgeCases:
    def test_t1_forward(self):
        """R36-4b-20: chunk_scan ON T=1 前向正常（增量解码基线）。"""
        torch.manual_seed(42)
        gdn = _make_gdn(chunk_scan=True, chunk_size=16)
        gdn.eval()
        x = torch.randn(1, 1, 128)
        out, present = gdn(x, use_cache=True)
        assert torch.isfinite(out).all()
        assert out.shape == (1, 1, 128)

    def test_t2_forward(self):
        """R36-4b-21: chunk_scan ON T=2 前向正常（短序列边界）。"""
        torch.manual_seed(42)
        gdn = _make_gdn(chunk_scan=True, chunk_size=16)
        gdn.eval()
        x = torch.randn(1, 2, 128)
        out, _ = gdn(x, use_cache=True)
        assert torch.isfinite(out).all()
        assert out.shape == (1, 2, 128)

    def test_t64_forward(self):
        """R36-4b-22: chunk_scan ON T=64 前向正常（多 chunk 场景）。"""
        torch.manual_seed(42)
        gdn = _make_gdn(chunk_scan=True, chunk_size=16, max_seq_length=64)
        gdn.eval()
        x = torch.randn(1, 64, 128)
        out, _ = gdn(x, use_cache=True)
        assert torch.isfinite(out).all()
        assert out.shape == (1, 64, 128)

    def test_large_alpha_stability(self):
        """R36-4b-23: chunk_scan ON alpha 接近 1（弱遗忘）时不发散（谱半径稳定）。"""
        torch.manual_seed(42)
        gdn = _make_gdn(alpha_init=4.0, chunk_scan=True, chunk_size=16)  # sigmoid(4)≈0.98
        gdn.eval()
        B, T = 2, 8
        x = torch.randn(B, T, 128) * 0.1  # 小输入避免累积爆炸
        out, _ = gdn(x, use_cache=False)
        assert torch.isfinite(out).all()

    def test_cache_parity_t64(self):
        """R36-4b-24: chunk_scan ON T=64 多 chunk cache parity（阈值 0.2）。"""
        torch.manual_seed(42)
        gdn = _make_gdn(chunk_scan=True, chunk_size=16, max_seq_length=64)
        gdn.eval()
        B, T = 1, 64
        x = torch.randn(B, T, 128)

        out_full, _ = gdn(x, use_cache=True)
        outs = []
        past = None
        for t in range(T):
            o, past = gdn(x[:, t:t+1], past_kv=past, use_cache=True, start_pos=t)
            outs.append(o)
        out_inc = torch.cat(outs, dim=1)

        max_diff = (out_inc - out_full).abs().max().item()
        # T=64 多 chunk 并行扫描 vs 顺序 bmm，浮点累积误差稍大
        assert max_diff < 0.2, f"T=64 chunk_scan cache parity 失败: max_diff={max_diff}"
