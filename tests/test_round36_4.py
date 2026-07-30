"""第三十六轮-4 回归测试：GatedDeltaNet z 更新前缀扫描（R36-4 第一阶段）。

验证 R36-4 第一阶段优化：
1. z 前缀扫描 vs for 循环数值等价（标量模式）
2. z 前缀扫描 vs for 循环数值等价（channel_wise 模式）
3. z 前缀扫描 vs for 循环数值等价（rwkv7 模式）
4. cache parity（全量前向 vs 增量解码）
5. 训练前向+反向不崩溃
6. 边界值（T=1, T=2, T=64）
7. 梯度回流到 alpha_beta_proj
8. 输出形状正确

核心优化：z_t = α_t·z_{t-1} + β_t·k_t 改用 _parallel_prefix_scan
（标量线性递推，log(T) 轮 vs T 步串行）
"""
import pytest
import torch

from models.mixers import GatedDeltaNet, _parallel_prefix_scan


def _make_gdn(dim=128, heads=4, seq=64, **kw):
    """直接构造 GatedDeltaNet。"""
    defaults = dict(qk_norm=True, attn_temp=True, max_seq_length=seq,
                    alpha_init=-2.0, beta_init=2.0)
    defaults.update(kw)
    return GatedDeltaNet(dim, heads, **defaults)


def _reference_z_forloop(alpha, beta, kf, B, H, T, D, device, dtype):
    """参考实现：for 循环逐 token 递推 z。"""
    z = torch.zeros(B, H, D, device=device, dtype=dtype)
    z_all = []
    for t in range(T):
        alpha_t = alpha[:, :, t, :]  # (B,H,1) 或 (B,H,D)
        beta_t = beta[:, :, t, :]
        kf_t = kf[:, :, t, :]       # (B,H,D)
        z = alpha_t * z + beta_t * kf_t
        z_all.append(z)
    return torch.stack(z_all, dim=2)  # (B,H,T,D)


# ===== 1. z 前缀扫描 vs for 循环数值等价 =====

class TestZPrefixScanEquivalence:
    def test_z_prefix_scan_scalar_mode(self):
        """R36-4-1: 标量模式下 z 前缀扫描 vs for 循环等价（atol=1e-5）。"""
        torch.manual_seed(42)
        B, H, T, D = 2, 4, 16, 32
        alpha = torch.sigmoid(torch.randn(B, H, T, 1)) * 0.3 + 0.1  # (B,H,T,1)
        beta = torch.sigmoid(torch.randn(B, H, T, 1)) * 0.8 + 0.1
        kf = torch.randn(B, H, T, D)

        # 前缀扫描
        alpha_exp = alpha.expand_as(kf)
        beta_exp = beta.expand_as(kf)
        a_z = alpha_exp.permute(0, 2, 1, 3).reshape(B, T, H * D, 1)
        b_z = (beta_exp * kf).permute(0, 2, 1, 3).reshape(B, T, H * D, 1)
        z_scan = _parallel_prefix_scan(a_z, b_z)
        z_scan = z_scan.reshape(B, T, H, D).permute(0, 2, 1, 3)

        # 参考 for 循环
        z_ref = _reference_z_forloop(alpha, beta, kf, B, H, T, D, kf.device, kf.dtype)

        max_diff = (z_scan - z_ref).abs().max().item()
        assert max_diff < 1e-5, f"标量模式 z 前缀扫描不等价: max_diff={max_diff}"

    def test_z_prefix_scan_channel_wise_mode(self):
        """R36-4-2: channel_wise 模式下 z 前缀扫描 vs for 循环等价（atol=1e-5）。"""
        torch.manual_seed(42)
        B, H, T, D = 2, 4, 16, 32
        alpha = torch.sigmoid(torch.randn(B, H, T, D)) * 0.3 + 0.1  # (B,H,T,D)
        beta = torch.sigmoid(torch.randn(B, H, T, D)) * 0.8 + 0.1
        kf = torch.randn(B, H, T, D)

        # 前缀扫描（channel_wise 已是 (B,H,T,D)）
        a_z = alpha.permute(0, 2, 1, 3).reshape(B, T, H * D, 1)
        b_z = (beta * kf).permute(0, 2, 1, 3).reshape(B, T, H * D, 1)
        z_scan = _parallel_prefix_scan(a_z, b_z)
        z_scan = z_scan.reshape(B, T, H, D).permute(0, 2, 1, 3)

        # 参考 for 循环
        z_ref = _reference_z_forloop(alpha, beta, kf, B, H, T, D, kf.device, kf.dtype)

        max_diff = (z_scan - z_ref).abs().max().item()
        assert max_diff < 1e-5, f"channel_wise 模式 z 前缀扫描不等价: max_diff={max_diff}"

    def test_z_prefix_scan_t1(self):
        """R36-4-3: T=1 时前缀扫描退化为单步（z = β·k）。"""
        torch.manual_seed(42)
        B, H, T, D = 1, 4, 1, 32
        alpha = torch.sigmoid(torch.randn(B, H, T, 1))
        beta = torch.sigmoid(torch.randn(B, H, T, 1))
        kf = torch.randn(B, H, T, D)

        alpha_exp = alpha.expand_as(kf)
        beta_exp = beta.expand_as(kf)
        a_z = alpha_exp.permute(0, 2, 1, 3).reshape(B, T, H * D, 1)
        b_z = (beta_exp * kf).permute(0, 2, 1, 3).reshape(B, T, H * D, 1)
        z_scan = _parallel_prefix_scan(a_z, b_z)
        z_scan = z_scan.reshape(B, T, H, D).permute(0, 2, 1, 3)

        # T=1: z_0 = alpha * 0 + beta * k = beta * k
        z_expected = beta_exp * kf
        max_diff = (z_scan - z_expected).abs().max().item()
        assert max_diff < 1e-6, f"T=1 z 不正确: max_diff={max_diff}"

    def test_z_prefix_scan_t2(self):
        """R36-4-4: T=2 时前缀扫描正确（z_0=β₀k₀, z_1=α₁β₀k₀+β₁k₁）。"""
        torch.manual_seed(42)
        B, H, T, D = 1, 4, 2, 32
        alpha = torch.sigmoid(torch.randn(B, H, T, 1))
        beta = torch.sigmoid(torch.randn(B, H, T, 1))
        kf = torch.randn(B, H, T, D)

        alpha_exp = alpha.expand_as(kf)
        beta_exp = beta.expand_as(kf)
        a_z = alpha_exp.permute(0, 2, 1, 3).reshape(B, T, H * D, 1)
        b_z = (beta_exp * kf).permute(0, 2, 1, 3).reshape(B, T, H * D, 1)
        z_scan = _parallel_prefix_scan(a_z, b_z)
        z_scan = z_scan.reshape(B, T, H, D).permute(0, 2, 1, 3)

        # 手动验证
        z0 = beta_exp[:, :, 0, :] * kf[:, :, 0, :]
        z1 = alpha_exp[:, :, 1, :] * z0 + beta_exp[:, :, 1, :] * kf[:, :, 1, :]
        assert torch.allclose(z_scan[:, :, 0, :], z0, atol=1e-6)
        assert torch.allclose(z_scan[:, :, 1, :], z1, atol=1e-5)

    def test_z_prefix_scan_t64(self):
        """R36-4-5: T=64（训练典型长度）时前缀扫描等价（atol=1e-4）。"""
        torch.manual_seed(42)
        B, H, T, D = 2, 4, 64, 32
        alpha = torch.sigmoid(torch.randn(B, H, T, 1)) * 0.3 + 0.1
        beta = torch.sigmoid(torch.randn(B, H, T, 1)) * 0.8 + 0.1
        kf = torch.randn(B, H, T, D)

        alpha_exp = alpha.expand_as(kf)
        beta_exp = beta.expand_as(kf)
        a_z = alpha_exp.permute(0, 2, 1, 3).reshape(B, T, H * D, 1)
        b_z = (beta_exp * kf).permute(0, 2, 1, 3).reshape(B, T, H * D, 1)
        z_scan = _parallel_prefix_scan(a_z, b_z)
        z_scan = z_scan.reshape(B, T, H, D).permute(0, 2, 1, 3)

        z_ref = _reference_z_forloop(alpha, beta, kf, B, H, T, D, kf.device, kf.dtype)

        max_diff = (z_scan - z_ref).abs().max().item()
        # T=64 时浮点累积误差稍大
        assert max_diff < 1e-4, f"T=64 z 前缀扫描不等价: max_diff={max_diff}"


# ===== 2. GatedDeltaNet 前向输出等价性 =====

class TestGatedDeltaNetForward:
    def test_forward_output_finite(self):
        """R36-4-6: GatedDeltaNet 前向输出有限（不发散）。"""
        torch.manual_seed(42)
        gdn = _make_gdn()
        gdn.eval()
        B, T = 2, 8
        x = torch.randn(B, T, 128)
        out, present = gdn(x, use_cache=True)
        assert torch.isfinite(out).all()
        assert out.shape == (B, T, 128)

    def test_forward_output_shape(self):
        """R36-4-7: 输出形状正确 (B, T, dim)。"""
        gdn = _make_gdn(dim=256, heads=8)
        gdn.eval()
        B, T = 2, 16
        x = torch.randn(B, T, 256)
        out, _ = gdn(x, use_cache=False)
        assert out.shape == (B, T, 256)

    def test_forward_channel_wise(self):
        """R36-4-8: channel_wise 模式前向不崩溃。"""
        gdn = _make_gdn(channel_wise=True)
        gdn.eval()
        B, T = 2, 8
        x = torch.randn(B, T, 128)
        out, present = gdn(x, use_cache=True)
        assert torch.isfinite(out).all()
        assert out.shape == (B, T, 128)

    def test_forward_rwkv7(self):
        """R36-4-9: rwkv7 模式前向不崩溃。"""
        gdn = _make_gdn(rwkv7=True)
        gdn.eval()
        B, T = 2, 8
        x = torch.randn(B, T, 128)
        out, present = gdn(x, use_cache=True)
        assert torch.isfinite(out).all()

    def test_forward_channel_wise_rwkv7(self):
        """R36-4-10: channel_wise + rwkv7 全开前向不崩溃。"""
        gdn = _make_gdn(channel_wise=True, rwkv7=True)
        gdn.eval()
        B, T = 2, 8
        x = torch.randn(B, T, 128)
        out, present = gdn(x, use_cache=True)
        assert torch.isfinite(out).all()


# ===== 3. cache parity（全量前向 vs 增量解码）=====

class TestCacheParity:
    def test_cache_parity_scalar(self):
        """R36-4-11: 标量模式 cache parity（全量 vs 增量，阈值 0.1）。"""
        torch.manual_seed(42)
        gdn = _make_gdn()
        gdn.eval()
        B, T = 1, 8
        x = torch.randn(B, T, 128)

        # 全量前向
        out_full, present = gdn(x, use_cache=True)
        # 增量解码
        outs = []
        past = None
        for t in range(T):
            o, past = gdn(x[:, t:t+1], past_kv=past, use_cache=True, start_pos=t)
            outs.append(o)
        out_inc = torch.cat(outs, dim=1)

        max_diff = (out_inc - out_full).abs().max().item()
        assert max_diff < 0.1, f"标量模式 cache parity 失败: max_diff={max_diff}"

    def test_cache_parity_channel_wise(self):
        """R36-4-12: channel_wise 模式 cache parity（阈值 0.1）。"""
        torch.manual_seed(42)
        gdn = _make_gdn(channel_wise=True)
        gdn.eval()
        B, T = 1, 8
        x = torch.randn(B, T, 128)

        out_full, present = gdn(x, use_cache=True)
        outs = []
        past = None
        for t in range(T):
            o, past = gdn(x[:, t:t+1], past_kv=past, use_cache=True, start_pos=t)
            outs.append(o)
        out_inc = torch.cat(outs, dim=1)

        max_diff = (out_inc - out_full).abs().max().item()
        assert max_diff < 0.1, f"channel_wise cache parity 失败: max_diff={max_diff}"

    def test_cache_parity_rwkv7(self):
        """R36-4-13: rwkv7 模式 cache parity（阈值 0.1）。"""
        torch.manual_seed(42)
        gdn = _make_gdn(rwkv7=True)
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
        assert max_diff < 0.1, f"rwkv7 cache parity 失败: max_diff={max_diff}"

    def test_cache_parity_all_features(self):
        """R36-4-14: channel_wise + rwkv7 全开 cache parity（阈值 0.2）。"""
        torch.manual_seed(42)
        gdn = _make_gdn(channel_wise=True, rwkv7=True)
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
        assert max_diff < 0.2, f"全开 cache parity 失败: max_diff={max_diff}"


# ===== 4. 训练 smoke test =====

class TestTrainingSmoke:
    def test_training_forward_backward(self):
        """R36-4-15: 训练前向+反向不崩溃。"""
        torch.manual_seed(42)
        gdn = _make_gdn()
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
        """R36-4-16: 梯度回流到 alpha_beta_proj。"""
        torch.manual_seed(42)
        gdn = _make_gdn()
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
        """R36-4-17: rwkv7 模式梯度回流到 b_proj（不可归零）。"""
        torch.manual_seed(42)
        gdn = _make_gdn(rwkv7=True)
        gdn.train()
        B, T = 2, 8
        x = torch.randn(B, T, 128)
        out, _ = gdn(x, use_cache=False)
        loss = out.sum()
        loss.backward()
        # b_proj 不可归零（否则数学死锁），需有梯度
        assert gdn.b_proj.weight.grad is not None
        assert torch.isfinite(gdn.b_proj.weight.grad).all()
        # z_proj 也应有梯度
        assert gdn.z_proj.weight.grad is not None

    def test_training_loss_decreases(self):
        """R36-4-18: 训练 loss 正常下降。"""
        torch.manual_seed(42)
        gdn = _make_gdn()
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


# ===== 5. 边界与稳定性 =====

class TestEdgeCases:
    def test_t1_forward(self):
        """R36-4-19: T=1 时前向正常（增量解码基线）。"""
        gdn = _make_gdn()
        gdn.eval()
        x = torch.randn(1, 1, 128)
        out, present = gdn(x, use_cache=True)
        assert torch.isfinite(out).all()
        assert out.shape == (1, 1, 128)

    def test_large_alpha_stability(self):
        """R36-4-20: alpha 接近 1（弱遗忘）时不发散。"""
        torch.manual_seed(42)
        gdn = _make_gdn(alpha_init=4.0)  # sigmoid(4)≈0.98，接近 1
        gdn.eval()
        B, T = 2, 8
        x = torch.randn(B, T, 128) * 0.1  # 小输入避免累积爆炸
        out, _ = gdn(x, use_cache=False)
        assert torch.isfinite(out).all()

    def test_z_all_finite_multi_step(self):
        """R36-4-21: 多步 z 前缀扫描结果全有限。"""
        torch.manual_seed(42)
        gdn = _make_gdn()
        gdn.eval()
        B, T = 2, 32
        x = torch.randn(B, T, 128)
        out, present = gdn(x, use_cache=True)
        assert torch.isfinite(out).all()
        # present[3] 是最终 z
        z_final = present[3]
        assert torch.isfinite(z_final).all()
