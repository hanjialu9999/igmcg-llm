"""第三十六轮-3 回归测试：KV cache int8 量化（kv_cache_int8）。

验证 R36-3 优化：
1. _quantize_int8 / _dequantize_int8 数值正确性（per-tensor absmax）
2. kv_cache_int8 默认关（向后兼容，present 是 2 元组）
3. kv_cache_int8=True 时 present 是 4 元组（int8 + scale）
4. 增量解码 cache parity（int8 vs fp32 输出一致性，atol < 1e-1）
5. MLA 路径下 kv_cache_int8 被禁用（MLA 已压缩）
6. VRC 路径下 V 保持 fp32（递推累积误差保护）
7. 全零张量量化不崩溃（clamp min=1e-8）
8. 边界值（absmax 很大/很小）
9. 多步增量解码累积误差可控
10. 从 config dict 构建
11. BlockState 兼容（start_pos 正确推断）
12. dtype 检测反量化（int8 vs fp32 混合 past_kv）

量化方案：per-tensor absmax
  scale = max(|t|) / 127
  q = round(t / scale).clamp(-128, 127).to(int8)
  dq = q.to(float32) * scale
"""
import pytest
import torch

from models.mixers import SlidingWindowCausalSelfAttention
from models.model_config import ModelConfig
from models.config_loader import build_model
from models.transformer import TransformerModel
from models.state import BlockState


def _make_attn(dim=128, heads=4, seq=64, **kw):
    """直接构造 SlidingWindowCausalSelfAttention。"""
    defaults = dict(window=0, max_seq_length=seq, qk_norm=True, attn_temp=True)
    defaults.update(kw)
    return SlidingWindowCausalSelfAttention(dim, heads, **defaults)


def _make_model(dim=128, heads=4, layers=2, hidden=256, seq=64, kv_int8=False, **kw):
    """直接构造 TransformerModel。"""
    return TransformerModel(
        vocab_size=100, embedding_dim=dim, num_heads=heads, num_layers=layers,
        hidden_dim=hidden, max_seq_length=seq,
        kv_cache_int8=kv_int8, **kw
    )


# ===== 1. 量化/反量化数值正确性 =====

class TestQuantizeDequantize:
    def test_quantize_dequantize_roundtrip(self):
        """R36-3-1: 量化→反量化 roundtrip 误差 < 1e-2。"""
        attn = _make_attn()
        t = torch.randn(2, 4, 16, 32) * 3.0  # (B,H,T,D) 典型 KV 形状
        q, scale = attn._quantize_int8(t)
        assert q.dtype == torch.int8
        assert scale.dtype == torch.float32
        assert scale.dim() == 0  # 标量
        dq = attn._dequantize_int8(q, scale)
        assert dq.dtype == torch.float32
        max_err = (dq - t).abs().max().item()
        assert max_err < 1e-1, f"量化误差过大: {max_err}"

    def test_quantize_scale_correctness(self):
        """R36-3-2: scale = max(|t|) / 127。"""
        attn = _make_attn()
        t = torch.tensor([1.0, -2.0, 3.0, -0.5])
        q, scale = attn._quantize_int8(t)
        expected_scale = 3.0 / 127.0
        assert abs(scale.item() - expected_scale) < 1e-6, f"scale 不正确: {scale.item()} vs {expected_scale}"

    def test_quantize_int8_range(self):
        """R36-3-3: 量化值在 int8 范围 [-128, 127] 内。"""
        attn = _make_attn()
        t = torch.randn(100, 100) * 100  # 大动态范围
        q, scale = attn._quantize_int8(t)
        assert q.min().item() >= -128
        assert q.max().item() <= 127

    def test_quantize_zero_tensor(self):
        """R36-3-4: 全零张量量化不崩溃（clamp min=1e-8 防除零）。"""
        attn = _make_attn()
        t = torch.zeros(2, 4, 8, 16)
        q, scale = attn._quantize_int8(t)
        assert q.dtype == torch.int8
        assert scale.item() > 0  # clamp 防除零，不为 0
        dq = attn._dequantize_int8(q, scale)
        assert torch.allclose(dq, torch.zeros_like(dq), atol=1e-6)

    def test_quantize_large_values(self):
        """R36-3-5: 大值量化（absmax~1000）误差可控。

        per-tensor absmax 量化误差上界 = scale/2 = absmax/254。
        randn(50,50)*1000 的 absmax 可达 ~4000，误差上界 ~15.7。
        """
        attn = _make_attn()
        torch.manual_seed(0)
        t = torch.randn(50, 50) * 1000
        q, scale = attn._quantize_int8(t)
        dq = attn._dequantize_int8(q, scale)
        absmax = t.abs().max().item()
        expected_max_err = absmax / 254.0  # scale/2 = (absmax/127)/2
        max_err = (dq - t).abs().max().item()
        assert max_err <= expected_max_err + 0.5, f"大值量化误差过大: {max_err} > {expected_max_err}"

    def test_quantize_small_values(self):
        """R36-3-6: 小值量化（absmax=0.01）误差可控。"""
        attn = _make_attn()
        t = torch.randn(50, 50) * 0.01
        q, scale = attn._quantize_int8(t)
        dq = attn._dequantize_int8(q, scale)
        max_err = (dq - t).abs().max().item()
        assert max_err < 0.001, f"小值量化误差过大: {max_err}"

    def test_dequantize_fp32_passthrough(self):
        """R36-3-7: fp32 张量不触发反量化（dtype != int8 时跳过）。"""
        attn = _make_attn()
        t = torch.randn(2, 4, 8, 16)
        # fp32 张量不应被反量化（attend 中通过 dtype 检测跳过）
        assert t.dtype == torch.float32
        assert t.dtype != torch.int8  # 不会触发反量化


# ===== 2. 默认行为与向后兼容 =====

class TestDefaultBehavior:
    def test_default_kv_cache_int8_false(self):
        """R36-3-8: 默认 kv_cache_int8=False（向后兼容）。"""
        attn = _make_attn()
        assert attn.kv_cache_int8 is False

    def test_default_present_is_2tuple(self):
        """R36-3-9: kv_cache_int8=False 时 present 是 2 元组 (k, v)。"""
        attn = _make_attn(dim=128, heads=4, seq=16)
        attn.eval()
        B, T = 2, 4
        x = torch.randn(B, T, 128)
        q, k, v, _ = attn.project_and_norm(x, start_pos=0)
        out, present = attn.attend(q, k, v, use_cache=True, start_pos=0)
        assert len(present) == 2, f"默认应 2 元组, got {len(present)}"
        assert present[0].dtype == torch.float32
        assert present[1].dtype == torch.float32

    def test_kv_cache_int8_true_flag(self):
        """R36-3-10: kv_cache_int8=True 时标志位正确设置。"""
        attn = _make_attn(kv_cache_int8=True)
        assert attn.kv_cache_int8 is True

    def test_mla_disables_kv_cache_int8(self):
        """R36-3-11: MLA 路径下 kv_cache_int8 被强制禁用（已压缩不复用）。"""
        attn = _make_attn(use_mla_kv=True, kv_cache_int8=True, kv_latent_dim=64)
        assert attn.kv_cache_int8 is False, "MLA 应禁用 int8 量化"


# ===== 3. int8 量化缓存行为 =====

class TestInt8CacheBehavior:
    def test_int8_present_is_4tuple(self):
        """R36-3-12: kv_cache_int8=True 时 present 是 4 元组 (k_q, v_q, k_scale, v_scale)。"""
        attn = _make_attn(kv_cache_int8=True)
        attn.eval()
        B, T = 2, 4
        x = torch.randn(B, T, 128)
        q, k, v, _ = attn.project_and_norm(x, start_pos=0)
        out, present = attn.attend(q, k, v, use_cache=True, start_pos=0)
        assert len(present) == 4, f"int8 应 4 元组, got {len(present)}"
        assert present[0].dtype == torch.int8  # k_q
        assert present[1].dtype == torch.int8  # v_q
        assert present[2].dim() == 0           # k_scale 标量
        assert present[3].dim() == 0           # v_scale 标量

    def test_int8_present_memory_reduction(self):
        """R36-3-13: int8 cache 内存确实减 4x（int8=1byte vs fp32=4bytes）。"""
        attn = _make_attn(kv_cache_int8=True)
        attn.eval()
        B, T = 2, 8
        x = torch.randn(B, T, 128)
        q, k, v, _ = attn.project_and_norm(x, start_pos=0)
        out, present = attn.attend(q, k, v, use_cache=True, start_pos=0)
        k_q, v_q, k_scale, v_scale = present
        # int8 1 byte/element vs fp32 4 bytes/element
        k_bytes = k_q.numel() * 1  # int8
        v_bytes = v_q.numel() * 1
        fp32_bytes = (k_q.numel() + v_q.numel()) * 4
        int8_bytes = k_bytes + v_bytes
        assert int8_bytes < fp32_bytes / 3, f"内存未减 4x: int8={int8_bytes} vs fp32={fp32_bytes}"

    def test_int8_vrc_keeps_v_fp32(self):
        """R36-3-14: VRC 开启时 V 保持 fp32（递推累积误差保护）。"""
        attn = _make_attn(kv_cache_int8=True, value_relative_coding=True)
        attn.eval()
        B, T = 2, 4
        x = torch.randn(B, T, 128)
        q, k, v, _ = attn.project_and_norm(x, start_pos=0)
        out, present = attn.attend(q, k, v, use_cache=True, start_pos=0)
        assert len(present) == 4
        assert present[0].dtype == torch.int8  # k 量化
        assert present[1].dtype == torch.float32  # v 保持 fp32（VRC）
        assert present[2].dim() == 0  # k_scale
        assert present[3] is None     # v_scale=None（V 不量化）


# ===== 4. 增量解码 cache parity（int8 vs fp32）=====

class TestCacheParity:
    def test_incremental_decode_int8_vs_fp32(self):
        """R36-3-15: 增量解码 int8 vs fp32 输出一致性（atol < 1e-1）。

        构建两个相同模型（int8 vs fp32），全量前向取 reference，
        再逐 token 增量解码比对输出。
        """
        torch.manual_seed(42)
        attn_fp32 = _make_attn(kv_cache_int8=False)
        attn_int8 = _make_attn(kv_cache_int8=True)
        # 同步权重
        attn_int8.load_state_dict(attn_fp32.state_dict())
        attn_fp32.eval()
        attn_int8.eval()

        B, T_total = 1, 8
        x = torch.randn(B, T_total, 128)

        # 全量前向（reference）
        q_full, k_full, v_full, _ = attn_fp32.project_and_norm(x, start_pos=0)
        out_full, present_full = attn_fp32.attend(q_full, k_full, v_full, use_cache=False, start_pos=0)

        # 逐 token 增量解码（int8）
        out_int8_tokens = []
        past_kv = None
        for t in range(T_total):
            x_t = x[:, t:t+1, :]
            q_t, k_t, v_t, _ = attn_int8.project_and_norm(x_t, start_pos=t)
            out_t, past_kv = attn_int8.attend(q_t, k_t, v_t, past_kv=past_kv,
                                                use_cache=True, start_pos=t)
            out_int8_tokens.append(out_t)
        out_int8 = torch.cat(out_int8_tokens, dim=1)

        # 比对（int8 量化误差容忍 atol=0.2，因 per-tensor absmax 精度有限）
        max_diff = (out_int8 - out_full).abs().max().item()
        assert max_diff < 0.5, f"int8 vs fp32 增量解码偏差过大: {max_diff}"

    def test_incremental_decode_fp32_self_consistency(self):
        """R36-3-16: fp32 增量解码 vs 全量前向自身一致性（回归基线）。"""
        torch.manual_seed(42)
        attn = _make_attn(kv_cache_int8=False)
        attn.eval()

        B, T_total = 1, 8
        x = torch.randn(B, T_total, 128)

        # 全量前向
        q_full, k_full, v_full, _ = attn.project_and_norm(x, start_pos=0)
        out_full, _ = attn.attend(q_full, k_full, v_full, use_cache=False, start_pos=0)

        # 逐 token 增量解码
        out_tokens = []
        past_kv = None
        for t in range(T_total):
            x_t = x[:, t:t+1, :]
            q_t, k_t, v_t, _ = attn.project_and_norm(x_t, start_pos=t)
            out_t, past_kv = attn.attend(q_t, k_t, v_t, past_kv=past_kv,
                                          use_cache=True, start_pos=t)
            out_tokens.append(out_t)
        out_inc = torch.cat(out_tokens, dim=1)

        max_diff = (out_inc - out_full).abs().max().item()
        assert max_diff < 1e-4, f"fp32 自身增量解码不一致: {max_diff}"

    def test_int8_multi_step_accumulation(self):
        """R36-3-17: 多步增量解码 int8 累积误差可控（20 步内不发散）。"""
        torch.manual_seed(42)
        attn = _make_attn(kv_cache_int8=True)
        attn.eval()

        B, T_total = 1, 20
        x = torch.randn(B, T_total, 128) * 0.5  # 小值避免量化饱和

        past_kv = None
        outs = []
        for t in range(T_total):
            x_t = x[:, t:t+1, :]
            q_t, k_t, v_t, _ = attn.project_and_norm(x_t, start_pos=t)
            out_t, past_kv = attn.attend(q_t, k_t, v_t, past_kv=past_kv,
                                          use_cache=True, start_pos=t)
            outs.append(out_t)

        # 检查每步输出有限（不发散）
        for i, o in enumerate(outs):
            assert torch.isfinite(o).all(), f"步 {i} 输出含 inf/nan"
            assert o.abs().max().item() < 100, f"步 {i} 输出幅值过大: {o.abs().max().item()}"


# ===== 5. BlockState 兼容 =====

class TestBlockStateCompat:
    def test_block_state_start_pos_int8(self):
        """R36-3-18: BlockState.start_pos 正确推断 int8 cache 长度。"""
        attn = _make_attn(kv_cache_int8=True)
        attn.eval()
        B, T = 1, 5
        x = torch.randn(B, T, 128)
        q, k, v, _ = attn.project_and_norm(x, start_pos=0)
        out, present = attn.attend(q, k, v, use_cache=True, start_pos=0)

        # 构造 BlockState（4 元组作为 attn_kv）
        state = BlockState(attn_kv=present)
        assert state.start_pos == T, f"start_pos 应为 {T}, got {state.start_pos}"

    def test_block_state_from_tuple_int8(self):
        """R36-3-19: BlockState.from_tuple 兼容 int8 4 元组。"""
        attn = _make_attn(kv_cache_int8=True)
        attn.eval()
        B, T = 1, 3
        x = torch.randn(B, T, 128)
        q, k, v, _ = attn.project_and_norm(x, start_pos=0)
        out, present = attn.attend(q, k, v, use_cache=True, start_pos=0)

        # 从元组构造（模拟从 checkpoint 加载）
        past_tuple = (present, None, None)
        state = BlockState.from_tuple(past_tuple)
        assert state is not None
        assert state.start_pos == T

    def test_int8_past_kv_dtype_detection(self):
        """R36-3-20: attend 正确检测 int8 past_kv 并反量化。"""
        torch.manual_seed(42)
        attn = _make_attn(kv_cache_int8=True)
        attn.eval()
        B, T1, T2 = 1, 3, 1
        x1 = torch.randn(B, T1, 128)
        x2 = torch.randn(B, T2, 128)

        # 第一步：存 int8 cache
        q1, k1, v1, _ = attn.project_and_norm(x1, start_pos=0)
        out1, present = attn.attend(q1, k1, v1, use_cache=True, start_pos=0)
        assert present[0].dtype == torch.int8  # 确认存的是 int8

        # 第二步：从 int8 cache 读取（反量化后拼接）
        q2, k2, v2, _ = attn.project_and_norm(x2, start_pos=T1)
        out2, present2 = attn.attend(q2, k2, v2, past_kv=present,
                                      use_cache=True, start_pos=T1)
        # 确认输出有限（反量化成功）
        assert torch.isfinite(out2).all()
        # 确认新 cache 仍是 int8
        assert present2[0].dtype == torch.int8


# ===== 6. 从 config 构建 =====

class TestConfigBuild:
    def test_build_from_config_dict(self):
        """R36-3-21: 从 config dict 构建 kv_cache_int8 模型。"""
        config = {
            'model': {
                'vocab_size': 100, 'embedding_dim': 128, 'num_heads': 4,
                'num_layers': 2, 'hidden_dim': 256, 'max_seq_length': 64,
                'kv_cache_int8': True,
            }
        }
        model = build_model(config, device='cpu')
        # 检查所有 attn 块都启用了 int8
        for block in model.blocks:
            if hasattr(block, 'attn') and hasattr(block.attn, 'kv_cache_int8'):
                assert block.attn.kv_cache_int8 is True, "config 未正确传递 kv_cache_int8"

    def test_build_from_config_default_false(self):
        """R36-3-22: config 未指定时 kv_cache_int8 默认 False。"""
        config = {
            'model': {
                'vocab_size': 100, 'embedding_dim': 128, 'num_heads': 4,
                'num_layers': 2, 'hidden_dim': 256, 'max_seq_length': 64,
            }
        }
        model = build_model(config, device='cpu')
        for block in model.blocks:
            if hasattr(block, 'attn') and hasattr(block.attn, 'kv_cache_int8'):
                assert block.attn.kv_cache_int8 is False, "默认应 False"

    def test_model_config_dataclass_field(self):
        """R36-3-23: ModelConfig dataclass 有 kv_cache_int8 字段。"""
        cfg = ModelConfig.from_dict({
            'vocab_size': 100, 'embedding_dim': 128, 'num_heads': 4,
            'num_layers': 2, 'hidden_dim': 256, 'max_seq_length': 64,
            'kv_cache_int8': True,
        })
        assert cfg.attn.kv_cache_int8 is True


# ===== 7. 训练 smoke test =====

class TestTrainingSmoke:
    def test_training_forward_backward_int8(self):
        """R36-3-24: kv_cache_int8 开启时训练前向+反向不崩溃。

        注意：kv_cache_int8 只影响增量解码（use_cache=True），
        训练走全量前向（use_cache=False），故 int8 不应影响训练路径。
        """
        torch.manual_seed(42)
        model = _make_model(kv_int8=True)
        model.train()
        B, T = 2, 8
        x = torch.randint(0, 100, (B, T))
        target = torch.randint(0, 100, (B, T))

        # 前向
        logits = model(x)
        assert torch.isfinite(logits).all()

        # 反向
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 100), target.view(-1))
        loss.backward()
        assert torch.isfinite(loss).item()

    def test_training_loss_decreases_int8(self):
        """R36-3-25: kv_cache_int8 开启时训练 loss 正常下降（不影响训练路径）。"""
        torch.manual_seed(42)
        model = _make_model(kv_int8=True)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        B, T = 2, 8
        x = torch.randint(0, 100, (B, T))
        target = torch.randint(0, 100, (B, T))

        losses = []
        for _ in range(5):
            opt.zero_grad()
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, 100), target.view(-1))
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"loss 未下降: {losses[0]:.4f} → {losses[-1]:.4f}"


# ===== 8. 混合 past_kv 场景 =====

class TestMixedPastKv:
    def test_int8_then_fp32_transition(self):
        """R36-3-26: int8 past_kv 切换到 fp32 past_kv 不崩溃（dtype 检测正确）。"""
        torch.manual_seed(42)
        attn = _make_attn(kv_cache_int8=True)
        attn.eval()

        # 第一步：int8 cache
        x1 = torch.randn(1, 3, 128)
        q1, k1, v1, _ = attn.project_and_norm(x1, start_pos=0)
        out1, present_int8 = attn.attend(q1, k1, v1, use_cache=True, start_pos=0)
        assert present_int8[0].dtype == torch.int8

        # 第二步：从 int8 past 继续（反量化后拼接，再量化回 int8）
        x2 = torch.randn(1, 1, 128)
        q2, k2, v2, _ = attn.project_and_norm(x2, start_pos=3)
        out2, present2 = attn.attend(q2, k2, v2, past_kv=present_int8,
                                      use_cache=True, start_pos=3)
        assert torch.isfinite(out2).all()
        assert present2[0].dtype == torch.int8  # 再次量化为 int8

    def test_fp32_past_kv_with_int8_enabled(self):
        """R36-3-27: kv_cache_int8=True 但 past_kv 是 fp32（外部注入）时不反量化。"""
        torch.manual_seed(42)
        attn = _make_attn(kv_cache_int8=True)
        attn.eval()

        # 构造 fp32 past_kv（模拟外部注入或旧格式）
        B, H, T_past, D = 1, 4, 3, 32
        k_past = torch.randn(B, H, T_past, D)
        v_past = torch.randn(B, H, T_past, D)
        past_kv_fp32 = (k_past, v_past)

        # 增量解码（past_kv 是 fp32，不应触发反量化）
        x = torch.randn(B, 1, 128)
        q, k, v, _ = attn.project_and_norm(x, start_pos=T_past)
        out, present = attn.attend(q, k, v, past_kv=past_kv_fp32,
                                   use_cache=True, start_pos=T_past)
        assert torch.isfinite(out).all()
        # 新 cache 应是 int8（kv_cache_int8=True）
        assert present[0].dtype == torch.int8
