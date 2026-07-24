from __future__ import annotations
import math
import threading
from typing import Optional, Dict, Tuple, Any
import torch
import torch.nn as nn
from models.constants import ROPE_BASE


# 第十九轮：YaRN 长度外推辅助函数（YaRN 论文 arXiv:2309.00071）
def _yarn_find_correction_dim(num_rotations: int, dim: int, base: float, max_position_embeddings: int) -> float:
    """计算给定旋转数对应的修正维度。"""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(low_rot: int, high_rot: int, dim: int, base: float,
                                 max_position_embeddings: int) -> Tuple[int, int]:
    """计算修正范围 [low, high]。"""
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(min_val: float, max_val: float, dim: int) -> torch.Tensor:
    """线性斜坡掩码：min_val 处为 0，max_val 处为 1，中间线性插值。"""
    if min_val == max_val:
        max_val += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    return torch.clamp(linear_func, 0, 1)


class RotaryEmbedding(nn.Module):
    """旋转位置编码 RoPE：对 Q/K 按位置旋转，天然支持长度外推。
    
    使用实例级缓存（而非模块级全局缓存），避免多线程/多设备冲突。
    提供可选的类级共享缓存（带锁）供性能敏感场景使用。
    """
    # 类级共享缓存（可选，需显式启用）
    _shared_cache: Dict[Tuple[str, str, int], Tuple[torch.Tensor, torch.Tensor]] = {}
    _shared_cache_lock = threading.RLock()
    _use_shared_cache = False

    def __init__(self, dim: int, base: float = ROPE_BASE, learnable: bool = False,
                 dim_fraction: float = 1.0,
                 yarn_scale: float = 1.0, yarn_beta: float = 0.1,
                 yarn_orig_max_seq_length: int = 0,
                 dim_wise: bool = False):
        super().__init__()
        # 第十五轮：Partial RoPE——仅前 dim*fraction 维度加 RoPE，后段 NoPE（纯内容维度）。
        # 灵感：Qwen3-Next（前 25%）+ MLA Decoupled RoPE。长度外推更稳，高频维度不承载位置信息。
        # dim_fraction=1.0 时全维旋转，完全向后兼容；<1.0 时后段不旋转。
        # 注意：rot_dim 必须是偶数（RoPE 按维度对旋转），向下取偶。
        self.dim_fraction = float(dim_fraction)
        self.rot_dim = max(2, int(dim * self.dim_fraction) // 2 * 2)  # 向下取偶，最少 2
        self.no_pe_dim = dim - self.rot_dim  # 不旋转的维度数（可为 0）
        # 第十九轮：YaRN 长度外推——非均匀频率缩放，免训练外推到更长序列
        # 灵感：YaRN (arXiv:2309.00071) + Randomized YaRN (ICLR 2026)
        # yarn_scale=1.0 时不缩放（向后兼容）；>1.0 时高频维度不缩放（外推）、低频维度缩放（插值）
        self.yarn_scale = float(yarn_scale)
        self.yarn_beta = float(yarn_beta)
        self.yarn_orig_max_seq_length = int(yarn_orig_max_seq_length)
        if self.yarn_scale > 1.0:
            inv_freq = self._compute_yarn_inv_freq(base)
        else:
            inv_freq = 1.0 / (base ** (torch.arange(0, self.rot_dim, 2).float() / self.rot_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.learnable = learnable
        # 阶段5：可学习 RoPE 频率缩放——让模型调整各频率尺度，更好适应长度/尺度。
        # inv_freq 实际 = buffer * exp(log_scale)，log_scale 每维可学（init 0 = 不变）。
        if learnable:
            self.rope_log_scale = nn.Parameter(torch.zeros(self.rot_dim // 2))
        # 第二十轮：维度级 RoPE 动态分配——逐维度对选择旋转/不旋转
        # 灵感：DPE (arXiv:2504.18857) + LongRoPE2 (ICML 2025)
        # 升级 Partial RoPE 的"前缀连续切分"为"逐维度离散分配"：
        # mask = sigmoid(dim_wise_logit)，mask≈1 旋转，mask≈0 不旋转（cos=1,sin=0）
        # init logit=0 → sigmoid=0.5 半旋转起步，训练中学习哪些维度对应旋转/不旋转
        # 与 Partial RoPE 正交：Partial 选择前缀维度对，dim_wise 在 rot_dim 内逐维度对选择
        # 与 YaRN 正交：YaRN 缩放 inv_freq 基底，dim_wise 决定是否启用旋转
        self.dim_wise_enabled = dim_wise
        if dim_wise:
            self.dim_wise_logit = nn.Parameter(torch.zeros(self.rot_dim // 2))
        # 实例级缓存：隔离不同模型/设备/dtype
        self._cache: Dict[Tuple[str, str, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._cache_lock = threading.RLock()

    @classmethod
    def enable_shared_cache(cls, enabled: bool = True):
        """启用/禁用类级共享缓存（跨实例共享，带锁保护）。"""
        cls._use_shared_cache = enabled
        if not enabled:
            cls._shared_cache.clear()

    def _compute_yarn_inv_freq(self, base: float) -> torch.Tensor:
        """YaRN 三段式非均匀频率缩放（第十九轮）。

        高频维度（短波长，index 小）：保持外推（不缩放），保留近距离位置精度。
        低频维度（长波长，index 大）：插值缩放（1/scale），支持更长序列。
        中间维度：线性过渡。

        与 Partial RoPE 正交：YaRN 缩放 rot_dim 维的频率，NoPE 维度不受影响。
        与可学习 RoPE 正交：YaRN 修改 inv_freq buffer 基底，rope_log_scale 在此之上微调。
        """
        orig_len = self.yarn_orig_max_seq_length or 2048
        dim = self.rot_dim
        # 计算修正范围：高频边界（low_rot=32）和低频边界（high_rot=1）
        low, high = _yarn_find_correction_range(32, 1, dim, base, orig_len)
        # 外推频率（不缩放）和插值频率（缩放 1/scale）
        freq_indices = torch.arange(0, dim, 2).float()
        inv_freq_extrapolation = 1.0 / (base ** (freq_indices / dim))
        inv_freq_interpolation = 1.0 / (self.yarn_scale * base ** (freq_indices / dim))
        # 掩码：高频维度 mask≈1（用外推），低频维度 mask≈0（用插值）
        mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2)
        inv_freq = inv_freq_interpolation * (1 - mask) + inv_freq_extrapolation * mask
        return inv_freq

    def _get_cos_sin(self, start_pos: int, seq_len: int, device: torch.device, dtype: torch.dtype, max_len: int = 2048) -> Tuple[torch.Tensor, torch.Tensor]:
        """按 (device, dtype, head_dim) 缓存整张位置表后按需切片。

        实例级缓存避免多线程/多设备污染；可选共享缓存用于性能敏感场景。

        注意（2026-07-18 修复 rope_learnable 训练崩溃）：缓存只存「无 grad 的基准表」
        （由 inv_freq buffer 计算，scale=1）。可学习路径在此之上按 rope_log_scale
        重新计算 cos/sin（带 grad），保证梯度回流到 rope_log_scale，且绝不跨 step
        复用带 grad 图的张量（否则多步训练 backward 触发「graph freed」错误）。
        """
        key = (str(device), str(dtype), self.inv_freq.shape[0])
        need = start_pos + seq_len

        # 非可学：直接返回缓存（基准表本身无 grad，可安全复用）
        if not self.learnable:
            with self._cache_lock:
                cached = self._cache.get(key)
                if cached is not None and cached[0].size(2) >= need:
                    cos_full, sin_full = cached
                    return cos_full[:, :, start_pos:need, :].to(dtype), sin_full[:, :, start_pos:need, :].to(dtype)
            if self._use_shared_cache:
                with self._shared_cache_lock:
                    cached = self._shared_cache.get(key)
                    if cached is not None and cached[0].size(2) >= need:
                        cos_full, sin_full = cached
                        return cos_full[:, :, start_pos:need, :].to(dtype), sin_full[:, :, start_pos:need, :].to(dtype)

        # 计算新缓存（基准表，由 inv_freq buffer 算，detach 确保无 grad 历史）
        L = min(max(need, 128), max_len)
        if need > max_len:
            import warnings
            warnings.warn(f'RoPE: need={need} > max_len={max_len}，位置将被 clamp，'
                          '可能影响长序列质量。建议增大 rope_max_len。')
        t = torch.arange(0, L, device=device).type_as(self.inv_freq)
        # 基准频率表（无 grad，inv_freq 为 buffer）
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_full = emb.cos()[None, None, :, :].to(dtype).detach()
        sin_full = emb.sin()[None, None, :, :].to(dtype).detach()

        # 存入实例缓存（仅非可学路径后续会命中；可学路径每步重算，不依赖此缓存）
        with self._cache_lock:
            self._cache[key] = (cos_full, sin_full)
        if self._use_shared_cache:
            with self._shared_cache_lock:
                self._shared_cache[key] = (cos_full, sin_full)

        # 可学习路径：按 rope_log_scale 重新计算 cos/sin（带 grad，梯度回流 rope_log_scale）
        if self.learnable:
            t_eff = torch.arange(start_pos, need, device=device).type_as(self.inv_freq)
            inv_freq_eff = self.inv_freq * torch.exp(self.rope_log_scale).to(self.inv_freq.device)
            freqs_eff = torch.outer(t_eff, inv_freq_eff)
            emb_eff = torch.cat((freqs_eff, freqs_eff), dim=-1)
            cos = emb_eff.cos()[None, None, :, :].to(dtype)
            sin = emb_eff.sin()[None, None, :, :].to(dtype)
            return cos, sin

        return cos_full[:, :, start_pos:need, :].to(dtype), sin_full[:, :, start_pos:need, :].to(dtype)

    def _apply_dim_wise_mask(self, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """维度级 RoPE 掩码：对 cos/sin 逐维度对施加可学软掩码。

        mask = sigmoid(dim_wise_logit)，shape [rot_dim//2]。
        cos_masked = cos * mask + (1 - mask)  → mask=0 时 cos=1（不旋转）
        sin_masked = sin * mask               → mask=0 时 sin=0（不旋转）
        cos/sin shape (... , rot_dim)，mask 按 rot_dim//2 广播后重复到 rot_dim。
        """
        if not self.dim_wise_enabled:
            return cos, sin
        mask = torch.sigmoid(self.dim_wise_logit)  # (rot_dim//2,)
        # cos/sin shape (..., rot_dim)，需要 mask shape (..., rot_dim) 广播
        # cos[..., :d] 和 cos[..., d:2d] 用同一个 mask（每对维度共享）
        d = self.dim_wise_logit.shape[0]
        mask_full = torch.cat([mask, mask], dim=-1)  # (rot_dim,)
        # 广播到 cos/sin 的前导维度
        cos = cos * mask_full + (1.0 - mask_full)
        sin = sin * mask_full
        return cos, sin

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0, max_len: int = 2048) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._get_cos_sin(start_pos, q.size(2), q.device, q.dtype, max_len=max_len)
        cos, sin = self._apply_dim_wise_mask(cos, sin)
        return self._rope_apply(q, cos, sin), self._rope_apply(k, cos, sin)

    def apply_to_single(self, x: torch.Tensor, start_pos: int = 0, max_len: int = 2048) -> torch.Tensor:
        """对单个张量应用 RoPE（第十七轮 MLA 支持）。

        MLA 场景：q 在 project_and_norm 中应用 RoPE（位置 start_pos），
        k 在 attend 内部还原后应用 RoPE（位置 0..T_total-1，拼接后）。
        与 forward(q,k) 的区别：forward 假设 q,k 同长度（project_and_norm 中未拼接），
        apply_to_single 支持独立长度和独立 start_pos（attend 中 k 是拼接后的 T_total）。
        """
        cos, sin = self._get_cos_sin(start_pos, x.size(2), x.device, x.dtype, max_len=max_len)
        cos, sin = self._apply_dim_wise_mask(cos, sin)
        return self._rope_apply(x, cos, sin)

    @staticmethod
    def _rope_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # 第十五轮：Partial RoPE——支持部分维度旋转。
        # cos/sin 的最后一维 = rot_dim（旋转维度数），x 最后一维 = dim（可能 > rot_dim）。
        # 前 rot_dim 维做旋转，后 no_pe_dim 维不变（NoPE 纯内容维度）。
        rot_dim = cos.size(-1)
        no_pe_dim = x.size(-1) - rot_dim
        if no_pe_dim > 0:
            # Partial RoPE：前段旋转，后段不变
            x_rot = x[..., :rot_dim]
            x_pass = x[..., rot_dim:]
            d = rot_dim // 2
            x1, x2 = x_rot[..., :d], x_rot[..., d:]
            cos_half = cos[..., :d]
            sin_half = sin[..., :d]
            x_rotated = torch.cat([
                x1 * cos_half - x2 * sin_half,
                x1 * sin_half + x2 * cos_half,
            ], dim=-1)
            return torch.cat([x_rotated, x_pass], dim=-1)
        # 全维旋转（原路径，dim_fraction=1.0）
        d = rot_dim // 2
        x1, x2 = x[..., :d], x[..., d:]
        cos_half = cos[..., :d]
        sin_half = sin[..., :d]
        return torch.cat([
            x1 * cos_half - x2 * sin_half,
            x1 * sin_half + x2 * cos_half,
        ], dim=-1)

    def clear_cache(self):
        """清空实例缓存（长时间运行时可调用防止内存泄漏）。"""
        with self._cache_lock:
            self._cache.clear()
