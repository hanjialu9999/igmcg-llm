from __future__ import annotations

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.functional import scaled_dot_product_attention
from typing import Optional, List, Tuple, Any, Dict, Callable

from models.constants import MASK_FILL_VALUE, ROPE_BASE


from models.norms import RMSNorm
from models.rope import RotaryEmbedding
from models.memory import MemoryBank
from models.mixers import (SlidingWindowCausalSelfAttention, LinearAttention,
                           AxialLinearAttention, DifferentialAttention, MambaSSM,
                           MambaSSMWithCAST, SwiGLU, apply_qk_norm_and_temp,
                           GatedDeltaNet)
from models.sampling import (apply_repetition_penalty, sample_next_token,
                             _decode_one_step)
from models.layers import CharMergeLayer
from models.state import BlockState
from models.gates import (GateConfig, apply_direct, apply_sigmoid_scalar,
                          apply_linear_gate, convex_combine_scalar,
                          convex_combine_linear, apply_correction)

# 向后兼容：保留原模块级符号的外部可见性（其它模块仍从 models.transformer 导入）
__all__ = [
    "RMSNorm", "RotaryEmbedding", "MemoryBank",
    "SlidingWindowCausalSelfAttention", "LinearAttention", "MambaSSM", "SwiGLU",
    "apply_qk_norm_and_temp", "apply_repetition_penalty", "sample_next_token",
    "_decode_one_step", "CharMergeLayer", "BlockState",
    "TransformerBlock", "_parse_layer_plan", "TransformerModel",
    "GatedDeltaNet",
]


class TransformerBlock(nn.Module):
    """可配置混合块：attn / ssm / hybrid(attn+ssm 并行)。Pre-LN。"""
    def __init__(self, dim: int, num_heads: int, hidden_dim: int, block_type: str = 'attn',
                 dropout: float = 0.0, max_seq_length: int = 64,
                 ssm_kwargs: Optional[Dict[str, Any]] = None, attn_kwargs: Optional[Dict[str, Any]] = None,
                 gate_cfg: Optional[GateConfig] = None,
                 gradient_checkpointing: bool = True, mixer: str = 'attn',
                 shared_qkv: Optional[nn.Linear] = None,
                 shared_proj: Optional[nn.Linear] = None,
                 shared_ffn: Optional[SwiGLU] = None,
                 shared_lns: Optional[Tuple[RMSNorm, RMSNorm]] = None,
                 ssm_as_memory: bool = False,
                 zero_centered_norm: bool = False):
        super().__init__()
        self.block_type = block_type
        self.drop = nn.Dropout(dropout)
        ssm_kwargs = ssm_kwargs or {}
        attn_kwargs = attn_kwargs or {}
        # 第十六轮：门控配置统一收口为 GateConfig（替代 6 个散落 bool 参数）
        gate_cfg = gate_cfg or GateConfig()
        self.residual_gate_enabled = gate_cfg.residual_gate
        self.hybrid_gate_enabled = gate_cfg.hybrid_gate
        self.highway_gate_enabled = gate_cfg.highway_gate and gate_cfg.residual_gate
        self.linear_correction_enabled = gate_cfg.linear_correction
        self.skip_enabled = gate_cfg.skip
        # 运行时增强开关（按开关粒度，用于"交替/分段增强"训练）：默认全开
        self._rt: Dict[str, bool] = {"residual_gate": True, "hybrid_gate": True,
                                     "highway_gate": True, "layer_film": True}
        self.gradient_checkpointing = gradient_checkpointing
        if self.linear_correction_enabled and mixer == 'attn_linear':
            # init -1.0（sigmoid≈0.27）使初始修正行为接近原凸组合默认（mixer_gate init 1.0
            # → sigmoid≈0.73 → 0.73h+0.27lh），开启时平滑过渡，训练中自决修正强度。
            self.correction_gate = nn.Parameter(torch.tensor(-1.0))
        # 第十一轮：SSM 输出作隐式记忆——hybrid 块中先算 SSM，把 ssm_h 投影为
        # 单个"SSM 摘要"记忆槽注入注意力 mem_kv，让注意力能查到 SSM 的序列理解。
        # 仅 hybrid 块生效；head_dim 与 MemoryBank 一致（dim // num_heads）。
        self.ssm_as_memory_enabled = ssm_as_memory and block_type == 'hybrid'
        if self.ssm_as_memory_enabled:
            _head_dim = dim // num_heads
            self.ssm_k_proj = nn.Linear(dim, _head_dim, bias=False)
            self.ssm_v_proj = nn.Linear(dim, _head_dim, bias=False)
        # Both attn and ssm blocks need a pre-norm layer
        if shared_lns is not None:
            self.ln1, self.ln2 = shared_lns
        else:
            self.ln1 = RMSNorm(dim, zero_centered=zero_centered_norm)
        if block_type in ('attn', 'hybrid'):
            # 阶段7 token mixer 选择：attn(默认) / linear(纯线性注意力) /
            # linear2d(2D 轴向线性注意力, O(T·√T)) /
            # attn_linear(attn+线性注意力 两路并行，可学 mixer_gate 自选择比例)。
            # hybrid_linear2d(hybrid块用 linear2d 做 token mixer + SSM 并行，全链路 O(T·√T))。
            # 旧配置字符串 'hybrid' 等价于 'attn_linear'（向后兼容）。
            if mixer == 'hybrid':
                mixer = 'attn_linear'
            self.mixer = mixer
            if mixer == 'hybrid_linear2d':
                # hybrid 块 token mixer 用 linear2d（替代标准 attn），与 SSM 并行
                self.attn, self.linear_attn, self.mixer_gate = self._build_attn_mixer(
                    'linear2d', dim, num_heads, max_seq_length, attn_kwargs,
                    shared_qkv=shared_qkv, shared_proj=shared_proj)
            else:
                self.attn, self.linear_attn, self.mixer_gate = self._build_attn_mixer(
                    mixer, dim, num_heads, max_seq_length, attn_kwargs,
                    shared_qkv=shared_qkv, shared_proj=shared_proj)
        if block_type in ('ssm', 'hybrid'):
            _ssm_type = ssm_kwargs.pop('ssm_type', 'standard')
            if _ssm_type == 'cast':
                self.ssm = MambaSSMWithCAST(dim, **ssm_kwargs)
            else:
                self.ssm = MambaSSM(dim, **ssm_kwargs)
        self.ln2 = shared_lns[1] if shared_lns is not None else RMSNorm(dim, zero_centered=zero_centered_norm)
        self.ffn = shared_ffn if shared_ffn is not None else SwiGLU(dim, hidden_dim)
        # ②/⑥ 每层可学习残差门控：x = x + gate * f(x)（init 1.0，默认行为不变）
        # 第十三轮：highway_gate 与静态 residual_gate 互斥——highway_gate=True 时不创建
        # sub1_gate/ffn_gate（避免 dead params：highway 路径从不读取静态门）。
        if self.residual_gate_enabled and not self.highway_gate_enabled:
            # hybrid 块的第一子层用 hybrid_attn_gate/hybrid_ssm_gate，sub1_gate 无用，跳过分配
            if block_type != 'hybrid':
                self.sub1_gate = nn.Parameter(torch.ones(1))   # 第一子层残差（attn 或 ssm）
            self.ffn_gate = nn.Parameter(torch.ones(1))    # FFN 子层残差
        # 第十三轮：动态残差门控（highway_gate）——input-dependent gate 替代静态标量
        # residual_gate。gate = sigmoid(W·x + b)，x_out = x + gate·f(x)。
        # init: W=0, b=3.0 → sigmoid(3)≈0.95 → x + 0.95·f(x) ≈ 原 residual_gate=1.0 行为，
        # 平滑过渡；训练中模型逐 token 自决残差强度（不确定时 gate↓ 保留输入，有把握时 gate↑ 强变换）。
        if self.highway_gate_enabled:
            if block_type != 'hybrid':
                self.sub1_highway = nn.Linear(dim, 1)
            self.ffn_highway = nn.Linear(dim, 1)
        # ⭐A 混合块内 attn/ssm 两路可学习门控（init 1.0）
        if self.hybrid_gate_enabled and block_type == 'hybrid':
            self.hybrid_attn_gate = nn.Parameter(torch.ones(1))
            self.hybrid_ssm_gate = nn.Parameter(torch.ones(1))
        # 阶段8.4：单动态门控 g_t（默认关，向后兼容）。用 g_t=sigmoid(W_g·ln1(x)) 逐位置混合
        # attn 与 ssm（g_t·attn_h + (1-g_t)·ssm_h），替代原两独立标量门控相加（双残差、非真融合）。
        # 单门控是凸组合、逐位置动态、参数更少，架构更干净（见 §8 推进顺序 #4）。
        self.hybrid_single_gate = gate_cfg.hybrid_single_gate and block_type == 'hybrid'
        if self.hybrid_single_gate:
            self.hybrid_mix = nn.Linear(dim, 1)
        # 阶段6：可学习跳过层（skip gate）——sigmoid 门控，模型自决本层是否跳过。
        # skip≈1 走残差（等效跳过该块计算），推理时可按阈值静态剪枝省算力。
        if self.skip_enabled:
            self.skip_gate = nn.Parameter(torch.ones(1))  # init 1.0 = 不跳过（默认保留全部层）
        self._skip_active = True

    @staticmethod
    def _build_attn_mixer(mixer: str, dim: int, num_heads: int, max_seq_length: int,
                           attn_kwargs: Dict[str, Any],
                           shared_qkv: Optional[nn.Linear] = None,
                           shared_proj: Optional[nn.Linear] = None
                           ) -> Tuple[nn.Module, Optional[nn.Module], Optional[nn.Parameter]]:
        """构造 attn 系 token mixer（A 项归一点，消除 attn/hybrid 块中的重复构建逻辑）。

        shared_qkv/shared_proj：层间共享投影（share_attn_proj=True 时由 TransformerModel
        创建并传入，各层复用同一组 QKV/Output 投影参数，减少 ~40% 参数量）。

        Returns:
            attn: 主注意力模块（SlidingWindowCausalSelfAttention 或 LinearAttention）
            linear_attn: 并行线性注意力分支（仅 mixer='attn_linear' 时非 None）
            mixer_gate: attn/linear 两路混合门控（仅 mixer='attn_linear' 时非 None）
        """
        if mixer == 'linear':
            attn = LinearAttention(dim, num_heads, max_seq_length=max_seq_length,
                                   qk_norm=attn_kwargs.get('qk_norm', True),
                                   attn_temp=attn_kwargs.get('attn_temp', True),
                                   feature=attn_kwargs.get('linear_attn_feature', 'relu'),
                                   rope_dim_fraction=attn_kwargs.get('rope_dim_fraction', 1.0),
                                   shared_qkv=shared_qkv, shared_proj=shared_proj)
            return attn, None, None
        if mixer == 'gated_delta':
            # 第十五轮：Gated DeltaNet——delta rule + α/β 门控，长程检索更精确
            attn = GatedDeltaNet(dim, num_heads, max_seq_length=max_seq_length,
                                 qk_norm=attn_kwargs.get('qk_norm', True),
                                 attn_temp=attn_kwargs.get('attn_temp', True),
                                 feature=attn_kwargs.get('linear_attn_feature', 'relu'),
                                 rope_dim_fraction=attn_kwargs.get('rope_dim_fraction', 1.0),
                                 alpha_init=attn_kwargs.get('delta_alpha_init', -2.0),
                                 beta_init=attn_kwargs.get('delta_beta_init', 2.0),
                                 shared_qkv=shared_qkv, shared_proj=shared_proj)
            return attn, None, None
        if mixer == 'linear2d':
            # 2D 轴向线性注意力：O(T·√T)，适合序列长度为完全平方数或接近的场景
            attn = AxialLinearAttention(dim, num_heads, max_seq_length=max_seq_length,
                                        qk_norm=attn_kwargs.get('qk_norm', True),
                                        attn_temp=attn_kwargs.get('attn_temp', True),
                                        feature=attn_kwargs.get('linear_attn_feature', 'relu'),
                                        shared_qkv=shared_qkv, shared_proj=shared_proj)
            return attn, None, None
        if mixer == 'diff':
            # 差分注意力：两组注意力差值消除噪声，增强关键特征
            attn = DifferentialAttention(dim, num_heads, max_seq_length=max_seq_length,
                                         qk_norm=attn_kwargs.get('qk_norm', True),
                                         attn_temp=attn_kwargs.get('attn_temp', True),
                                         shared_qkv=shared_qkv, shared_proj=shared_proj)
            return attn, None, None
        if mixer == 'attn_linear':
            attn_only = {k: v for k, v in attn_kwargs.items()
                         if k not in ('linear_attn_feature', 'linear_attn_head_dim',
                                      'delta_alpha_init', 'delta_beta_init')}
            attn = SlidingWindowCausalSelfAttention(dim, num_heads, max_seq_length=max_seq_length,
                                                     shared_qkv=shared_qkv, shared_proj=shared_proj,
                                                     **attn_only)
            linear_attn = LinearAttention(dim, num_heads, max_seq_length=max_seq_length,
                                          qk_norm=attn_kwargs.get('qk_norm', True),
                                          attn_temp=attn_kwargs.get('attn_temp', True),
                                          feature=attn_kwargs.get('linear_attn_feature', 'relu'),
                                          head_dim=attn_kwargs.get('linear_attn_head_dim', None),
                                          rope_learnable=attn_kwargs.get('rope_learnable', False),
                                          rope_dim_fraction=attn_kwargs.get('rope_dim_fraction', 1.0),
                                          shared_qkv=shared_qkv, shared_proj=shared_proj)
            mixer_gate = nn.Parameter(torch.ones(1))
            return attn, linear_attn, mixer_gate
        # 默认：标准滑动窗口因果注意力
        attn_only = {k: v for k, v in attn_kwargs.items()
                     if k not in ('linear_attn_feature', 'linear_attn_head_dim',
                                  'delta_alpha_init', 'delta_beta_init')}
        attn = SlidingWindowCausalSelfAttention(dim, num_heads, max_seq_length=max_seq_length,
                                                 shared_qkv=shared_qkv, shared_proj=shared_proj,
                                                 **attn_only)
        return attn, None, None

    def _run_attn_mixer(self, xn: torch.Tensor, attn_past_kv, use_cache: bool, start_pos: int,
                         mem_kv, ckpt: bool) -> Tuple[torch.Tensor, Any]:
        """统一运行 attn 系 token mixer（A 项归一点），返回 (h, present)。

        attn 块与 hybrid 块内部的 attn 部分共用此入口，消除各自重复的
        project_and_norm/attend/linear_attn 混合逻辑。present 在 mixer='attn_linear'
        时已是 (k, v, linear_S, z) 四元组，供块级 (attn_kv, ssm_state, ssm_conv_state) 包裹。
        """
        if ckpt and hasattr(self.attn, 'attend'):
            # SlidingWindowCausalSelfAttention：拆分 project_and_norm / attend 以缩小检查点重算区
            q, k, v = self.attn.project_and_norm(xn, start_pos)
            h, present = checkpoint(self.attn.attend, q, k, v, attn_past_kv, use_cache, start_pos, mem_kv, use_reentrant=False)
            # 第十五轮：output_gate 在 forward 中应用，但 ckpt 路径绕过 forward 直接调 attend，
            # 须在此补应用，否则训练时 output_gate 参数无梯度（forward 未被调用）。
            if getattr(self.attn, 'output_gate_enabled', False):
                h = h * torch.sigmoid(self.attn.output_gate(h))
        else:
            # LinearAttention 等无 attend 接口的 mixer：直接对整层前向做检查点
            if ckpt:
                h, present = checkpoint(self.attn, xn, attn_past_kv, use_cache, start_pos, mem_kv, use_reentrant=False)
            else:
                h, present = self.attn(xn, attn_past_kv, use_cache, start_pos, memory_kv=mem_kv)
        if self.linear_attn is not None:
            # 阶段7：混合 mixer（attn + 线性注意力并行）
            lh, lpresent = self.linear_attn(xn, attn_past_kv, use_cache, start_pos, memory_kv=mem_kv)
            if self.linear_correction_enabled:
                # 第十一轮：线性注意力修正模式——主注意力 h 为基础，线性注意力 lh 提供
                # "修正项"（lh - h 是线性注意力捕捉到、主注意力遗漏的部分）。correction_gate
                # init 0（sigmoid≈0.5 但 gate=0 时 h 不变，向后兼容）；训练中自决修正强度。
                # 相比原凸组合（mg·h+(1-mg)·lh），修正模式保留主注意力的主体地位，线性注意力
                # 仅补充差异，避免线性注意力噪声主导。
                h = apply_correction(self.correction_gate, h, lh)
            else:
                # 原凸组合：mixer_gate 自选择 attn/linear 比例
                h = convex_combine_scalar(self.mixer_gate, h, lh)
            if use_cache and lpresent is not None:
                # 两路 KV 缓存合并：把线性注意力状态 S 和分母累积 z 塞进 attn_kv 元组
                # (k, v, linear_S, z_final)。block forward 会自行包装为
                # (attn_kv, ssm_state, ssm_conv_state) 三元组，此处只返回 attn_kv 部分。
                present = (present[0], present[1], lpresent[2], lpresent[3] if len(lpresent) > 3 else None)
        return h, present

    def _run_ssm(self, xn: torch.Tensor, ssm_past_state, ssm_past_conv_state,
                 use_cache: bool, ckpt: bool):
        """统一运行 SSM（ckpt/non-ckpt 分支归一点），返回 (h, state, conv_state)。

        消除 ssm 块 / hybrid+ssm_as_memory / hybrid 并行三处重复的 if-ckpt 分支。
        """
        if ckpt:
            return checkpoint(self.ssm, xn, ssm_past_state, ssm_past_conv_state,
                              use_cache, use_reentrant=False)
        return self.ssm(xn, past_state=ssm_past_state,
                        past_conv_state=ssm_past_conv_state, use_cache=use_cache)

    def forward(self, x: torch.Tensor, past_kv: Optional[Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]] = None, use_cache: bool = False, start_pos: int = 0, ssm_past_state: Optional[torch.Tensor] = None, ssm_past_conv_state: Optional[torch.Tensor] = None,
                memory: Optional['MemoryBank'] = None) -> Tuple[torch.Tensor, Optional[Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]]]:
        present: Optional[Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]] = None
        ssm_present_state: Optional[torch.Tensor] = None
        ssm_present_conv_state: Optional[torch.Tensor] = None
        # Extract attention KV from past_kv tuple (attn_kv, ssm_state, ssm_conv_state)
        attn_past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if past_kv is not None:
            attn_past_kv = past_kv[0]
        mem_kv = memory.get_kv() if memory is not None else None
        ckpt = self.training and self.gradient_checkpointing
        # 第十三轮：动态残差门控 highway_gate 优先于静态 residual_gate
        # highway_gate: gate = sigmoid(W·x + b)，逐 token 动态（init b=3 → sigmoid≈0.95）
        # residual_gate: 静态标量 gate（init 1.0），整个层共享
        # SEL 交替训练：_rt["highway_gate"]=False 时回退到静态 residual_gate 行为
        if self.highway_gate_enabled and self._rt.get("highway_gate", True):
            # 第一子层动态门控（hybrid 块用 hybrid_attn_gate/hybrid_ssm_gate，无 sub1_highway）
            if self.block_type != 'hybrid' and hasattr(self, 'sub1_highway'):
                gate1 = torch.sigmoid(self.sub1_highway(x))  # (B,T,1)
            else:
                gate1 = None
            gate2 = torch.sigmoid(self.ffn_highway(x))  # (B,T,1)
        else:
            gate1 = (getattr(self, 'sub1_gate', None) if (self.residual_gate_enabled and self._rt["residual_gate"]) else None)
            gate2 = (getattr(self, 'ffn_gate', None) if (self.residual_gate_enabled and self._rt["residual_gate"]) else None)
        # 阶段6：跳过层门控（skip_gate 经 sigmoid 映射到 (0,1)；_skip_active=False 时跳过失效恒为 1）
        sk = torch.sigmoid(self.skip_gate) if (self.skip_enabled and getattr(self, '_skip_active', True)) else None

        if self.block_type == 'attn':
            # 阶段7 token mixer（attn / linear / attn_linear），统一经 _run_attn_mixer 运行，
            # 两者共享同一 ln1(x) 避免重复 RMSNorm。
            xn = self.ln1(x)
            h, present = self._run_attn_mixer(xn, attn_past_kv, use_cache, start_pos, mem_kv, ckpt)
            h_eff = apply_direct(sk, h)
            x = x + self.drop(apply_direct(gate1, h_eff))
        elif self.block_type == 'ssm':
            h, ssm_present_state, ssm_present_conv_state = self._run_ssm(
                self.ln1(x), ssm_past_state, ssm_past_conv_state, use_cache, ckpt)
            h_eff = apply_direct(sk, h)
            x = x + self.drop(apply_direct(gate1, h_eff))
        elif self.block_type == 'hybrid':
            xn = self.ln1(x)
            if self.ssm_as_memory_enabled:
                # 第十一轮：SSM 作隐式记忆——先算 SSM，把 ssm_h mean-pool 投影为
                # 单个记忆槽注入 mem_kv，让注意力能"查到"SSM 的序列理解。
                ssm_h, ssm_present_state, ssm_present_conv_state = self._run_ssm(
                    xn, ssm_past_state, ssm_past_conv_state, use_cache, ckpt)
                # ssm_h: (B,T,D) → mean-pool (B,1,D) → 投影 (B,1,head_dim)
                ssm_pool = ssm_h.mean(dim=1, keepdim=True)
                ssm_k = self.ssm_k_proj(ssm_pool)  # (B,1,head_dim)
                ssm_v = self.ssm_v_proj(ssm_pool)
                # 合并到 mem_kv（无 MemoryBank 时直接创建）
                if mem_kv is not None:
                    mk, mv, meta = mem_kv
                    mem_kv = (torch.cat([mk, ssm_k], dim=1), torch.cat([mv, ssm_v], dim=1), meta)
                else:
                    mem_kv = (ssm_k, ssm_v, None)
                h, attn_present = self._run_attn_mixer(xn, attn_past_kv, use_cache, start_pos, mem_kv, ckpt)
            else:
                # 原始并行：attn 与 ssm 同时算（不互相依赖）
                h, attn_present = self._run_attn_mixer(xn, attn_past_kv, use_cache, start_pos, mem_kv, ckpt)
                ssm_h, ssm_present_state, ssm_present_conv_state = self._run_ssm(
                    xn, ssm_past_state, ssm_past_conv_state, use_cache, ckpt)
            h_eff = apply_direct(sk, h)
            ssm_eff = apply_direct(sk, ssm_h)
            if self.hybrid_single_gate and self._rt.get("hybrid_gate", True):
                # 阶段8.4：单动态门控 —— g_t 逐位置混合 attn/ssm 两路（凸组合）。
                # g_t=sigmoid(W_g·ln1(x)) ∈(0,1)，out = g_t·attn_h + (1-g_t)·ssm_h。
                mixed = convex_combine_linear(self.hybrid_mix, xn, h_eff, ssm_eff)
                x = x + self.drop(mixed)
            elif self.hybrid_gate_enabled and self._rt["hybrid_gate"]:
                # ⭐A 混合块：attn 与 ssm 两路各自可学习门控，让模型自决每层偏重
                x = x + self.drop(apply_direct(self.hybrid_attn_gate, h_eff)) \
                      + self.drop(apply_direct(self.hybrid_ssm_gate, ssm_eff))
            else:
                x = x + self.drop(h_eff) + self.drop(ssm_eff)
            if use_cache:
                present = (attn_present, ssm_present_state, ssm_present_conv_state)
        # 块输出后写入可学习压缩记忆（记忆存压缩表示，由 LM loss 监督）
        if memory is not None:
            memory.write(x)
        # FFN 子层：重算力部分（SwiGLU）放入检查点，轻量 ln2 与门控在区外
        if ckpt:
            f = checkpoint(self.ffn, self.ln2(x), use_reentrant=False)
        else:
            f = self.ffn(self.ln2(x))
        x = x + self.drop(apply_direct(gate2, f))
        # Combine attn KV cache and SSM state
        if use_cache:
            if self.block_type == 'attn':
                # present 来自 _run_attn_mixer：(k,v) 二元组或 (k,v,linear_S,z) 四元组
                # 统一包装为块级 (attn_kv, ssm_state, ssm_conv_state) 三元组
                present = (present, None, None)
            elif self.block_type == 'ssm':
                present = (None, ssm_present_state, ssm_present_conv_state)
            elif self.block_type == 'hybrid':
                present = (attn_present, ssm_present_state, ssm_present_conv_state)
        return x, present

    def set_enhancements_active(self, spec):
        """运行时开关（按开关粒度）：`spec=True/False` 全开/全关；`spec=dict` 按键更新。
        用于“交替/分段增强”训练，关闭时跳过对应残差门控/混合门控（恒等）。"""
        if isinstance(spec, bool):
            on = spec
            self._rt = {"residual_gate": on, "hybrid_gate": on,
                        "highway_gate": on, "layer_film": on}
        elif isinstance(spec, dict):
            for k, v in spec.items():
                if k in self._rt:
                    self._rt[k] = bool(v)
        else:
            raise TypeError(f"set_enhancements_active 期望 bool 或 dict，收到 {type(spec)}")
        if hasattr(self, 'attn'):
            self.attn.set_enhancements_active(spec)

    def set_skip_active(self, active: bool = True):
        """运行时开关跳过层门控（推理剪枝时关闭则所有层恒保留）。"""
        self._skip_active = bool(active)


def _parse_layer_plan(layer_plan: Optional[List[str] | str], num_layers: int) -> List[str]:
    """layer_plan: None / 'attn' / 'attn,ssm,attn,ssm' / list。
     返回长度为 num_layers 的 block 类型列表。"""
    if layer_plan is None:
        return ['attn'] * num_layers
    if isinstance(layer_plan, str):
        if ',' not in layer_plan:
            return [layer_plan.strip()] * num_layers
        parts = [p.strip() for p in layer_plan.split(',') if p.strip()]
    else:
        parts = list(layer_plan)
    if len(parts) != num_layers:
        raise ValueError(f"layer_plan 长度 {len(parts)} 与 num_layers {num_layers} 不一致")
    valid = {'attn', 'ssm', 'hybrid'}
    for p in parts:
        if p not in valid:
            raise ValueError(f"未知 block 类型: {p}（可选 {valid}）")
    return parts


class CrossLayerRouter(nn.Module):
    """跨层稀疏路由信息流动（DenseNet 风格 top-k 跳跃连接 + 输入相关门控）。

    每层 j 拥有路由器 W_j: Linear(D, 1)，对前 j 层的输出 h_i 打分（基于 h_i 的
    token-mean），选 top-k 个最高分前层，按 sigmoid(score) 加权累加注入当前层输入 x。

    特性（用户要求）：
      - 稀疏：top-k 选择（k < num_layers），仅 k 个前层参与（非全连接 DenseNet）
      - 残差：routed 加到 x 上（x + routed/k），保留原信息流，不替换
      - 选择性：sigmoid 门控，模型自决每层多借前层信息（gate∈(0,1)）
      - 输入相关：路由分数依赖前层输出 h_i（不同输入不同路由路径）

    参数量：每层 D+1 个（router weight + bias），共 O(L*D)。
    init: weight=0, bias=-3 → sigmoid(-3)≈0.05，弱初始注入（不破坏预训练行为），
    训练中 router 自决是否增大门控（拉高 bias 或调 weight）。
    """

    def __init__(self, dim: int, num_layers: int, topk: int = 2):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.topk = max(1, min(topk, num_layers - 1))
        # 每层一个路由器；第 0 层无前层，用 Identity 占位保持索引对齐
        self.routers = nn.ModuleList([
            nn.Linear(dim, 1, bias=True) if i > 0 else nn.Identity()
            for i in range(num_layers)
        ])
        # init: weight=0 + bias=-3 → score=-3 → sigmoid≈0.05（弱初始注入）
        for r in self.routers:
            if isinstance(r, nn.Linear):
                nn.init.zeros_(r.weight)
                nn.init.constant_(r.bias, -3.0)

    def route(self, layer_idx: int, x: torch.Tensor,
              prev_outputs: List[torch.Tensor]) -> torch.Tensor:
        """对当前层输入 x 注入跨层路由残差（稀疏 top-k + sigmoid 门控 + 残差加法）。

        Args:
            layer_idx: 当前层索引（0-indexed；0 层无前层直接返回 x）
            x: (B, T, D) 当前层输入
            prev_outputs: 之前所有层的输出列表（长度 = layer_idx）

        Returns:
            (B, T, D) 注入路由残差后的新 x（x + mean(gates * selected) / k）
        """
        if layer_idx == 0 or not prev_outputs:
            return x
        router = self.routers[layer_idx]
        # 批量堆叠前层输出：(B, num_prev, T, D)
        prev_stack = torch.stack(prev_outputs, dim=1)
        B, num_prev, T, D = prev_stack.shape
        # token-mean 表示每层输出特征：(B, num_prev, D)
        prev_mean = prev_stack.mean(dim=2)
        # 路由打分：(B, num_prev, D) -> (B, num_prev, 1) -> (B, num_prev)
        scores = router(prev_mean).squeeze(-1)
        # top-k 稀疏选择——用 one-hot 掩码替代 gather/scatter（DML 兼容）
        # 用 torch.eye 高级索引构建精确 k 个 1 的 one-hot（只读 op，DML 原生支持）。
        # 历史 Bug（已修）：曾用 `mask = (scores >= threshold).float()`，但 init 时
        # 所有 router weight=0/bias=-3 导致 scores 全并列（=-3），>= 阈值会选中所有
        # num_prev 项而非 k 项，造成 num_prev/k 倍过注入（layer 11,topk=2 时 5.5x），
        # 直接破坏"弱注入"设计意图。torch.topk 按索引序打破并列，精确选 k 个。
        k = min(self.topk, num_prev)
        topk_vals, topk_idx = torch.topk(scores, k, dim=-1)  # (B, k)
        eye = torch.eye(num_prev, device=scores.device, dtype=scores.dtype)
        mask = eye[topk_idx].sum(dim=1)  # (B, num_prev) 精确 k 个 1
        # sigmoid 选择性门控（对所有前层算，mask 清零非 top-k）
        gates = torch.sigmoid(scores) * mask  # (B, num_prev)
        # 加权求和：einsum 避免 gather/scatter，DML 原生支持
        routed = torch.einsum('bn,bntd->btd', gates, prev_stack)  # (B, T, D)
        routed = routed / k  # 归一化（保持 magnitude 不随 k 爆炸）
        return x + routed


class TransformerModel(nn.Module):
    """现代 decoder-only 语言模型（Pre-LN + RMSNorm + RoPE + SwiGLU + 权重共享）。

     支持混合架构：通过 layer_plan 指定每层为 attn / ssm / hybrid。
     默认 layer_plan=None 时全为 attn，与旧权重完全兼容。
    """

    # INT-3：增强开关键名单一事实来源（attn 层 qk_norm/attn_temp + block 层
    # residual_gate/hybrid_gate + 第十三轮 layer_film/highway_gate 的并集中所有
    # 可分段 SEL 交替的开关），供 train.py 的 enhancement_schedule 与测试派生，
    # 避免键名清单散落多处漂移。
    ENHANCEMENT_KEYS = ("qk_norm", "attn_temp", "residual_gate", "hybrid_gate",
                        "layer_film", "highway_gate", "input_highway")

    def __init__(self, vocab_size: int, embedding_dim: int, num_heads: int, num_layers: int,
                 hidden_dim: int, max_seq_length: int, dropout: float = 0.0, tie_weights: bool = True,
                 gradient_checkpointing: bool = True,
                 layer_plan: Optional[List[str] | str] = None,
                 ssm_d_state: int = 16, ssm_d_inner_factor: int = 1, ssm_dt_rank: Optional[int] = None,
                 ssm_conv_kernel: int = 3, ssm_dt_proj_bias_init: float = 0.1,
                 ssm_a_log_init_range: List[float] = [-1, 1],
                 ssm_D_init: float = 1.0,
                  attn_window: int = 0, attn_rel_bias: bool = False,
                  rope_base: float = ROPE_BASE, rope_max_len: int = 4096,
                  # 注意两个"长度"语义不同、勿混（额外7 澄清）：
                  #   max_seq_length —— 本模型训练/生成的上下文窗口上限（用于生成截断、复杂度归一）；
                  #   rope_max_len     —— RoPE/注意力缓冲区的位置编码容量，向下传给各 block/attn 作其
                  #                       max_seq_length。默认与前者一致；显式配置时才单独覆盖。
                  #   两者经 config_loader 默认对齐（rope_max_len 缺省回退到 max_seq_length）。
                 mask_fill_value: float = MASK_FILL_VALUE,
                  qk_norm: bool = True, attn_temp: bool = True,
                    residual_gate: bool = True, hybrid_gate: bool = True,
                    hybrid_single_gate: bool = False,
                    char_merge: bool = False, char_merge_kernel: int = 3,
                    char_merge_dropout: float = 0.0,
                    memory_size: int = 0, memory_comp_dim: int = 32,
                    memory_retrieval: bool = False, memory_sparse_topk: int = 0,
                    memory_forget: bool = False, memory_product_key: bool = False,
                    memory_retrieval_full: bool = False, memory_retrieval_topk: int = 32,
                    rope_learnable: bool = False, alibi: bool = False,
                   layer_skip: bool = False, learn_window: bool = False, window_base: int = 64,
                   mixer: str = 'attn',
                   linear_attn_feature: str = 'relu',
                   linear_attn_head_dim: Optional[int] = None,
                   pe_gate: bool = False,
                   rope_dim_fraction: float = 1.0,
                   output_gate: bool = False,
                   zero_centered_norm: bool = False,
                   delta_alpha_init: float = -2.0,
                   delta_beta_init: float = 2.0,
                   ngram_fusion: bool = False, ngram_model=None,
                   ngram_gate_scale: float = 1.0, igmcg: bool = False,
                   share_attn_proj: bool = False,
                   share_ffn: bool = False,
                   share_norm: bool = False,
                   ssm_type: str = 'standard',
                   linear_correction: bool = False,
                   cross_layer_routing: bool = False,
                   cross_layer_topk: int = 2,
                   ssm_as_memory: bool = False,
                   cross_ssm_transfer: bool = False,
                   progressive_residual: bool = False,
                   layer_film: bool = False,
                   highway_gate: bool = False,
                   input_highway: bool = False,
                   layer_contrastive: bool = False,
                   shared_alibi: bool = False):
        super(TransformerModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.attn_window = attn_window
        self.gradient_checkpointing = gradient_checkpointing
        # 阶段8.1：n-gram 神经融合——把统计 n-gram 先验经可学习门控 g_t=sigmoid(h_t·W_g)
        # 逐位置加回 logits（z_neural + g_t·ngram_vec.detach()）。主干 z_neural 仍吃完整
        # CE 梯度（gate 只缩放外部统计向量、不缩放主干），故不会塌缩、主干始终是独立 LM。
        # 模型于每步自决多信 n-gram：自身不确定时 g_t↑（靠统计兜底），有把握时 g_t↓。
        # 默认关（向后兼容、不增参数、不构建统计表）；开启时由调用方传入已构建的 ngram_model。
        self.ngram_fusion_enabled = bool(ngram_fusion) and (ngram_model is not None)
        self.ngram_model = ngram_model if self.ngram_fusion_enabled else None
        self.ngram_gate_scale = ngram_gate_scale
        # 阶段8.7 IGMCG 2.0：IGMCG（直觉引导）与 n-gram 融合训练，且由模型自决：
        #  - 是否使用 IGMCG（igmcg_use_gate 逐位置 sigmoid，可归零 → 模型自选"用不用"）；
        #  - 各阶 n-gram 占比（ngram_order_logits 可学 softmax → 模型自选"用几个/哪种 n"）。
        # 二者均仅在 ngram_fusion 开启时构建（IGMCG 依赖 n-gram 统计缓冲）。
        # 默认关、向后兼容、旧权重无这些参数（strict=False 安全）。
        self.igmcg_enabled = bool(igmcg) and self.ngram_fusion_enabled
        if self.ngram_fusion_enabled:
            self.ngram_gate = nn.Linear(embedding_dim, 1)
            # 可学 n-gram 阶混合权重（替代固定 l1/l2/l3）：softmax 后逐阶加权混合 logprob。
            _K = getattr(self.ngram_model, 'max_order', 3) if self.ngram_model is not None else 3
            self.ngram_order_logits = nn.Parameter(torch.zeros(_K))
            if self.igmcg_enabled:
                # 逐位置"是否启用 IGMCG 引导"门控 + 直觉条件投影（7 维直觉→标量偏置，按序列）。
                self.igmcg_use_gate = nn.Linear(embedding_dim, 1)
                self.intuition_proj = nn.Linear(7, 1)
        # 增量解码 n-gram 上下文滚动缓冲（末 ctx_len token），由 forward 维护；全量/训练时为 None。
        self._ngram_last_ids = None
        # 阶段8.2：推理期静态剪枝标记（prune_layers 填充；默认空=不剪）
        self._pruned_layers = set()
        self.layer_plan = _parse_layer_plan(layer_plan, num_layers)
        self.rope_base = rope_base
        self.rope_max_len = rope_max_len
        # 阶段2：可学习压缩记忆（memory_size>0 时启用），存压缩表示 + 可学门控选槽
        self.memory_enabled = memory_size > 0
        self.memory_size = memory_size
        self.memory_comp_dim = memory_comp_dim
        self.memory_retrieval = memory_retrieval
        self.memory_sparse_topk = memory_sparse_topk
        self.memory_forget = memory_forget

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.drop = nn.Dropout(dropout)
        # 轻量学习型分词层：字符级输入时启用，把相邻字符融合为词表示
        self.char_merge_enabled = char_merge
        if char_merge:
            self.char_merge = CharMergeLayer(
                embedding_dim, kernel_size=char_merge_kernel,
                dropout=char_merge_dropout)
        ssm_kwargs = dict(
            d_state=ssm_d_state,
            d_inner_factor=ssm_d_inner_factor,
            dt_rank=ssm_dt_rank,
            conv_kernel=ssm_conv_kernel,
            dt_proj_bias_init=ssm_dt_proj_bias_init,
            a_log_init_range=ssm_a_log_init_range,
            D_init=ssm_D_init,
            ssm_type=ssm_type,
        )
        attn_kwargs = dict(window=attn_window, rel_bias=attn_rel_bias,
                           qk_norm=qk_norm, attn_temp=attn_temp,
                           rope_learnable=rope_learnable, alibi=alibi,
                           retrieval_full=memory_retrieval_full,
                           retrieval_topk=memory_retrieval_topk,
                           learn_window=learn_window, window_base=window_base,
                           linear_attn_feature=linear_attn_feature,
                           linear_attn_head_dim=linear_attn_head_dim,
                           pe_gate=pe_gate,
                           rope_dim_fraction=rope_dim_fraction,
                           output_gate=output_gate,
                           delta_alpha_init=delta_alpha_init,
                           delta_beta_init=delta_beta_init)
        # 第十五轮：保存 GatedDeltaNet 专用初始化参数，供 _apply_specialized_inits 重置
        # （_init_weights 通用 N(0,0.02) 会覆盖 alpha_proj/beta_proj 的 zero weight + 专用 bias）
        self._delta_alpha_init = float(delta_alpha_init)
        self._delta_beta_init = float(delta_beta_init)
        # 层间共享 attention projection（share_attn_proj=True）：
        # 各层复用同一组 QKV + Output 投影参数，减少 ~40% 参数量并起正则化作用。
        # 仅影响 attn/linear 混合器（SSM/FFN/Memory 参数保持独立）。
        self.share_attn_proj = share_attn_proj
        _shared_qkv = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False) if share_attn_proj else None
        _shared_proj = nn.Linear(embedding_dim, embedding_dim, bias=False) if share_attn_proj else None
        if share_attn_proj:
            self.shared_qkv = _shared_qkv
            self.shared_proj = _shared_proj
        # 层间共享 FFN（share_ffn=True）：各层复用同一组 SwiGLU 参数
        _shared_ffn = SwiGLU(embedding_dim, hidden_dim) if share_ffn else None
        if share_ffn:
            self.shared_ffn = _shared_ffn
        # 层间共享 LayerNorm（share_norm=True）：各层复用同一组 RMSNorm 参数
        _shared_lns = (RMSNorm(embedding_dim, zero_centered=zero_centered_norm),
                       RMSNorm(embedding_dim, zero_centered=zero_centered_norm)) if share_norm else None
        if share_norm:
            self.shared_lns = nn.ModuleList(_shared_lns)
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, hidden_dim, block_type=bt,
                             dropout=dropout, max_seq_length=rope_max_len,
                             ssm_kwargs=ssm_kwargs, attn_kwargs=attn_kwargs,
                             gate_cfg=GateConfig(
                                 residual_gate=residual_gate,
                                 hybrid_gate=hybrid_gate,
                                 hybrid_single_gate=hybrid_single_gate,
                                 skip=layer_skip,
                                 linear_correction=linear_correction,
                                 highway_gate=highway_gate,
                             ),
                             gradient_checkpointing=gradient_checkpointing,
                             mixer=mixer,
                             shared_qkv=_shared_qkv, shared_proj=_shared_proj,
                             shared_ffn=_shared_ffn, shared_lns=_shared_lns,
                             ssm_as_memory=ssm_as_memory,
                             zero_centered_norm=zero_centered_norm)
            for bt in self.layer_plan
        ])
        # 第十四轮：ALiBi 跨层共享——所有注意力层共用第一层的 alibi_slopes buffer
        # 减参（num_layers→1 组斜率）+ 确保跨层位置建模一致
        if shared_alibi and alibi:
            _shared_slopes = None
            for blk in self.blocks:
                if hasattr(blk, 'attn') and hasattr(blk.attn, 'alibi_slopes'):
                    if _shared_slopes is None:
                        _shared_slopes = blk.attn.alibi_slopes
                    else:
                        blk.attn.alibi_slopes = _shared_slopes
        self.ln_f = RMSNorm(embedding_dim, zero_centered=zero_centered_norm)
        self.output_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self._tie_weights = tie_weights
        if tie_weights:
            self.output_head.weight = self.embedding.weight
        # 可学习压缩记忆：固定槽，压缩矩阵 + 写入门控均参与 LM loss 监督
        if self.memory_enabled:
            self.memory_bank = MemoryBank(
                embedding_dim, num_slots=memory_size, comp_dim=memory_comp_dim,
                head_dim=embedding_dim // num_heads, dropout=dropout,
                retrieval=memory_retrieval, sparse_topk=memory_sparse_topk,
                forget=memory_forget, product_key=memory_product_key)
        # 第十一轮：跨层稀疏路由信息流动（DenseNet 风格 top-k 跳跃连接）
        self.cross_layer_routing = cross_layer_routing
        self.cross_layer_topk = cross_layer_topk
        if cross_layer_routing and num_layers > 1:
            self.cross_router = CrossLayerRouter(embedding_dim, num_layers,
                                                 topk=min(cross_layer_topk, num_layers - 1))
        # 第十二轮：层间 SSM 状态传递——hybrid 块间传递 SSM 信息
        # 前一个 hybrid 块的输出经 cross_ssm_proj 投影后加到下一个 hybrid 块的输入，
        # 让深层 SSM 能接收到浅层 SSM 的序列理解（跨层 SSM 信息流）。
        # 与 ssm_as_memory（SSM→Attention）互补：ssm_as_memory 是块内 SSM→Attn，
        # cross_ssm_transfer 是块间 SSM→SSM（经投影残差注入）。
        self.cross_ssm_transfer = cross_ssm_transfer
        if cross_ssm_transfer and any(bt == 'hybrid' for bt in self.layer_plan):
            self.cross_ssm_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        # 第十二轮：渐进式残差——浅层残差大（保留信息），深层残差小（激进变换）
        # 通过缩放残差门控的 init 值实现：layer 0 不变（1.0），后续层按 1/sqrt(depth) 衰减
        self.progressive_residual = progressive_residual
        # 第十三轮：跨层 FiLM 调制——浅层输出经线性投影产生 (γ, β)，调制深层 input：
        #   x_out = x * (1 + γ) + β（init γ=0, β=0 → 恒等，向后兼容）
        # 与 cross_layer_routing（top-k 残差注入）/ cross_ssm_transfer（SSM 传递）正交：
        #   layer_film 是仿射调制（multiplicative + additive），更细粒度的跨层信息流。
        # 每层一个 Linear(D → 2D)（layer 0 用 Identity 占位保持索引对齐）。
        self.layer_film_enabled = layer_film and num_layers > 1
        if self.layer_film_enabled:
            self.layer_film_projs = nn.ModuleList([
                nn.Linear(embedding_dim, 2 * embedding_dim) if i > 0 else nn.Identity()
                for i in range(num_layers)
            ])
        # 第十三轮：动态残差门控已透传到 TransformerBlock（highway_gate 参数）
        self.highway_gate = highway_gate
        # SEL 交替训练：layer_film 模型级开关（默认 True，由 set_enhancements_active 切换）
        self._rt_layer_film = self.layer_film_enabled
        # 第十四轮：输入全局高速公路——embedding 输出 x0 经门控注入每层
        # input_highway_proj: Linear(D, D) 投影 x0；input_highway_gates[i]: Linear(D, 1) 逐层门控
        # init: proj weight=0（弱注入），gate bias=-3（sigmoid≈0.05，开始几乎不影响）
        self.input_highway_enabled = input_highway and num_layers > 1
        self._rt_input_highway = self.input_highway_enabled
        if self.input_highway_enabled:
            self.input_highway_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.input_highway_gates = nn.ModuleList([
                nn.Linear(embedding_dim, 1) if i > 0 else nn.Identity()
                for i in range(num_layers)
            ])
        # 第十四轮：层间对比绑定——训练期计算相邻层余弦相似度损失，防深层遗忘浅层特征
        # 损失存入 self._contrastive_loss（训练循环可加到主 loss）；eval 时不计算
        self.layer_contrastive_enabled = layer_contrastive and num_layers > 1
        self._contrastive_loss = None
        # 第十四轮：ALiBi 跨层共享——所有注意力层共用同一组 alibi_slopes（减参+一致位置建模）
        # 在 block 创建后统一绑定（见下方 blocks 构建完毕后的 shared_alibi 处理）
        self.shared_alibi_enabled = shared_alibi and alibi
        # 权重初始化（_init_weights 遍历所有 Linear 用 N(0,0.02)，再对 SSM 调 proper_init 覆盖）
        self._init_weights()
        # 专用初始化必须在 _init_weights 之后重新应用（否则被通用 N(0,0.02)/zeros 覆盖）：
        #   - cross_ssm_proj.weight=0（弱注入，开始时不影响模型）
        #   - cross_router.routers: weight=0, bias=-3（sigmoid≈0.05 弱注入）
        #   - progressive_residual: 残差门控按 1/sqrt(depth) 衰减
        #   - layer_film_projs: weight=0, bias=0 → γ=β=0 → 恒等
        #   - highway_gate (sub1_highway/ffn_highway): weight=0, bias=3.0 → sigmoid≈0.95
        # 与 SSM proper_init 同理（也在 _init_weights 后调用）。
        self._apply_specialized_inits()

    @classmethod
    def from_config(cls, cfg: 'ModelConfig', ngram_model=None) -> TransformerModel:
        """从 ModelConfig dataclass 构建（替代 config_loader 中 42 个 mc.get()）。"""
        return cls(
            vocab_size=cfg.vocab_size,
            embedding_dim=cfg.embedding_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            hidden_dim=cfg.hidden_dim,
            max_seq_length=cfg.max_seq_length,
            dropout=cfg.dropout,
            tie_weights=cfg.tie_weights,
            gradient_checkpointing=cfg.gradient_checkpointing,
            layer_plan=cfg.layer_plan,
            ssm_d_state=cfg.ssm.d_state,
            ssm_d_inner_factor=cfg.ssm.d_inner_factor,
            ssm_dt_rank=cfg.ssm.dt_rank,
            ssm_conv_kernel=cfg.ssm.conv_kernel,
            ssm_dt_proj_bias_init=cfg.ssm.dt_proj_bias_init,
            ssm_a_log_init_range=cfg.ssm.a_log_init_range,
            ssm_D_init=cfg.ssm.D_init,
            attn_window=cfg.attn.window,
            attn_rel_bias=cfg.attn.rel_bias,
            rope_base=cfg.rope_base,
            rope_max_len=cfg.rope_max_len,
            mask_fill_value=cfg.mask_fill_value,
            qk_norm=cfg.attn.qk_norm,
            attn_temp=cfg.attn.attn_temp,
            residual_gate=cfg.residual_gate,
            hybrid_gate=cfg.hybrid_gate,
            hybrid_single_gate=cfg.hybrid_single_gate,
            char_merge=cfg.char_merge,
            char_merge_kernel=cfg.char_merge_kernel,
            char_merge_dropout=cfg.char_merge_dropout,
            memory_size=cfg.memory.size,
            memory_comp_dim=cfg.memory.comp_dim,
            memory_retrieval=cfg.memory.retrieval,
            memory_sparse_topk=cfg.memory.sparse_topk,
            memory_forget=cfg.memory.forget,
            memory_product_key=cfg.memory.product_key,
            memory_retrieval_full=cfg.memory.retrieval_full,
            memory_retrieval_topk=cfg.memory.retrieval_topk,
            rope_learnable=cfg.attn.rope_learnable,
            alibi=cfg.attn.alibi,
            layer_skip=cfg.layer_skip,
            learn_window=cfg.attn.learn_window,
            window_base=cfg.attn.window_base,
            mixer=cfg.attn.mixer,
            linear_attn_feature=cfg.attn.linear_attn_feature,
            linear_attn_head_dim=cfg.attn.linear_attn_head_dim,
            pe_gate=cfg.attn.pe_gate,
            rope_dim_fraction=cfg.attn.rope_dim_fraction,
            output_gate=cfg.attn.output_gate,
            zero_centered_norm=cfg.attn.zero_centered_norm,
            delta_alpha_init=cfg.attn.delta_alpha_init,
            delta_beta_init=cfg.attn.delta_beta_init,
            ngram_fusion=cfg.ngram_fusion,
            ngram_model=ngram_model,
            ngram_gate_scale=cfg.ngram_gate_scale,
            igmcg=cfg.igmcg,
            share_attn_proj=cfg.share_attn_proj,
            share_ffn=cfg.share_ffn,
            share_norm=cfg.share_norm,
            ssm_type=cfg.ssm.ssm_type,
            linear_correction=cfg.attn.linear_correction,
            cross_layer_routing=cfg.cross_layer_routing,
            cross_layer_topk=cfg.cross_layer_topk,
            ssm_as_memory=cfg.ssm_as_memory,
            cross_ssm_transfer=cfg.cross_ssm_transfer,
            progressive_residual=cfg.progressive_residual,
            layer_film=cfg.layer_film,
            highway_gate=cfg.highway_gate,
            input_highway=cfg.input_highway,
            layer_contrastive=cfg.layer_contrastive,
            shared_alibi=cfg.shared_alibi,
        )

    def set_enhancements_active(self, spec):
        """运行时开关（按开关粒度）：`spec=True/False` 全开/全关；`spec=dict` 按键更新。
        用于"交替/分段增强"训练（关闭则跳过对应增强，恒等）。"""
        # 模型级特性（layer_film / input_highway）开关
        if isinstance(spec, bool):
            self._rt_layer_film = spec and self.layer_film_enabled
            self._rt_input_highway = spec and self.input_highway_enabled
        elif isinstance(spec, dict):
            if 'layer_film' in spec:
                self._rt_layer_film = bool(spec['layer_film']) and self.layer_film_enabled
            if 'input_highway' in spec:
                self._rt_input_highway = bool(spec['input_highway']) and self.input_highway_enabled
        # 块级特性（residual_gate/hybrid_gate/highway_gate/qk_norm/attn_temp）开关
        for blk in self.blocks:
            blk.set_enhancements_active(spec)

    def set_gradient_checkpointing(self, enabled: bool):
        """统一开关梯度检查点（同步到各 block；torch.compile 路径应设为 False）。"""
        self.gradient_checkpointing = enabled
        for blk in self.blocks:
            blk.gradient_checkpointing = enabled

    def set_skip_active(self, active: bool = True):
        """统一开关跳过层门控（同步到各 block）。"""
        for blk in self.blocks:
            blk.set_skip_active(active)

    def set_ngram_fusion_active(self, active: bool = True):
        """运行时开关 n-gram 神经融合（训练全开、推理可按需关）。"""
        self._ngram_fusion_active = bool(active) and self.ngram_fusion_enabled

    def set_ngram_gate_scale(self, scale: float):
        """推理期总闸：用户在 (0, 1+] 间缩放门控输出（1.0=模型自决，0=拔掉 n-gram）。"""
        self.ngram_gate_scale = float(scale)

    def compute_complexity(self) -> torch.Tensor:
        """阶段6/8：计算当前模型结构的"激活复杂度"标量（用于复杂度奖励正则）。

        各激活组件的归一化成本累加：
          - 未跳过的层才计入（skip_gate≈0 则该层成本趋零）；
          - 线性注意力成本低于 softmax 注意力（约 0.3x）；
          - 滑动窗口越小成本越低（window/max_seq_length）；learn_window 时用连续软窗口
            sigmoid(log_window)*window_base 参与成本，使复杂度奖励可经梯度调节可学窗口；
          - 记忆槽越多成本越高（memory_size/max_seq_length）。
        返回值随可学参数（skip_gate / mixer_gate / log_window）变化，可导。
        """
        total = torch.tensor(0.0, device=self.embedding.weight.device)
        for blk in self.blocks:
            # 跳过层：skip_gate→(0,1)，≈0 则该层不计入
            if getattr(blk, 'skip_enabled', False):
                keep = torch.sigmoid(blk.skip_gate).sum()
            else:
                keep = torch.ones(1, device=total.device).sum()
            layer_cost = keep
            # mixer：线性注意力更省（约 0.3x）。hybrid 按 mixer_gate 比例插值；
            # 纯 linear mixer（self.attn 即 LinearAttention、linear_attn 为 None）直接 0.3x 折扣。
            if getattr(blk, 'linear_attn', None) is not None:
                mg = torch.sigmoid(blk.mixer_gate).sum()
                attn_cost = mg * 1.0 + (1.0 - mg) * 0.3
                layer_cost = layer_cost * attn_cost
            elif hasattr(blk, 'attn') and isinstance(blk.attn, LinearAttention):
                layer_cost = layer_cost * 0.3
            # 窗口成本：相对 max_seq_length。learn_window 时用连续软窗口
            # (sigmoid(log_window)*window_base) 参与成本计算，使复杂度奖励能经梯度
            # 调节可学窗口（离散 window 经 round(exp) 不可导，故走软代理）。
            eff_window = 0
            if hasattr(blk, 'attn') and getattr(blk.attn, 'learn_window', False):
                eff_window = (torch.sigmoid(blk.attn.log_window) * blk.attn.window_base)
            elif hasattr(blk, 'attn') and getattr(blk.attn, 'window', 0) > 0:
                eff_window = min(blk.attn.window, self.max_seq_length)
            elif hasattr(blk, 'attn') and isinstance(blk.attn, LinearAttention) and getattr(self, 'attn_window', 0) > 0:
                # 线性注意力同样处理窗口内 token，按模型配置窗口计成本（再乘 0.3x 折扣）
                eff_window = min(self.attn_window, self.max_seq_length)
            if isinstance(eff_window, torch.Tensor) or (isinstance(eff_window, int) and eff_window > 0):
                wcost = eff_window / max(self.max_seq_length, 1)
                layer_cost = layer_cost * wcost
            total = total + layer_cost
        # 记忆预算：记忆槽数相对序列长度
        if self.memory_enabled and self.memory_size > 0:
            total = total + torch.tensor(self.memory_size / max(self.max_seq_length, 1),
                                         device=total.device)
        return total

    def max_complexity(self) -> float:
        """阶段8.2：结构复杂度的理论上限（所有层全保留、窗口取最大、含记忆），用作
        hinge 预算约束的归一化分母。纯 Python 标量，无张量、不参与反向。"""
        full = float(len(self.blocks))
        if self.memory_enabled and self.memory_size > 0:
            full += self.memory_size / max(self.max_seq_length, 1)
        return full

    def prune_layers(self, threshold: float = 0.5):
        """阶段8.2：推理期静态剪枝——跳过 skip_gate 概率 > threshold 的层（sigmoid 直阈）。

        skip_gate 经 straight-through 训练后，推理时把"几乎必跳过"的层直接移除，
        实现真实的推理提速（不止是软正则）。置 threshold<=0 取消剪枝（全保留）。

        返回被剪掉的层索引列表。
        """
        self._prune_threshold = float(threshold)
        pruned = []
        for i, blk in enumerate(self.blocks):
            if getattr(blk, 'skip_enabled', False):
                p = float(torch.sigmoid(blk.skip_gate).item())
                if p > threshold:
                    pruned.append(i)
        self._pruned_layers = set(pruned) if threshold > 0 else set()
        return pruned

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    # CharMergeLayer 的 gate.bias 有专用初始化（-1.0 → sigmoid≈0.27，初期少融合），
                    # 跳过通用零初始化以保留该设计意图。
                    if not (hasattr(self, 'char_merge') and m is getattr(self.char_merge, 'gate', None)):
                        nn.init.zeros_(m.bias)
        nn.init.normal_(self.embedding.weight, 0, 0.02)
        # SSM 模块用更专业的初始化覆盖通用初始化
        for m in self.modules():
            if type(m) is MambaSSM:
                m.proper_init()

    def _apply_specialized_inits(self):
        """重新应用被 _init_weights 通用 N(0,0.02)/zeros 覆盖的专用初始化。

        历史 bug：cross_ssm_proj / cross_router.routers 在 __init__ 中先做专用初始化
        （zero weight / bias=-3），随后 _init_weights 遍历所有 nn.Linear 用通用分布
        覆盖，导致弱注入设计意图失效（cross_router.bias 由 -3→0，sigmoid 由 0.05→0.5）。
        此方法在 _init_weights 之后调用，恢复专用初始化。
        """
        # cross_ssm_proj.weight=0（弱注入，开始时不影响模型）
        if self.cross_ssm_transfer and hasattr(self, 'cross_ssm_proj'):
            nn.init.zeros_(self.cross_ssm_proj.weight)
        # cross_router.routers: weight=0, bias=-3（sigmoid≈0.05 弱注入）
        if hasattr(self, 'cross_router'):
            for r in self.cross_router.routers:
                if isinstance(r, nn.Linear):
                    nn.init.zeros_(r.weight)
                    nn.init.constant_(r.bias, -3.0)
        # progressive_residual: 残差门控按 1/sqrt(depth) 衰减
        # - 静态 residual_gate 模式：缩放 sub1_gate/ffn_gate（乘性门）
        # - highway_gate 模式：缩放 sub1_highway/ffn_highway 的 bias（sigmoid 前缩放，
        #   bias=3/sqrt(d) → sigmoid 衰减，等效深层残差更小）
        # 注意：highway bias 缩放必须在 highway_gate init 之后执行（否则被覆盖）
        if self.progressive_residual:
            for i, blk in enumerate(self.blocks):
                if i > 0:
                    g = 1.0 / math.sqrt(i + 1)
                    with torch.no_grad():
                        if hasattr(blk, 'sub1_gate'):
                            blk.sub1_gate.fill_(g)
                        if hasattr(blk, 'ffn_gate'):
                            blk.ffn_gate.fill_(g)
        # layer_film_projs: weight=0, bias=0 → γ=β=0 → 恒等（向后兼容）
        if self.layer_film_enabled:
            for proj in self.layer_film_projs:
                if isinstance(proj, nn.Linear):
                    nn.init.zeros_(proj.weight)
                    nn.init.zeros_(proj.bias)
        # highway_gate: sub1_highway/ffn_highway weight=0, bias=3.0
        # → sigmoid(3)≈0.95 → x + 0.95·f(x) ≈ 原 residual_gate=1.0 行为
        # progressive_residual + highway_gate 组合：bias=3.0/sqrt(depth)（深层衰减）
        for i, blk in enumerate(self.blocks):
            if getattr(blk, 'highway_gate_enabled', False):
                # layer 0 不衰减（bias=3.0），layer i>0 按 1/sqrt(depth) 衰减
                bias_val = 3.0 if i == 0 or not self.progressive_residual else 3.0 / math.sqrt(i + 1)
                if hasattr(blk, 'sub1_highway'):
                    nn.init.zeros_(blk.sub1_highway.weight)
                    nn.init.constant_(blk.sub1_highway.bias, bias_val)
                if hasattr(blk, 'ffn_highway'):
                    nn.init.zeros_(blk.ffn_highway.weight)
                    nn.init.constant_(blk.ffn_highway.bias, bias_val)
        # 第十四轮：input_highway 专用初始化
        # proj weight=0（弱注入，开始时 proj(x0)=0 不影响模型）
        # gate bias=-3（sigmoid≈0.05，逐层逐渐学习该注入多少原始信号）
        if self.input_highway_enabled:
            nn.init.zeros_(self.input_highway_proj.weight)
            for gate in self.input_highway_gates:
                if isinstance(gate, nn.Linear):
                    nn.init.zeros_(gate.weight)
                    nn.init.constant_(gate.bias, -3.0)
        # 第十五轮：GatedDeltaNet 专用初始化重置
        # alpha_proj/beta_proj 在 __init__ 中已做 zero weight + 专用 bias（alpha_init/beta_init），
        # 但 _init_weights 遍历 nn.Linear 时用 N(0,0.02) 覆盖了 weight，用 zeros 覆盖了 bias。
        # 此处按 GatedDeltaNet 设计意图恢复：weight=0（弱门控，逐 token 由模型自决），
        # bias=alpha_init/beta_init（alpha sigmoid≈0.12 弱遗忘起步 / beta sigmoid≈0.88 强写入起步）。
        # 同时重置 SlidingWindowCausalSelfAttention.output_gate（weight=0, bias=0 → sigmoid=0.5 半通起步）。
        for blk in self.blocks:
            attn = getattr(blk, 'attn', None)
            if attn is None:
                continue
            if hasattr(attn, 'alpha_proj'):
                nn.init.zeros_(attn.alpha_proj.weight)
                nn.init.constant_(attn.alpha_proj.bias, self._delta_alpha_init)
            if hasattr(attn, 'beta_proj'):
                nn.init.zeros_(attn.beta_proj.weight)
                nn.init.constant_(attn.beta_proj.bias, self._delta_beta_init)
            if getattr(attn, 'output_gate_enabled', False) and hasattr(attn, 'output_gate'):
                nn.init.zeros_(attn.output_gate.weight)
                nn.init.zeros_(attn.output_gate.bias)

    def tie_weights(self):
        """重新绑定 output_head 和 embedding 的权重（在 .to(device) 后调用以确保共享生效）。"""
        if self._tie_weights:
            self.output_head.weight = self.embedding.weight

    def to(self, *args: Any, **kwargs: Any):
        """重写 to() 方法，在设备迁移后自动重新绑定权重共享。"""
        module = super().to(*args, **kwargs)
        if self._tie_weights:
            self.output_head.weight = self.embedding.weight
        # 第十四轮：shared_alibi 在 .to(device) 后重新绑定共享
        # PyTorch _apply 遍历每个 module 独立处理 buffer，会打破 alibi_slopes 的对象共享
        # （数值仍正确，但失去减参优势）。重新绑定恢复共享关系。
        if self.shared_alibi_enabled:
            _shared = None
            for blk in self.blocks:
                if hasattr(blk, 'attn') and hasattr(blk.attn, 'alibi_slopes'):
                    if _shared is None:
                        _shared = blk.attn.alibi_slopes
                    else:
                        blk.attn.alibi_slopes = _shared
        return module

    def forward(self, src: torch.Tensor, past_key_values: Optional[List[Optional[Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]]]] = None, use_cache: bool = False, intuition: Optional[torch.Tensor] = None, igmcg_force_off: bool = False, temperature: float = 1.0) -> Tuple[torch.Tensor, Optional[List[Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]]]]:
        # src: (batch, seq_len)；RoPE 在注意力内部按位置旋转，无需外部 PE
        # 阶段8.7 IGMCG 2.0：intuition 为 (B,7) 连续直觉向量（训练期可作为条件输入，推理期可选）；
        # igmcg_force_off 用于训练期 IGMCG-SEL（随机整批关闭 IGMCG 引导，让模型学"何时用"）。
        x = self.embedding(src) * math.sqrt(self.embedding_dim)
        x = self.drop(x)
        # 学习型分词：字符级序列融合为词表示（门控卷积，受 LM loss 监督）
        if self.char_merge_enabled:
            x = self.char_merge(x)
        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)
        # 将旧元组包装为 BlockState（向后兼容：from_tuple 处理 None 和旧格式）
        block_states = [BlockState.from_tuple(pk) for pk in past_key_values]
        presents: List[Optional[BlockState]] = []
        start_pos = 0
        if use_cache:
            for bs in block_states:
                if bs is not None and bs.attn_kv is not None:
                    start_pos = bs.start_pos
                    break
        ssm_states: List[Optional[torch.Tensor]] = []
        ssm_conv_states: List[Optional[torch.Tensor]] = []
        if use_cache:
            for bs in block_states:
                ssm_states.append(bs.ssm_hidden if bs is not None else None)
                ssm_conv_states.append(bs.ssm_conv if bs is not None else None)
        else:
            ssm_states = [None] * len(self.blocks)
            ssm_conv_states = [None] * len(self.blocks)

        # 可学习压缩记忆：每个样本独立槽。重置时机：
        #  - 训练/全量前向（use_cache=False）：每 batch 独立，首步重建；
        #  - 增量解码首步（use_cache=True 且 past_key_values 为空）：新序列起点，重建；
        #  - 增量解码后续步（use_cache=True 且已有 past）：保留记忆并持续累积，
        #    否则生成期每步 reset 会让记忆只剩当前 token，与训练行为（整条序列累积）脱节。
        memory = None
        if self.memory_enabled:
            memory = self.memory_bank
            # 新序列起点判定：非缓存全量前向，或缓存解码且尚无任何 past（即首个生成步）。
            # 注意 past_key_values 是空列表 [None]*N 而非 None，故需逐个判空；否则训练后
            # 同一实例直接 generate 会沿用训练期的 batch 大小槽，导致形状不匹配。
            is_fresh = (not use_cache) or all(pk is None for pk in past_key_values)
            if is_fresh or memory.slots.shape[0] != x.size(0):
                # 首步重建记忆槽：用记忆库权重所在设备（DML 别名 privateuseone:0 的权威设备），
                # 避免后续热路径每步因 x.device 被剥索引而触发 .to() 拷贝。
                memory.reset(x.size(0), self.memory_bank.compress.weight.device, x.dtype)
                # 清理注意力掩码缓存（训练期 _bias_cache/_mask 按大 T 构建，解码首步 T 不同，
                # 避免复用到错误尺寸的缓存导致形状不匹配）。
                for blk in self.blocks:
                    if hasattr(blk, 'attn'):
                        blk.attn._bias_key = None
                        blk.attn._cached_T = -1

        # 第十一轮：跨层稀疏路由——收集每层输出供后续层 top-k 路由（残差注入）。
        # 仅 cross_layer_routing=True 且 num_layers>1 时启用（cross_router 已在 __init__ 创建）。
        prev_outputs: List[torch.Tensor] = [] if self.cross_layer_routing else []
        # 第十二轮：层间 SSM 状态传递——跟踪前一个 hybrid 块的输出
        prev_hybrid_x: Optional[torch.Tensor] = None
        # 第十四轮：输入全局高速公路——保存 embedding 输出 x0 供每层门控注入
        # 增量解码（use_cache=True）时首步缓存 x0（整条 prompt 的 embedding），
        # 后续步 src 只有 1 个 token，若不缓存会让 input_highway 注入错误内容（单 token embedding）。
        # 此外，后续步 x shape [B,1,D] 与 x0 shape [B,T_prompt,D] 不匹配，
        # 直接 broadcast 会让 x 被放大到 [B,T_prompt,D] 破坏后续层——故后续步取 x0 mean-pool 对齐。
        if self.input_highway_enabled:
            if use_cache and past_key_values is not None and any(pk is not None for pk in past_key_values):
                # 增量解码后续步：用首步缓存的 x0，但 mean-pool 到 [B,1,D] 与当前 x [B,1,D] 对齐
                x0_cached = getattr(self, '_cached_x0', None)
                if x0_cached is None or x0_cached.shape[0] != x.shape[0]:
                    x0 = x  # 兜底：缓存丢失，用当前 x
                else:
                    # mean-pool prompt 整体信息为单 slot，注入当前 token
                    x0 = x0_cached.mean(dim=1, keepdim=True)
            else:
                # 全量前向或增量解码首步：当前 x 即完整 embedding
                x0 = x
                if use_cache:
                    self._cached_x0 = x.detach()  # 缓存供后续步用（detach 避免长生命周期图）
        else:
            x0 = None
        # 性能优化：input_highway_proj(x0) 预计算一次（x0 在循环中不变），避免每层重复 Linear
        x0_proj = self.input_highway_proj(x0) if x0 is not None else None
        # 第十四轮：层间对比绑定——训练期累积相邻层余弦相似度损失
        if self.training and self.layer_contrastive_enabled:
            self._contrastive_loss = x.new_zeros(())
            _prev_layer_out = x.detach()
        else:
            self._contrastive_loss = None
            _prev_layer_out = None
        for i, block in enumerate(self.blocks):
            if (not self.training) and getattr(self, '_pruned_layers', None) and i in self._pruned_layers:
                presents.append(block_states[i])
                # pruned 层输出 = 输入（x 不变），作为下一层路由候选保留
                if self.cross_layer_routing:
                    prev_outputs.append(x)
                continue
            # 第十三轮：跨层 FiLM 调制——用前一层输出 x（此时 x 即 layer i-1 的输出）
            # 经 layer_film_projs[i] 产生 (γ, β)，仿射调制当前层输入：x = x*(1+γ) + β
            # init γ=β=0 → 恒等，向后兼容；逐 token 调制（非标量），比 cross_layer_routing
            # 的 top-k 残差注入更细粒度（multiplicative + additive）。
            # SEL 交替训练：_rt["layer_film"]=False 时跳过（恒等）
            if (self.layer_film_enabled and i > 0
                    and getattr(self, '_rt_layer_film', True)):
                gamma_beta = self.layer_film_projs[i](x)  # (B, T, 2D)
                gamma, beta = gamma_beta.chunk(2, dim=-1)  # each (B, T, D)
                # tanh 限制 gamma∈(-1,1) → x*(1+γ)∈(0, 2x)，防止深层堆叠数值爆炸
                gamma = torch.tanh(gamma)
                x = x * (1.0 + gamma) + beta
            # 第十四轮：输入全局高速公路——embedding 输出 x0 经门控注入当前层
            # gate=sigmoid(W·x+b)，init b=-3 → sigmoid≈0.05（弱注入，训练中自决）
            # SEL 交替训练：_rt_input_highway=False 时跳过
            # 性能优化：input_highway_proj(x0) 在循环外计算一次（x0 不变），避免每层重复 Linear
            if (self.input_highway_enabled and i > 0
                    and self._rt_input_highway):
                gate = torch.sigmoid(self.input_highway_gates[i](x))  # (B, T, 1)
                x = x + gate * x0_proj
            # 跨层路由：在 block 处理前注入前层 top-k 加权残差到 x（稀疏+残差+选择性）
            if self.cross_layer_routing and i > 0 and prev_outputs:
                x = self.cross_router.route(i, x, prev_outputs)
            # 第十二轮：层间 SSM 状态传递——hybrid 块间传递 SSM 信息
            # 前一个 hybrid 块的输出经 cross_ssm_proj 投影后加到当前 hybrid 块的输入
            if (self.cross_ssm_transfer and i > 0 and prev_hybrid_x is not None
                    and self.layer_plan[i] == 'hybrid'):
                x = x + self.cross_ssm_proj(prev_hybrid_x)
            ssm_past_state = ssm_states[i] if use_cache else None
            ssm_past_conv_state = ssm_conv_states[i] if use_cache else None
            # block 内部仍用旧元组接口（TransformerBlock.forward 未改），传入 block_states[i].to_tuple()
            x, present = block(x, block_states[i].to_tuple() if block_states[i] is not None else None,
                               use_cache, start_pos, ssm_past_state, ssm_past_conv_state, memory)
            presents.append(BlockState.from_tuple(present))
            # 第十四轮：层间对比绑定——累积相邻层 (1 - cos_sim) 损失
            # detach _prev_layer_out 使梯度只回流到当前层（推当前层向上一层靠拢，不反过来）
            if _prev_layer_out is not None:
                cos_sim = F.cosine_similarity(x, _prev_layer_out, dim=-1).mean()
                self._contrastive_loss = self._contrastive_loss + (1.0 - cos_sim)
                _prev_layer_out = x.detach()
            if self.cross_layer_routing:
                prev_outputs.append(x)
            # 记录 hybrid 块输出供下一个 hybrid 块使用
            if self.cross_ssm_transfer and self.layer_plan[i] == 'hybrid':
                prev_hybrid_x = x
        x = self.ln_f(x)
        # 阶段8.1：n-gram 神经融合——z_neural + g_t·ngram_vec。ngram_vec 是固定统计缓冲
        # （.detach() 不引梯度，主干 z_neural 仍吃完整 CE 梯度、不被缩放 → 不塌缩）。
        # g_t=sigmoid(h_t·W_g) 逐位置自决多信 n-gram，且随 use_cache 增量解码逐 token 计算也一致
        # （前向每步传入当前序列，logprob_matrix 按位置上下文查表，与全量路径共享同一张表）。
        if self.ngram_fusion_enabled and getattr(self, '_ngram_fusion_active', True):
            fused = self._apply_ngram_fusion(
                x, src, use_cache, past_key_values,
                temperature, igmcg_force_off, intuition)
            if use_cache:
                return fused, [bs.to_tuple() if bs is not None else None for bs in presents]
            return fused
        if use_cache:
            return self.output_head(x), [bs.to_tuple() if bs is not None else None for bs in presents]
        return self.output_head(x)

    def reset_ngram_state(self) -> None:
        """集中管理增量解码 n-gram 滚动缓冲（_ngram_last_ids）的重置。

        避免调用方（generate.py / model.generate）直接戳实例变量，统一由模型拥有者
        管理状态（INT 后续整合：消除跨模块直接改模型内部状态的隐患）。"""
        self._ngram_last_ids = None

    def _apply_ngram_fusion(
        self, x: torch.Tensor, src: torch.Tensor,
        use_cache: bool, past_key_values: Optional[List],
        temperature: float, igmcg_force_off: bool,
        intuition: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """将 n-gram 统计先验融合到主干 logits，返回 (B, T, V) 对数概率。

        逻辑从 forward 内联提取（原 L742-801），保持前向路径关注点分离。
        """
        ctx_len = max(1, getattr(self.ngram_model, 'max_order', 10) - 1)
        with torch.no_grad():
            if use_cache and past_key_values is not None:
                if getattr(self, '_ngram_last_ids', None) is None or \
                        self._ngram_last_ids.shape[0] != src.shape[0]:
                    pad = self.ngram_model.vocab.pad_idx \
                        if hasattr(self.ngram_model.vocab, 'pad_idx') else 0
                    self._ngram_last_ids = src.new_full((src.shape[0], ctx_len), pad)
                ngram_ord = self.ngram_model.logprob_orders_incremental(
                    self._ngram_last_ids, src, x.device)
                self._ngram_last_ids = torch.cat([self._ngram_last_ids, src], dim=1)[:, -ctx_len:]
            else:
                ngram_ord = self.ngram_model.logprob_orders_matrix(src, x.device)
        # 爆炸防护 0：ngram_ord 可能含 -inf（log(0)）或 NaN（空上下文除零），
        # 在加权混合前先兜底，防 -inf * weight 传播到 ngram_vec。
        ngram_ord = torch.nan_to_num(ngram_ord, nan=0.0, posinf=0.0, neginf=-30.0)
        # 爆炸防护 0b：ngram_order_logits clamp 到 [-10, 10]——防 softmax 饱和
        # （极端 logits 使某阶权重→1，其他→0，失去多阶混合意义；softmax 本身
        # 数值稳定但失去多阶信息）。init=0 → softmax 均匀，clamp 不影响初始行为。
        _ow = torch.softmax(self.ngram_order_logits.clamp(-10.0, 10.0), dim=0)
        ngram_vec = (ngram_ord * _ow.view(1, 1, 1, -1)).sum(-1)
        # 爆炸防护 1：n-gram log 概率理论上 ≤ 0，但数值噪声可能略正；极罕见 token
        # 的 log prob 可达 -50+，经 gate 放大后使 logits 爆炸。clamp 到 [-30, 0]
        # （exp(-30)≈1e-13，已远低于 softmax 有效阈值，不影响采样分布）。
        ngram_vec = ngram_vec.clamp(-30.0, 0.0)
        z = self.output_head(x)
        # 爆炸防护 2：温度 clamp 到 [0.01, 10]——过小 z/_t 爆炸到 ±inf，过大退化为均匀。
        _t = torch.as_tensor(max(0.01, min(10.0, float(temperature))), dtype=z.dtype, device=z.device).view(-1, 1, 1)
        logp = F.log_softmax(z / _t, dim=-1)
        g_strength = torch.sigmoid(self.ngram_gate(x))
        # 爆炸防护 3：ngram_gate_scale clamp 到 [0, 10]——用户可经 set_ngram_gate_scale
        # 设大值（如 100），此时 gate·ngram_vec 可使 logits 极端化致采样塌缩。
        _scale = max(0.0, min(10.0, float(self.ngram_gate_scale)))
        if self.igmcg_enabled and not igmcg_force_off:
            _shift = 0.0
            if intuition is not None:
                _shift = self.intuition_proj(intuition).unsqueeze(1)
            p_use = torch.sigmoid(self.igmcg_use_gate(x) + _shift)
            gate = p_use * g_strength * _scale
        else:
            gate = g_strength * _scale
        # 爆炸防护 3b：gate clamp 到 [0, 10]——g_strength∈(0,1) * _scale∈[0,10] ≤10，
        # 但浮点误差可能略超；clamp 确保 gate·ngram_vec 的 magnitude 有界。
        gate = gate.clamp(0.0, 10.0)
        fused = logp + gate * ngram_vec
        # 爆炸防护 4：NaN/Inf 兜底——直接 nan_to_num（无 .any() 的 CPU 同步税）。
        # 正常情况下无 NaN/Inf，nan_to_num 是 no-op（DML 原生 kernel，开销极小）。
        fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=-30.0)
        # 爆炸防护 5：最终 logits clamp 到 [-100, 100]——防 exp() 溢出，softmax 后不影响排名。
        return fused.clamp(-100.0, 100.0)

    def generate(self, token_ids: List[int], max_length: int = 50, temperature: float = 1.0, top_k: int = 50,
                  device: str = 'cpu', repetition_penalty: float = 2.0,
                  ngram_fn: Optional[Callable[[List[int], str], torch.Tensor]] = None, ngram_weight: float = 0.0,
                  eos_id: int = 3, pad_id: int = 0, sep_id: int = 4,
                  min_length: int = 3, eos_penalty: float = -5.0) -> List[int]:
        """生成文本（自回归解码）。

        Args:
            token_ids: 初始 token id 列表
            max_length: 最大生成长度
            temperature: 采样温度
            top_k: top-k 采样，<=0 禁用，>=vocab_size 视为全词表
            device: 设备
            repetition_penalty: 重复惩罚系数
            ngram_fn: n-gram 先验函数
            ngram_weight: n-gram 权重
            eos_id: EOS token id
            pad_id: PAD token id
            sep_id: SEP token id
            min_length: 最小生成长度（不含 prompt），默认 3（避免过短生成）
            eos_penalty: EOS 惩罚值，默认 -5.0（负值抑制 EOS，正值鼓励 EOS）
        """
        self.eval()
        # 增量解码 n-gram 滚动缓冲在每次新序列开头清空，避免跨序列串味。
        self.reset_ngram_state()
        generated = list(token_ids)
        max_seq_length = self.max_seq_length
        eos_token_id = eos_id
        pad_token_id = pad_id
        sep_token_id = sep_id
        # 现在支持混合架构的增量解码（SSM 也有增量状态）
        use_cache = True

        def sample_step(logits_t: torch.Tensor) -> Optional[int]:
            # INT-2：单步采样统一走 sample_next_token（与 IGMCG 批量解码共用单一事实来源）
            # 阶段8.9：n-gram 融合路径下，温度已在 forward 内作用于主干 logits，
            # 此处标记 temperature_applied 避免对整体（含 n-gram 先验）再除以 τ。
            return sample_next_token(
                logits_t, temperature=temperature, repetition_penalty=repetition_penalty,
                generated_ids=generated, ngram_fn=ngram_fn, ngram_weight=ngram_weight,
                device=device, pad_id=pad_token_id, sep_id=sep_token_id, eos_id=eos_token_id,
                generated_len=len(generated) - len(token_ids), min_length=min_length,
                eos_penalty=eos_penalty, top_k=top_k, vocab_size=logits_t.shape[0],
                raw_logits=logits_t,
                temperature_applied=getattr(self, 'ngram_fusion_enabled', False),
            )

        with torch.no_grad():
            past = None
            cur_pos = 0
            # use_cache 固定为 True（增量解码），无 cache 的每步全量重算分支已删除（死代码、
            # 且极慢易错）；如未来需要无缓存生成，应单独实现而非复用此循环。
            input_ids = torch.tensor([generated], dtype=torch.long, device=device)
            logits, past = self.forward(input_ids, past_key_values=None, use_cache=True,
                                        temperature=temperature)
            cur_pos = input_ids.size(1)

            for _ in range(max_length):
                if cur_pos >= max_seq_length:
                    break
                next_token = sample_step(logits[0, -1, :])
                if next_token is None:
                    break
                generated.append(next_token)
                if next_token == eos_token_id and len(generated) - len(token_ids) >= min_length:
                    break
                past, logits, cur_pos = _decode_one_step(
                    self, next_token, past, cur_pos, device=device, use_cache=True,
                    temperature=temperature,
                    temperature_applied=getattr(self, 'ngram_fusion_enabled', False))
        return generated
