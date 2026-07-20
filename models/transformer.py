from __future__ import annotations

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.functional import scaled_dot_product_attention
import threading
from typing import Optional, List, Tuple, Any, Dict, Callable

from models.constants import MASK_FILL_VALUE, ROPE_BASE


from models.norms import RMSNorm
from models.rope import RotaryEmbedding
from models.memory import MemoryBank
from models.mixers import (SlidingWindowCausalSelfAttention, LinearAttention,
                           MambaSSM, SwiGLU, apply_qk_norm_and_temp)
from models.sampling import (apply_repetition_penalty, sample_next_token,
                             _decode_one_step)
from models.layers import CharMergeLayer

# 向后兼容：保留原模块级符号的外部可见性（其它模块仍从 models.transformer 导入）
__all__ = [
    "RMSNorm", "RotaryEmbedding", "MemoryBank",
    "SlidingWindowCausalSelfAttention", "LinearAttention", "MambaSSM", "SwiGLU",
    "apply_qk_norm_and_temp", "apply_repetition_penalty", "sample_next_token",
    "_decode_one_step", "CharMergeLayer",
    "TransformerBlock", "_parse_layer_plan", "TransformerModel",
]


class TransformerBlock(nn.Module):
    """可配置混合块：attn / ssm / hybrid(attn+ssm 并行)。Pre-LN。"""
    def __init__(self, dim: int, num_heads: int, hidden_dim: int, block_type: str = 'attn',
                 dropout: float = 0.0, max_seq_length: int = 64,
                 ssm_kwargs: Optional[Dict[str, Any]] = None, attn_kwargs: Optional[Dict[str, Any]] = None,
                 residual_gate: bool = True, hybrid_gate: bool = True, gradient_checkpointing: bool = True,
                 skip: bool = False, mixer: str = 'attn',
                 hybrid_single_gate: bool = False):
        super().__init__()
        self.block_type = block_type
        self.drop = nn.Dropout(dropout)
        ssm_kwargs = ssm_kwargs or {}
        attn_kwargs = attn_kwargs or {}
        # ②/⑥ 残差门控 & ⭐A 混合路径门控开关（默认关，向后兼容）
        self.residual_gate_enabled = residual_gate
        self.hybrid_gate_enabled = hybrid_gate
        # 运行时增强开关（按开关粒度，用于“交替/分段增强”训练）：默认全开
        self._rt: Dict[str, bool] = {"residual_gate": True, "hybrid_gate": True}
        self.gradient_checkpointing = gradient_checkpointing
        # Both attn and ssm blocks need a pre-norm layer
        self.ln1 = RMSNorm(dim)
        if block_type in ('attn', 'hybrid'):
            # 阶段7 token mixer 选择：attn(默认) / linear(纯线性注意力) /
            # attn_linear(attn+线性注意力 两路并行，可学 mixer_gate 自选择比例)。
            # 旧配置字符串 'hybrid' 等价于 'attn_linear'（向后兼容）。
            if mixer == 'hybrid':
                mixer = 'attn_linear'
            self.mixer = mixer
            self.attn, self.linear_attn, self.mixer_gate = self._build_attn_mixer(
                mixer, dim, num_heads, max_seq_length, attn_kwargs)
        if block_type in ('ssm', 'hybrid'):
            self.ssm = MambaSSM(dim, **ssm_kwargs)
        self.ln2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden_dim)
        # ②/⑥ 每层可学习残差门控：x = x + gate * f(x)（init 1.0，默认行为不变）
        if residual_gate:
            # hybrid 块的第一子层用 hybrid_attn_gate/hybrid_ssm_gate，sub1_gate 无用，跳过分配
            if block_type != 'hybrid':
                self.sub1_gate = nn.Parameter(torch.ones(1))   # 第一子层残差（attn 或 ssm）
            self.ffn_gate = nn.Parameter(torch.ones(1))    # FFN 子层残差
        # ⭐A 混合块内 attn/ssm 两路可学习门控（init 1.0）
        if hybrid_gate and block_type == 'hybrid':
            self.hybrid_attn_gate = nn.Parameter(torch.ones(1))
            self.hybrid_ssm_gate = nn.Parameter(torch.ones(1))
        # 阶段8.4：单动态门控 g_t（默认关，向后兼容）。用 g_t=sigmoid(W_g·ln1(x)) 逐位置混合
        # attn 与 ssm（g_t·attn_h + (1-g_t)·ssm_h），替代原两独立标量门控相加（双残差、非真融合）。
        # 单门控是凸组合、逐位置动态、参数更少，架构更干净（见 §8 推进顺序 #4）。
        self.hybrid_single_gate = hybrid_single_gate and block_type == 'hybrid'
        if self.hybrid_single_gate:
            self.hybrid_mix = nn.Linear(dim, 1)
        # 阶段6：可学习跳过层（skip gate）——sigmoid 门控，模型自决本层是否跳过。
        # skip≈1 走残差（等效跳过该块计算），推理时可按阈值静态剪枝省算力。
        self.skip_enabled = skip
        if skip:
            self.skip_gate = nn.Parameter(torch.ones(1))  # init 1.0 = 不跳过（默认保留全部层）
        self._skip_active = True

    @staticmethod
    def _build_attn_mixer(mixer: str, dim: int, num_heads: int, max_seq_length: int,
                           attn_kwargs: Dict[str, Any]) -> Tuple[nn.Module, Optional[nn.Module], Optional[nn.Parameter]]:
        """构造 attn 系 token mixer（A 项归一点，消除 attn/hybrid 块中的重复构建逻辑）。

        Returns:
            attn: 主注意力模块（SlidingWindowCausalSelfAttention 或 LinearAttention）
            linear_attn: 并行线性注意力分支（仅 mixer='attn_linear' 时非 None）
            mixer_gate: attn/linear 两路混合门控（仅 mixer='attn_linear' 时非 None）
        """
        if mixer == 'linear':
            # 阶段7：纯线性注意力（O(N) token mixer）
            attn = LinearAttention(dim, num_heads, max_seq_length=max_seq_length,
                                   qk_norm=attn_kwargs.get('qk_norm', True),
                                   attn_temp=attn_kwargs.get('attn_temp', True),
                                   feature=attn_kwargs.get('linear_attn_feature', 'relu'))
            return attn, None, None
        if mixer == 'attn_linear':
            # 阶段7：attn + 线性注意力 两路并行，可学习 mixer_gate 自选择用多少
            attn_only = {k: v for k, v in attn_kwargs.items()
                         if k not in ('linear_attn_feature', 'linear_attn_head_dim')}
            attn = SlidingWindowCausalSelfAttention(dim, num_heads, max_seq_length=max_seq_length, **attn_only)
            linear_attn = LinearAttention(dim, num_heads, max_seq_length=max_seq_length,
                                          qk_norm=attn_kwargs.get('qk_norm', True),
                                          attn_temp=attn_kwargs.get('attn_temp', True),
                                          feature=attn_kwargs.get('linear_attn_feature', 'relu'),
                                          head_dim=attn_kwargs.get('linear_attn_head_dim', None),
                                          rope_learnable=attn_kwargs.get('rope_learnable', False))
            mixer_gate = nn.Parameter(torch.ones(1))  # init 1.0 → 偏 attn
            return attn, linear_attn, mixer_gate
        # 默认：标准滑动窗口因果注意力
        attn_only = {k: v for k, v in attn_kwargs.items()
                     if k not in ('linear_attn_feature', 'linear_attn_head_dim')}
        attn = SlidingWindowCausalSelfAttention(dim, num_heads, max_seq_length=max_seq_length, **attn_only)
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
        else:
            # LinearAttention 等无 attend 接口的 mixer：直接对整层前向做检查点
            if ckpt:
                h, present = checkpoint(self.attn, xn, attn_past_kv, use_cache, start_pos, mem_kv, use_reentrant=False)
            else:
                h, present = self.attn(xn, attn_past_kv, use_cache, start_pos, memory_kv=mem_kv)
        if self.linear_attn is not None:
            # 阶段7：混合 mixer（attn + 线性注意力并行），mixer_gate 自选择比例
            lh, lpresent = self.linear_attn(xn, attn_past_kv, use_cache, start_pos, memory_kv=mem_kv)
            mg = torch.sigmoid(self.mixer_gate).to(xn.device)
            h = mg * h + (1.0 - mg) * lh
            if use_cache and lpresent is not None:
                # 两路 KV 缓存合并：把线性注意力状态 S 和分母累积 z 塞进 attn_kv 元组
                # (k, v, linear_S, z_final)，保持块级 (attn_kv, ssm_state, ssm_conv_state) 三元组不变。
                present = ((present[0], present[1], lpresent[2], lpresent[3] if len(lpresent) > 3 else None), None, None)
        return h, present

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
        gate1 = (getattr(self, 'sub1_gate', None) if (self.residual_gate_enabled and self._rt["residual_gate"]) else None)
        gate2 = (self.ffn_gate if (self.residual_gate_enabled and self._rt["residual_gate"]) else None)
        # 阶段6：跳过层门控（skip_gate 经 sigmoid 映射到 (0,1)；_skip_active=False 时跳过失效恒为 1）
        sk = torch.sigmoid(self.skip_gate).to(x.device) if (self.skip_enabled and getattr(self, '_skip_active', True)) else None

        if self.block_type == 'attn':
            # 阶段7 token mixer（attn / linear / attn_linear），统一经 _run_attn_mixer 运行，
            # 两者共享同一 ln1(x) 避免重复 RMSNorm。
            xn = self.ln1(x)
            h, present = self._run_attn_mixer(xn, attn_past_kv, use_cache, start_pos, mem_kv, ckpt)
            h_eff = (sk * h) if sk is not None else h
            x = x + self.drop(gate1 * h_eff if gate1 is not None else h_eff)
        elif self.block_type == 'ssm':
            if ckpt:
                h, ssm_present_state, ssm_present_conv_state = checkpoint(self.ssm, self.ln1(x), ssm_past_state, ssm_past_conv_state, use_cache, use_reentrant=False)
            else:
                h, ssm_present_state, ssm_present_conv_state = self.ssm(self.ln1(x), past_state=ssm_past_state, past_conv_state=ssm_past_conv_state, use_cache=use_cache)
            h_eff = (sk * h) if sk is not None else h
            x = x + self.drop(gate1 * h_eff if gate1 is not None else h_eff)
        elif self.block_type == 'hybrid':
            xn = self.ln1(x)
            # attn 部分经 _run_attn_mixer 运行（与 attn 块共用）；ssm 部分并行。
            h, attn_present = self._run_attn_mixer(xn, attn_past_kv, use_cache, start_pos, mem_kv, ckpt)
            if ckpt:
                ssm_h, ssm_present_state, ssm_present_conv_state = checkpoint(self.ssm, xn, ssm_past_state, ssm_past_conv_state, use_cache, use_reentrant=False)
            else:
                ssm_h, ssm_present_state, ssm_present_conv_state = self.ssm(xn, past_state=ssm_past_state, past_conv_state=ssm_past_conv_state, use_cache=use_cache)
            h_eff = (sk * h) if sk is not None else h
            if self.hybrid_single_gate and self._rt.get("hybrid_gate", True):
                # 阶段8.4：单动态门控 —— g_t 逐位置混合 attn/ssm 两路（凸组合）。
                # g_t=sigmoid(W_g·ln1(x)) ∈(0,1)，out = g_t·attn_h + (1-g_t)·ssm_h。
                g = torch.sigmoid(self.hybrid_mix(xn)).to(x.device)  # (B,T,1)
                mixed = g * h_eff + (1.0 - g) * ssm_h              # (B,T,D)
                x = x + self.drop(mixed)
            elif self.hybrid_gate_enabled and self._rt["hybrid_gate"]:
                # ⭐A 混合块：attn 与 ssm 两路各自可学习门控，让模型自决每层偏重
                x = x + self.drop(self.hybrid_attn_gate * h_eff) \
                      + self.drop(self.hybrid_ssm_gate * ssm_h)
            else:
                x = x + self.drop(h_eff) + self.drop(ssm_h)
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
        x = x + self.drop(gate2 * f if gate2 is not None else f)
        # Combine attn KV cache and SSM state
        if use_cache:
            if self.block_type == 'attn':
                # hybrid mixer 已构造 (attn_kv, linear_S, None) 三元组，勿重复包裹
                if getattr(self, 'linear_attn', None) is None:
                    present = (present, None, None)  # (attn_kv, ssm_state, ssm_conv_state)
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
            self._rt = {"residual_gate": on, "hybrid_gate": on}
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


class TransformerModel(nn.Module):
    """现代 decoder-only 语言模型（Pre-LN + RMSNorm + RoPE + SwiGLU + 权重共享）。

     支持混合架构：通过 layer_plan 指定每层为 attn / ssm / hybrid。
     默认 layer_plan=None 时全为 attn，与旧权重完全兼容。
    """

    # INT-3：增强开关键名单一事实来源（attn 层 qk_norm/attn_temp + block 层
    # residual_gate/hybrid_gate 的并集中所有可分段 SEL 交替的开关），供
    # train.py 的 enhancement_schedule 与测试派生，避免键名清单散落 4 处漂移。
    ENHANCEMENT_KEYS = ("qk_norm", "attn_temp", "residual_gate", "hybrid_gate")

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
                 mask_fill_value: float = -1e9,
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
                   ngram_fusion: bool = False, ngram_model=None,
                   ngram_gate_scale: float = 1.0, igmcg: bool = False):
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
        )
        attn_kwargs = dict(window=attn_window, rel_bias=attn_rel_bias,
                           qk_norm=qk_norm, attn_temp=attn_temp,
                           rope_learnable=rope_learnable, alibi=alibi,
                           retrieval_full=memory_retrieval_full,
                           retrieval_topk=memory_retrieval_topk,
                           learn_window=learn_window, window_base=window_base,
                           linear_attn_feature=linear_attn_feature,
                           linear_attn_head_dim=linear_attn_head_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, hidden_dim, block_type=bt,
                             dropout=dropout, max_seq_length=rope_max_len,
                             ssm_kwargs=ssm_kwargs, attn_kwargs=attn_kwargs,
                             residual_gate=residual_gate, hybrid_gate=hybrid_gate,
                             hybrid_single_gate=hybrid_single_gate,
                             gradient_checkpointing=gradient_checkpointing,
                             skip=layer_skip, mixer=mixer)
            for bt in self.layer_plan
        ])
        self.ln_f = RMSNorm(embedding_dim)
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
        # 权重初始化（_init_weights 遍历所有 Linear 用 N(0,0.02)，再对 SSM 调 proper_init 覆盖）
        self._init_weights()

    def set_enhancements_active(self, spec):
        """运行时开关（按开关粒度）：`spec=True/False` 全开/全关；`spec=dict` 按键更新。
        用于“交替/分段增强”训练（关闭则跳过对应增强，恒等）。"""
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
            elif isinstance(blk.attn, LinearAttention) and getattr(self, 'attn_window', 0) > 0:
                # 线性注意力同样处理窗口内 token，按模型配置窗口计成本（再乘 0.3x 折扣）
                eff_window = min(self.attn_window, self.max_seq_length)
            if eff_window:
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
            if isinstance(m, MambaSSM):
                m.proper_init()

    def tie_weights(self):
        """重新绑定 output_head 和 embedding 的权重（在 .to(device) 后调用以确保共享生效）。"""
        if self._tie_weights:
            self.output_head.weight = self.embedding.weight

    def to(self, *args: Any, **kwargs: Any):
        """重写 to() 方法，在设备迁移后自动重新绑定权重共享。"""
        module = super().to(*args, **kwargs)
        if self._tie_weights:
            self.output_head.weight = self.embedding.weight
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
        presents: List[Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]] = []
        start_pos = 0
        if use_cache:
            for pk in past_key_values:
                if pk is not None:
                    # pk is (attn_kv, ssm_state, ssm_conv_state)
                    if pk[0] is not None:
                        start_pos = pk[0][0].size(2)
                        break
        ssm_states: List[Optional[torch.Tensor]] = []
        ssm_conv_states: List[Optional[torch.Tensor]] = []
        if use_cache:
            # Extract SSM past states
            for pk in past_key_values:
                if pk is not None and pk[1] is not None:
                    ssm_states.append(pk[1])
                else:
                    ssm_states.append(None)
                if pk is not None and pk[2] is not None:
                    ssm_conv_states.append(pk[2])
                else:
                    ssm_conv_states.append(None)
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

        for i, block in enumerate(self.blocks):
            # 阶段8.2：推理期静态剪枝——被 prune_layers 标记的层直接跳过（直通，无计算）。
            # 仅推理模式生效；训练模式（self.training）下忽略剪枝，避免静态剪枝状态
            # 残留到训练/验证造成静默质量退化（prune_layers 是持久标记，非自动重置）。
            if (not self.training) and getattr(self, '_pruned_layers', None) and i in self._pruned_layers:
                presents.append(past_key_values[i] if past_key_values is not None else None)
                continue
            ssm_past_state = ssm_states[i] if use_cache else None
            ssm_past_conv_state = ssm_conv_states[i] if use_cache else None
            # 检查点（仅重算力部分）已在 block 内部按 self.gradient_checkpointing 处理，此处直接调用
            x, present = block(x, past_key_values[i], use_cache, start_pos, ssm_past_state, ssm_past_conv_state, memory)
            presents.append(present)
        x = self.ln_f(x)
        # 阶段8.1：n-gram 神经融合——z_neural + g_t·ngram_vec。ngram_vec 是固定统计缓冲
        # （.detach() 不引梯度，主干 z_neural 仍吃完整 CE 梯度、不被缩放 → 不塌缩）。
        # g_t=sigmoid(h_t·W_g) 逐位置自决多信 n-gram，且随 use_cache 增量解码逐 token 计算也一致
        # （前向每步传入当前序列，logprob_matrix 按位置上下文查表，与全量路径共享同一张表）。
        if self.ngram_fusion_enabled and getattr(self, '_ngram_fusion_active', True):
            # 阶段8.1 n-gram 神经融合：z_neural + g_t·ngram_vec（g_t 逐位置可学门控）。
            # 全量/训练路径（非增量）：src 含完整上下文，直接 logprob_matrix(src)。
            # 增量解码路径（use_cache 且 past 非空，即 generate 逐 token 喂入）：src 仅是新 token。
            # 阶段8.8：改用滚动增量查表 logprob_orders_incremental——仅就"新 token 各位置"按滚动
            # 上下文(末 ctx_len token，ctx_len=max_order-1)查表，不重建整段 ctx → 每步 O(T)。
            # 滚动缓冲 _ngram_last_ids 仅在增量分支维护，全量分支不写实例状态，
            # 避免两次独立调用间相互污染（保证全量/cache 单调用 parity）。
            ctx_len = max(1, getattr(self.ngram_model, 'max_order', 10) - 1)
            if use_cache and past_key_values is not None:
                if getattr(self, '_ngram_last_ids', None) is None or \
                        self._ngram_last_ids.shape[0] != src.shape[0]:
                    pad = self.ngram_model.vocab.pad_idx \
                        if hasattr(self.ngram_model.vocab, 'pad_idx') else 0
                    self._ngram_last_ids = src.new_full((src.shape[0], ctx_len), pad)
                ngram_ord = self.ngram_model.logprob_orders_incremental(
                    self._ngram_last_ids, src, x.device).detach()          # (B,T,V,K) 仅新位置
                # 更新滚动缓冲（保留末 ctx_len token），供下一步增量解码
                self._ngram_last_ids = torch.cat([self._ngram_last_ids, src], dim=1)[:, -ctx_len:]
            else:
                ngram_ord = self.ngram_model.logprob_orders_matrix(src, x.device).detach()  # (B,T,V,K)
            # 阶段8.7：可学阶混合——softmax(order_logits) 对 K 阶 logprob 加权混合（模型自选各阶占比）。
            _ow = torch.softmax(self.ngram_order_logits, dim=0)             # (K,)
            ngram_vec = (ngram_ord * _ow.view(1, 1, 1, -1)).sum(-1)         # (B,T,V) 处于 log 概率空间(≈-7..-1)
            # 阶段8.7/8.8：融合改为在"对数概率空间"进行，修复原 z(原始 logits, ±数十) 直接加
            # gate·ngram_vec(log 概率, ≈-7) 的量纲错位——原写法须让 gate 学到超大尺度才有意义，
            # n-gram 先验事实上只是微小扰动。现：logp = log_softmax(z) + gate·ngram_vec，
            # gate 语义即"先验权重"(∈(0,1) 表示混合比例)，与 ngram_vec 同尺度、可直接调节概率。
            # softmax 单调，返回 logp 与返回 logits 在采样上等价，不破坏下游（仅采样消费该输出）。
            z = self.output_head(x)                                         # (B,T,V) 主干 logits
            # 阶段8.9 温度作用域修正：温度只缩放"主干"分布（log_softmax(z/τ)），
            # 不缩放 n-gram 先验（gate*ngram_vec 处于 log 概率空间，属外部固定先验，
            # 不应被采样温度非线性放大/压缩）。旧写法 fused=logp+gate*ngram_vec 后由
            # sample_next_token 对整体除以 τ，等价于把先验也缩放 τ 倍，温度越高先验越被
            # 错误放大（τ>1 时 prior/τ 变小、反而压低先验；语义与"温度仅控随机度"相悖）。
            # 现温度在 log_softmax 内部作用于 z：logp_τ=log_softmax(z/τ)，fused=logp_τ+prior，
            # 与采样中单独对主干做温度在 softmax 前缩放完全一致；τ=1 时与原行为逐位相同。
            # temperature 可为标量或 (B,) 张量（批量解码时各候选温度不同）；
            # 统一整形为 (B,1,1) 与 (B,T,V) 主干 logits 广播。
            _t = torch.as_tensor(temperature, dtype=z.dtype, device=z.device).view(-1, 1, 1)
            logp = F.log_softmax(z / _t, dim=-1)                           # (B,T,V) 同尺度 log 概率（已温度缩放主干）
            # 门控角色分离（消除 8.1/8.7 双 (0,1) sigmoid 冗余）：
            #  - igmcg_use_gate（仅 IGMCG 启用时）："是否启用 IGMCG 引导"的逐位置自决门控（含直觉条件偏置）；
            #  - ngram_gate：逐位置"对 n-gram 先验的置信/强度"（8.1 语义，保留以兼容已训练权重）；
            #  - ngram_gate_scale：推理期总闸（用户 0~1+ 缩放，1.0=模型自决）。
            # 二者相乘仍∈(0,1)：igmcg_use_gate 为"用不用"决策、ngram_gate 为"信多少"强度，分工不冗余。
            g_strength = torch.sigmoid(self.ngram_gate(x))                  # (B,T,1) 强度
            if self.igmcg_enabled and not igmcg_force_off:
                _shift = 0.0
                if intuition is not None:
                    # 7 维直觉向量投影为 (B,1) 序列级偏置，广播到 (B,T,1) 影响 use 门控（融合训练直觉）。
                    _shift = self.intuition_proj(intuition).unsqueeze(1)   # (B,1,1)
                p_use = torch.sigmoid(self.igmcg_use_gate(x) + _shift)     # (B,T,1) 用/不用决策
                gate = p_use * g_strength * self.ngram_gate_scale
            else:
                gate = g_strength * self.ngram_gate_scale
            fused = logp + gate * ngram_vec
            if use_cache:
                return fused, presents
            return fused
        if use_cache:
            return self.output_head(x), presents
        return self.output_head(x)

    def reset_ngram_state(self) -> None:
        """集中管理增量解码 n-gram 滚动缓冲（_ngram_last_ids）的重置。

        避免调用方（generate.py / model.generate）直接戳实例变量，统一由模型拥有者
        管理状态（INT 后续整合：消除跨模块直接改模型内部状态的隐患）。"""
        self._ngram_last_ids = None

    def generate(self, token_ids: List[int], max_length: int = 50, temperature: float = 1.0, top_k: int = 50,
                  device: str = 'cpu', repetition_penalty: float = 1.2,
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
                    temperature=temperature)
        return generated
