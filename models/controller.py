# -*- coding: utf-8 -*-
"""R42 Controller/Generator 双模型编排 —— Controller 模型。

核心思想（详见 AGENT_MEMORY.md §10）：
  独立「编排/决策模型（Controller）」收口决策类功能，主模型（Generator）只管生成。
  Controller 用 GatedDeltaNet（线性复杂度 O(T) 看全上下文；状态矩阵 S 天然=上下文压缩）
  产出 3 类控制信号，经「条件化总线」注入 Generator：
    ① 压缩记忆 mem_kv (B, M, gen_head_dim) → 注入 Generator 各 block 的 attention
    ② FiLM 调制 per-layer (γ, β) (B, T, 2*gen_dim) → 仿射调制 Generator 各层输入
    ③ 生成方向向量 (B, gen_dim) → 加到 Generator embedding 输出（动态 prefix bias）

DML 兼容：复用项目已验证的 GatedDeltaNet（R36-4b DML 兼容 + chunk_scan），
全用 4 算子等价式 / no_grad / dense 化，不引入新 gather/scatter。

中性初始化：所有控制信号投影 weight=0/bias=0 → Controller 开启但信号=0 时
Generator 输出与 controller=False 完全一致（向后兼容旧权重 + 训练稳定起步）。

增量解码：GatedDeltaNet 有状态 S 天然缓存，每步前向单 token；
mem_kv 来自 S 投影（无需重算历史），FiLM/direction 来自当前 token 输出。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from models.mixers import GatedDeltaNet
from models.norms import RMSNorm


@dataclass
class ControllerOutput:
    """Controller 前向输出的三类控制信号。

    所有字段 None 表示该信号未启用（Controller 配置关闭对应输出）。
    Generator 据此条件化前向：mem_kv 拼到 attention KV；film 调制各层输入；
    direction 加到 embedding 输出。
    """
    # ① 压缩记忆：(mk, mv) 各 (B, M, gen_head_dim)；None 表示未启用
    mem_kv: Optional[Tuple[torch.Tensor, torch.Tensor]]
    # ② FiLM per-layer：List[(gamma, beta) 各 (B, T, gen_dim)]，长度=gen_layers；
    #    layer 0 元素为 None（Generator 跳过 i==0，与 layer_film 一致）；None 表示未启用
    film_per_layer: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]
    # ③ 生成方向：(B, gen_dim)；None 表示未启用
    direction: Optional[torch.Tensor]


class ControllerModel(nn.Module):
    """Controller 模型：轻量 GatedDeltaNet 决策器，产出 3 类控制信号条件化 Generator。

    架构：
      - 共享 Generator 的 embedding 层（省参数 + 词表对齐）
      - Pre-norm + controller_layers 层 GatedDeltaNet mixer（线性复杂度看全上下文）
      - 末层 GatedDeltaNet 的状态 S（B, H_ctrl, D_ctrl, D_ctrl）= 上下文压缩矩阵
      - 末层输出 x (B, T, ctrl_dim) → 各控制信号投影

    控制信号产出（均零初始化 → 中性起步）：
      ① mem_kv: S 平均到 (B, D_ctrl, D_ctrl) → mem_query (M, D_ctrl) einsum →
         (B, M, D_ctrl) → Linear(D_ctrl, 2*gen_head_dim) → chunk → (mk, mv)
      ② FiLM: 末层输出 x → 每层 Linear(ctrl_dim, 2*gen_dim) → chunk → (γ, β)
      ③ direction: x.mean(T) → Linear(ctrl_dim, gen_dim) → (B, gen_dim)

    增量解码：每步单 token 前向，mixer 内部 S 状态递推；
    mem_kv 从当前 S 投影（无历史依赖），FiLM/direction 从当前 token 输出。

    参数:
        gen_dim: Generator 隐藏维（embedding_dim）
        gen_heads: Generator 注意力头数
        gen_layers: Generator 层数（FiLM 需逐层投影）
        ctrl_dim: Controller 隐藏维（=gen_dim 时共享投影最省；可更小降算力）
        ctrl_heads: Controller 注意力头数
        ctrl_layers: Controller 层数（2-3 足够，决策不生成）
        mem_slots: 压缩记忆槽数 M（注入 Generator attention 的 mem_kv 行数）
        max_seq_length: 最大序列长度（RoPE 缓冲区）
        embedding_layer: 共享的 nn.Embedding（Generator 传入引用）
        use_direction/use_film/use_memory_compress: 控制信号开关（按配置启用）
    """

    def __init__(self, gen_dim: int, gen_heads: int, gen_layers: int,
                 ctrl_dim: int, ctrl_heads: int, ctrl_layers: int,
                 mem_slots: int, max_seq_length: int,
                 embedding_layer: nn.Embedding,
                 use_direction: bool = True, use_film: bool = True,
                 use_memory_compress: bool = True):
        super().__init__()
        assert ctrl_dim % ctrl_heads == 0, (
            f"ctrl_dim ({ctrl_dim}) must be divisible by ctrl_heads ({ctrl_heads})")
        self.gen_dim = gen_dim
        self.gen_heads = gen_heads
        self.gen_layers = gen_layers
        self.ctrl_dim = ctrl_dim
        self.ctrl_heads = ctrl_heads
        self.ctrl_layers = ctrl_layers
        self.mem_slots = mem_slots
        self.use_direction = use_direction
        self.use_film = use_film
        self.use_memory_compress = use_memory_compress
        self.gen_head_dim = gen_dim // gen_heads
        self.ctrl_head_dim = ctrl_dim // ctrl_heads
        # 共享 embedding（由 Generator 传入引用，省一份词表参数）
        self.embedding = embedding_layer
        # ctrl_dim != gen_dim 时需投影：embedding 输出 (B,T,gen_dim) → (B,T,ctrl_dim)
        # gen_dim==ctrl_dim 时跳过投影（零开销，共享 embedding 直连 ln_pre）
        self.input_proj = (nn.Linear(gen_dim, ctrl_dim, bias=False)
                           if ctrl_dim != gen_dim else None)
        # Pre-norm + per-layer norm（与 Generator 同 RMSNorm）
        self.ln_pre = RMSNorm(ctrl_dim)
        self.ln_layers = nn.ModuleList([RMSNorm(ctrl_dim) for _ in range(ctrl_layers)])
        # GatedDeltaNet mixer：线性复杂度看全上下文，S 矩阵=上下文压缩
        # 用项目已验证的默认参数（alpha_init=-2 弱遗忘 / beta_init=2 强写入）
        self.mixers = nn.ModuleList([
            GatedDeltaNet(dim=ctrl_dim, num_heads=ctrl_heads, qk_norm=True, attn_temp=True,
                          max_seq_length=max_seq_length, alpha_init=-2.0, beta_init=2.0)
            for _ in range(ctrl_layers)])
        # ① 压缩记忆投影：S (B, H_ctrl, D_head, D_head) → mem_query (M, H_ctrl, D_head) einsum
        #   → (B, M, D_head) → Linear(D_head, 2*gen_head_dim) → chunk → (mk, mv)
        # S 的状态矩阵是 GatedDeltaNet 的上下文压缩（delta rule 天然压历史），每头独立一份。
        if use_memory_compress:
            self.mem_query = nn.Parameter(torch.randn(mem_slots, ctrl_heads, self.ctrl_head_dim) * 0.02)
            self.mem_proj = nn.Linear(self.ctrl_head_dim, 2 * self.gen_head_dim, bias=False)
        # ② FiLM per-layer 投影：末层输出 (B, T, ctrl_dim) → 每层 Linear(ctrl_dim, 2*gen_dim)
        #   layer 0 用 Identity 占位（Generator 跳过 i==0，与 layer_film 一致）
        if use_film:
            self.film_projs = nn.ModuleList([
                nn.Linear(ctrl_dim, 2 * gen_dim) if i > 0 else nn.Identity()
                for i in range(gen_layers)])
        # ③ 生成方向投影：x.mean(T) (B, ctrl_dim) → Linear(ctrl_dim, gen_dim) → (B, gen_dim)
        if use_direction:
            self.direction_proj = nn.Linear(ctrl_dim, gen_dim, bias=False)

    def _apply_neutral_inits(self) -> None:
        """零初始化所有控制信号投影 → 中性起步（Controller 不干预 Generator）。

        必须在 _init_weights 通用 N(0,0.02) 之后调用（否则被覆盖），与
        TransformerModel._apply_specialized_inits 同模式。mem_query 保留小随机
        初始化（零初始化会让 S 投影恒为零，无法提取压缩信息；mem_proj 为零已足够
        保证 mem_kv=0 中性）。

        同时重置 GatedDeltaNet 的 alpha_beta_proj 专用初始化（与 TransformerBlock
        中 attn.alpha_beta_proj 同模式：weight=0 + bias 前 _gate_out 维 alpha_init /
        后 _gate_out 维 beta_init），因 _init_weights 通用 N(0,0.02) 会覆盖 __init__
        中的专用设置。
        """
        if self.use_memory_compress:
            nn.init.zeros_(self.mem_proj.weight)
            # mem_query 保留小随机（N(0,0.02) 已由 _init_weights 设好，此处不覆盖）
        if self.use_film:
            for proj in self.film_projs:
                if isinstance(proj, nn.Linear):
                    nn.init.zeros_(proj.weight)
                    nn.init.zeros_(proj.bias)
        if self.use_direction:
            nn.init.zeros_(self.direction_proj.weight)
        # GatedDeltaNet alpha_beta_proj 专用初始化重置（与 TransformerModel.
        # _apply_specialized_inits 中 attn.alpha_beta_proj 同逻辑）
        for mixer in self.mixers:
            if hasattr(mixer, 'alpha_beta_proj'):
                nn.init.zeros_(mixer.alpha_beta_proj.weight)
                _g = getattr(mixer, '_gate_out', None)
                if _g is not None:
                    _bias = torch.empty(2 * _g, dtype=mixer.alpha_beta_proj.bias.dtype,
                                         device=mixer.alpha_beta_proj.bias.device)
                    _bias[:_g].fill_(mixer._alpha_init)
                    _bias[_g:].fill_(mixer._beta_init)
                    with torch.no_grad():
                        mixer.alpha_beta_proj.bias.copy_(_bias)

    def forward(self, token_ids: torch.Tensor,
                past_kv: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor,
                                                       torch.Tensor, torch.Tensor]]]] = None,
                use_cache: bool = False,
                start_pos: int = 0
                ) -> Tuple[ControllerOutput, Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor,
                                                                          torch.Tensor, torch.Tensor]]]]]:
        """Controller 前向：产出 3 类控制信号 + 各层 mixer cache（增量解码用）。

        Args:
            token_ids: (B, T) token id 序列
            past_kv: 各层 mixer 的 past cache 列表 [(k, v, S, z), ...]；None/空表 = 全量前向
            use_cache: True 返回 presents 供增量解码；False 不返回（省内存）
            start_pos: 当前 token 在序列中的位置（RoPE 用；增量解码=已生成 token 数）

        Returns:
            signals: ControllerOutput（mem_kv / film_per_layer / direction）
            presents: 各层 mixer present cache 列表（use_cache=False 时为 None）

        DML 注意：内部强制 use_cache=True 跑 mixer（始终获取 S 矩阵作压缩记忆源），
        仅在外部 use_cache=False 时丢弃 presents。开销可忽略（2 层小模型）。
        """
        B, T = token_ids.shape
        # 共享 embedding：token_ids → (B, T, gen_dim)（与 Generator 同源词表）
        # ctrl_dim != gen_dim 时经 input_proj 投影到 (B, T, ctrl_dim)
        x = self.embedding(token_ids) * math.sqrt(self.gen_dim)
        if self.input_proj is not None:
            x = self.input_proj(x)
        x = self.ln_pre(x)
        presents: Optional[List] = [] if use_cache else None
        last_S: Optional[torch.Tensor] = None
        for i, (ln, mixer) in enumerate(zip(self.ln_layers, self.mixers)):
            xn = ln(x)
            past_i = past_kv[i] if (past_kv is not None and i < len(past_kv)) else None
            # 内部强制 use_cache=True 以获取 S（压缩记忆源）；外部 use_cache 决定是否返回
            h, present = mixer(xn, past_kv=past_i, use_cache=True, start_pos=start_pos)
            x = x + h
            if use_cache and presents is not None:
                presents.append(present)
            if i == self.ctrl_layers - 1:
                # 末层 S: present=(k, v, S, z)，S shape (B, H_ctrl, D_head, D_head)
                last_S = present[2]
        # ---- 产出 3 类控制信号（均零初始化 → 中性起步） ----
        # ① 压缩记忆：S (B, H_ctrl, D_head, D_head) → mem_query (M, H_ctrl, D_head) einsum
        #   → (B, M, D_head) → mem_proj → (B, M, 2*gen_head_dim) → chunk → (mk, mv)
        mem_kv = None
        if self.use_memory_compress and last_S is not None:
            # einsum('mhd,bhde->bme', mem_query, S) → (B, M, D_head)
            # 每个 mem_slot m 对每头 h 用查询向量 q[m,h] 读 S[h] 的 e 维 → 聚合头间到 m 维
            mem = torch.einsum('mhd,bhde->bme', self.mem_query, last_S)
            mem = self.mem_proj(mem)  # (B, M, 2*gen_head_dim)
            mk, mv = mem.chunk(2, dim=-1)  # 各 (B, M, gen_head_dim)
            mem_kv = (mk, mv)
        # ② FiLM per-layer：末层输出 x (B, T, ctrl_dim) → 每层 Linear → (B, T, 2*gen_dim) → chunk
        film_per_layer: Optional[List] = None
        if self.use_film:
            film_per_layer = []
            for i in range(self.gen_layers):
                if i == 0:
                    film_per_layer.append(None)  # Generator 跳过 i==0
                else:
                    gamma_beta = self.film_projs[i](x)  # (B, T, 2*gen_dim)
                    gamma, beta = gamma_beta.chunk(2, dim=-1)  # 各 (B, T, gen_dim)
                    film_per_layer.append((gamma, beta))
        # ③ 生成方向：x.mean(T) (B, ctrl_dim) → Linear → (B, gen_dim)
        direction = None
        if self.use_direction:
            x_mean = x.mean(dim=1)  # (B, ctrl_dim)
            direction = self.direction_proj(x_mean)  # (B, gen_dim)
        signals = ControllerOutput(mem_kv=mem_kv, film_per_layer=film_per_layer,
                                   direction=direction)
        return signals, presents

    @staticmethod
    def convert_legacy_state_dict(state_dict: dict) -> dict:
        """Controller 无历史权重（R42 新模块），直接透传。

        预留接口与 MemoryBank/GatedDeltaNet/SwiGLU 同模式，便于未来字段迁移。
        """
        return state_dict
