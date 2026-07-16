from __future__ import annotations

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.functional import scaled_dot_product_attention
import threading
from typing import Optional, List, Tuple, Any, Dict, Callable


class CharMergeLayer(nn.Module):
    """轻量学习型分词层（Learned Segmentation）。

    输入为字符级序列 (B, T, D)，通过双向门控卷积把相邻字符向量融合成
    "词级表示"。融合门控由模型自己学习，受 LM loss 监督 —— 即"切词/合并词"
    变成可微、可优化的过程，无需静态 BPE 词表。开销仅约注意力的 1~2%。

    设计：
    - depthwise 因果卷积（kernel=3）提取左右邻域聚合；
    - 门控 z = sigmoid(线性) 在"原始字符向量"与"邻域聚合"间插值；
    - 残差连接保证字符信息不丢。
    """

    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        # 因果卷积：左侧+自身（保持自回归，不窥未来）
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=True)
        self.norm = RMSNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        # 邻域聚合：转 (B, D, T) 卷积，因果左侧填充
        x_t = x.transpose(1, 2)
        agg = F.conv1d(x_t, self.conv.weight, None,
                       padding=self.pad, groups=D)
        agg = agg.transpose(1, 2)  # (B, T, D)
        # 门控：当前字符 vs 邻域聚合
        z = torch.sigmoid(self.gate(x))
        out = z * agg + (1 - z) * x
        out = self.norm(out)
        return self.drop(out)


class MemoryBank(nn.Module):
    """可学习压缩记忆（阶段2）。

    维护固定大小记忆槽，存 token 表示的**压缩形式**（压缩矩阵可学），
    并用可学门控选择"保留哪些信息"。记忆作为额外 KV 源供注意力检索，
    全部参数受 LM loss 监督 —— 压缩方法与保留策略都由模型自己优化。

    写入（soft write）：当前表示经可学压缩矩阵压成小向量，按门控 softmax
    对 M 个槽做加权更新（可微，无需硬选择）；读取：记忆槽解压后投影为
    K/V 拼接到注意力 KV 之后，作为全局可检索上下文。
    """

    def __init__(self, dim: int, num_slots: int = 64, comp_dim: int = 32,
                 head_dim: Optional[int] = None, dropout: float = 0.0,
                 retrieval: bool = False, sparse_topk: int = 0):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.comp_dim = comp_dim
        self.head_dim = head_dim or dim
        self.retrieval_enabled = retrieval
        self.sparse_topk = sparse_topk
        # 压缩 / 解压矩阵（可学）：把 D 维表示压到 comp_dim 再还原
        self.compress = nn.Linear(dim, comp_dim, bias=False)
        self.decompress = nn.Linear(comp_dim, dim, bias=False)
        # 写入门控：对当前表示打分，决定写入各槽的权重
        self.write_gate = nn.Linear(dim, num_slots, bias=True)
        self.drop = nn.Dropout(dropout)
        # 记忆槽的 K/V 投影（解压后表示 → 注意力各头 K/V 维度）
        self.mem_k = nn.Linear(dim, self.head_dim, bias=False)
        self.mem_v = nn.Linear(dim, self.head_dim, bias=False)
        # 阶段3 可学习检索门控：单个可学标量缩放记忆召回强度（sigmoid 软增强/抑制），受 LM loss 监督
        if retrieval:
            self.retrieval_gate = nn.Parameter(torch.zeros(1))
        self._init_slots()

    def _init_slots(self):
        # 记忆槽以压缩空间零初始化（forward 首步由 reset 填充）
        self.register_buffer('slots', torch.zeros(1, self.num_slots, self.comp_dim),
                             persistent=False)

    def reset(self, batch: int, device: torch.device, dtype: torch.dtype):
        """新建 batch 大小的记忆（每个样本独立槽）。"""
        self.slots = torch.zeros(batch, self.num_slots, self.comp_dim,
                                 device=device, dtype=dtype)

    def write(self, x: torch.Tensor) -> None:
        """x: (B, T, D) 当前层表示，soft 写入记忆。"""
        B, T, D = x.shape
        if self.slots.shape[0] != B:
            self.reset(B, x.device, x.dtype)
        # 压缩当前表示
        comp = self.compress(x)  # (B, T, comp_dim)
        # 写入权重：对每步表示，softmax 分配到 M 个槽（可学门控）
        gate = torch.softmax(self.write_gate(x), dim=-1)  # (B, T, M)
        # 按 gate 把压缩表示累加到槽（加权求和，可微）
        update = torch.einsum('btm,btc->bmc', gate, comp)  # (B, M, comp_dim)
        # 移动平均式软写入（保留历史记忆，新信息按 gate 权重叠加）
        self.slots = self.slots + update
        # 归一化防止数值膨胀
        self.slots = self.slots / (1e-6 + self.slots.norm(dim=-1, keepdim=True))

    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        """返回记忆的 K/V：(B, M, head_dim) 及检索元信息（门控/稀疏）。"""
        decomp = self.decompress(self.slots)  # (B, M, D)
        k = self.mem_k(decomp)
        v = self.mem_v(decomp)
        meta = None
        if self.retrieval_enabled or self.sparse_topk > 0:
            meta = {
                'retrieval_gate': self.retrieval_gate if self.retrieval_enabled else None,
                'sparse_topk': self.sparse_topk,
                'num_slots': self.num_slots,
            }
        return k, v, meta


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization（LLaMA 风格，比 LayerNorm 更省且更稳定）。"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """旋转位置编码 RoPE：对 Q/K 按位置旋转，天然支持长度外推。
    
    使用实例级缓存（而非模块级全局缓存），避免多线程/多设备冲突。
    提供可选的类级共享缓存（带锁）供性能敏感场景使用。
    """
    # 类级共享缓存（可选，需显式启用）
    _shared_cache: Dict[Tuple[str, str, int], Tuple[torch.Tensor, torch.Tensor]] = {}
    _shared_cache_lock = threading.RLock()
    _use_shared_cache = False

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        # 实例级缓存：隔离不同模型/设备/dtype
        self._cache: Dict[Tuple[str, str, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._cache_lock = threading.RLock()

    @classmethod
    def enable_shared_cache(cls, enabled: bool = True):
        """启用/禁用类级共享缓存（跨实例共享，带锁保护）。"""
        cls._use_shared_cache = enabled
        if not enabled:
            cls._shared_cache.clear()

    def _get_cos_sin(self, start_pos: int, seq_len: int, device: torch.device, dtype: torch.dtype, max_len: int = 2048) -> Tuple[torch.Tensor, torch.Tensor]:
        """按 (device, dtype, head_dim) 缓存整张位置表后按需切片。
        
        实例级缓存避免多线程/多设备污染；可选共享缓存用于性能敏感场景。
        """
        key = (str(device), str(dtype), self.inv_freq.shape[0])
        need = start_pos + seq_len
        
        # 先尝试实例缓存
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None and cached[0].size(2) >= need:
                cos_full, sin_full = cached
                return cos_full[:, :, start_pos:need, :].to(dtype), sin_full[:, :, start_pos:need, :].to(dtype)
        
        # 可选：尝试共享缓存
        if self._use_shared_cache:
            with self._shared_cache_lock:
                cached = self._shared_cache.get(key)
                if cached is not None and cached[0].size(2) >= need:
                    cos_full, sin_full = cached
                    return cos_full[:, :, start_pos:need, :].to(dtype), sin_full[:, :, start_pos:need, :].to(dtype)

        # 计算新缓存
        L = max(need, min(max_len, 4096))
        t = torch.arange(0, L, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_full = emb.cos()[None, None, :, :].to(dtype)
        sin_full = emb.sin()[None, None, :, :].to(dtype)

        # 存入实例缓存
        with self._cache_lock:
            self._cache[key] = (cos_full, sin_full)
        
        # 可选：存入共享缓存
        if self._use_shared_cache:
            with self._shared_cache_lock:
                self._shared_cache[key] = (cos_full, sin_full)

        return cos_full[:, :, start_pos:need, :].to(dtype), sin_full[:, :, start_pos:need, :].to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0, max_len: int = 2048) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._get_cos_sin(start_pos, q.size(2), q.device, q.dtype, max_len=max_len)
        return self._rope_apply(q, cos, sin), self._rope_apply(k, cos, sin)

    @staticmethod
    def _rope_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        d = x.size(-1) // 2
        x1, x2 = x[..., :d], x[..., d:]
        rot = torch.cat((-x2, x1), dim=-1)
        return x * cos.to(x.dtype) + rot * sin.to(x.dtype)

    def clear_cache(self):
        """清空实例缓存（长时间运行时可调用防止内存泄漏）。"""
        with self._cache_lock:
            self._cache.clear()


class SlidingWindowCausalSelfAttention(nn.Module):
    """因果自注意力，可选滑动窗口 + 可学习相对位置偏置。
     CUDA/CPU 用原生 fused SDPA；AMD DirectML 的 fused 内核会触发原生崩溃，
     故 DML(及其他后端)走手动 matmul+softmax+因果掩码 以规避该 bug。
    """
    def __init__(self, dim: int, num_heads: int, window: int = 0, rel_bias: bool = False, max_seq_length: int = 64,
                 qk_norm: bool = True, attn_temp: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window = int(window or 0)
        self.rel_bias = rel_bias
        self.max_seq_length = max_seq_length
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        if self.rel_bias:
            # T5 风格相对位置偏置表：(heads, 2T-1)
            self.rel_bias_table = nn.Parameter(torch.zeros(num_heads, 2 * max_seq_length - 1))
        # ① QK-Norm：对 Q/K 各自做 RMSNorm 后再进注意力，与 RoPE 互补、稳定训练（默认开）
        self.qk_norm_enabled = qk_norm
        if qk_norm:
            self.qk_norm = RMSNorm(self.head_dim)
        # ⑤ 可学习注意力温度：softmax(score / T)，T=exp(log_temp) 恒正（默认开）
        self.attn_temp_enabled = attn_temp
        if attn_temp:
            self.log_temp = nn.Parameter(torch.zeros(1))
        # 运行时增强开关（按开关粒度，用于“交替/分段增强”训练）：默认全开
        self._rt: Dict[str, bool] = {"qk_norm": True, "attn_temp": True}
        self._cached_T = -1
        self._mask: Optional[torch.Tensor] = None
        self._rbias: Optional[torch.Tensor] = None

    def _build_masks(self, T: int, device: torch.device):
        # Check if we need to rebuild: length changed OR device changed
        if self._cached_T == T and self._mask is not None:
            if self._mask.device == device:
                return
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
        if self.window > 0:
            dist = torch.arange(T, device=device).unsqueeze(1) - torch.arange(T, device=device).unsqueeze(0)
            window_mask = dist > self.window
            mask = causal | window_mask
        else:
            mask = causal
        self._mask = mask  # True = 禁止
        if self.rel_bias:
            idx = torch.arange(T, device=device).unsqueeze(1) - torch.arange(T, device=device).unsqueeze(0)
            idx = (idx + T - 1).clamp(0, 2 * self.max_seq_length - 1)
            self._rbias = self.rel_bias_table[:, idx]  # (H, T, T)
        self._cached_T = T

    def forward(self, x: torch.Tensor, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False, start_pos: int = 0,
                memory_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        q, k, v = self.project_and_norm(x, start_pos)
        return self.attend(q, k, v, past_kv, use_cache, start_pos, memory_kv)

    def set_enhancements_active(self, spec):
        """运行时开关（按开关粒度）：`spec=True/False` 全开/全关；`spec=dict` 仅更新存在的键。
        用于“交替/分段增强”训练，关闭时跳过对应 QK-Norm/可学习温度（恒等）。"""
        if isinstance(spec, bool):
            on = spec
            self._rt = {"qk_norm": on, "attn_temp": on}
        elif isinstance(spec, dict):
            for k, v in spec.items():
                if k in self._rt:
                    self._rt[k] = bool(v)
        else:
            raise TypeError(f"set_enhancements_active 期望 bool 或 dict，收到 {type(spec)}")

    def project_and_norm(self, x: torch.Tensor, start_pos: int = 0):
        """廉价部分（在梯度检查点重算区域之外执行，避免被反向重算放大）：
        QKV 投影 + ①QK-Norm + ⑤可学习温度 + RoPE。返回已归一化/旋转后的 (q, k, v)。"""
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # ① QK-Norm：投影后、RoPE 前对 Q/K 各自归一化（运行时开关 _rt 可跳过）
        if self.qk_norm_enabled and self._rt["qk_norm"]:
            q = self.qk_norm(q)
            k = self.qk_norm(k)
        # ⑤ 可学习温度：温度恒正（T=exp(log_temp)），直接缩放 Q/K 幅值（等价 softmax(score/T)）。
        #    融合为单次标量乘法 q*=exp(-0.5*log_temp)，免去额外 sqrt 算子。
        if self.attn_temp_enabled and self._rt["attn_temp"]:
            scale = torch.exp(-0.5 * self.log_temp)
            q = q * scale
            k = k * scale
        q, k = self.rope(q, k, start_pos=start_pos, max_len=self.max_seq_length)
        return q, k, v

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False,
                start_pos: int = 0,
                memory_kv: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """重算力部分（在梯度检查点重算区域内执行）：scores/softmax/proj。
        大幅激活（scores 张量）不落盘、反向时重算，保留大模型显存收益。
        memory_kv: (mk, mv, meta) 可学习压缩记忆的 K/V + 检索元信息（门控/稀疏）。"""
        B, _, Tq, _ = q.shape
        # 阶段3 可学习检索：先计算记忆段偏置（门控缩放 + top-k 稀疏），注入后续 attn_mask
        mem_bias: Optional[torch.Tensor] = None
        mem_cols = 0
        if memory_kv is not None and memory_kv[2] is not None:
            mk, mv, meta = memory_kv
            mem_cols = mk.size(1)
            # 记忆查询相似度（每槽点积）：(B,H,Tq,M)，廉价（M 小）
            mlogits = torch.einsum('bhqd,bhmd->bhqm', q,
                                   mk.unsqueeze(1).expand(-1, self.num_heads, -1, -1))
            if meta.get('retrieval_gate') is not None:
                # 可学门控：sigmoid → (0,1) 软增强/抑制记忆召回，受 LM loss 监督
                gate = torch.sigmoid(meta['retrieval_gate']).view(1, 1, 1, 1)
                mlogits = mlogits * gate
            if meta.get('sparse_topk', 0) and meta['sparse_topk'] < mem_cols:
                k_keep = meta['sparse_topk']
                # 每查询保留 top-k 记忆槽，余下压到 -inf（可微稀疏，降低无关记忆干扰）
                thr = torch.kthvalue(mlogits, mem_cols - k_keep + 1, dim=-1).values  # (B,H,Tq)
                drop = mlogits < thr.unsqueeze(-1)
                mlogits = mlogits.masked_fill(drop, -1e9)
            mem_bias = mlogits  # (B,H,Tq,M)，作为 scores 的可加偏置

        if use_cache:
            # 增量解码：拼接待拼接的 K/V 缓存，仅对当前 token 做注意力
            if past_kv is not None:
                pk, pv = past_kv
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)
            if memory_kv is not None:
                mk, mv = memory_kv[0], memory_kv[1]
                # (B,M,D) -> (B,H,M,D) 各头共享记忆 KV
                mk = mk.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
                mv = mv.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
                k = torch.cat([mk, k], dim=2)
                v = torch.cat([mv, v], dim=2)
            present = (k, v)
            Tkv = k.size(2)
            qpos = torch.arange(start_pos, start_pos + Tq, device=q.device).unsqueeze(1)
            kpos = torch.arange(0, Tkv, device=q.device).unsqueeze(0)
            causal_mask = kpos > qpos
            if self.window > 0:
                causal_mask = causal_mask | (qpos - kpos > self.window)
            attn_mask = (causal_mask.float() * -1e9).unsqueeze(0)   # (1,1,T,Tkv)
            # 记忆槽位置在窗口 KV 之前（seq 起点之前），永远不被因果遮蔽，
            # 但也不参与"未来"泄露：记忆是历史压缩，视为已发生，不施加 causal 惩罚
            if self.rel_bias:
                idx = (qpos - kpos + Tkv - 1).clamp(0, 2 * self.max_seq_length - 1)
                attn_mask = attn_mask + self.rel_bias_table[:, idx].unsqueeze(0)
            if mem_bias is not None:
                # mem_bias 仅覆盖记忆段前 mem_cols 列，需扩到全宽 Tkv 再注入
                full_bias = torch.zeros(B, self.num_heads, Tq, Tkv, device=q.device)
                full_bias[..., :mem_cols] = mem_bias
                attn_mask = attn_mask + full_bias  # 注入检索门控 + 稀疏
            if q.device.type == 'cuda':
                out = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            else:
                out = self._manual_attention(q, k, v, attn_mask)
            out = out.transpose(1, 2).reshape(B, Tq, self.num_heads * self.head_dim)
            return self.proj(out), present

        # —— 非缓存（训练 / 含 SSM 模型全量重算）路径 ——
        T = q.size(2)
        self._build_masks(T, q.device)
        Tkv = k.size(2)
        # 训练路径把记忆拼到 K/V 之前（记忆在前，窗口/全量在后）
        if memory_kv is not None:
            mk, mv = memory_kv[0], memory_kv[1]
            # (B,M,D) -> (B,H,M,D) 各头共享记忆 KV，拼到 K/V 序列维度之前
            mk = mk.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            mv = mv.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            k = torch.cat([mk, k], dim=2)
            v = torch.cat([mv, v], dim=2)
            Tkv = k.size(2)
        # 统一构造 (1,1,T,Tkv) 注意力掩码：记忆段全 0（全局可检索），
        # 主序列段按 causal / window / rel_bias 遮蔽
        attn_mask = torch.zeros(1, 1, T, Tkv, device=q.device)
        if self.window > 0:
            qpos = torch.arange(0, T, device=q.device).unsqueeze(1)
            kpos = torch.arange(0, Tkv, device=q.device).unsqueeze(0)
            m = (qpos - kpos > self.window).float() * -1e9  # (T, Tkv)
            m = m.unsqueeze(0).unsqueeze(0)  # (1,1,T,Tkv)
            if memory_kv is not None:
                m[:, :, :, :mem_cols] = 0.0  # 记忆段不受窗口限制
            attn_mask = attn_mask + m
        elif self.rel_bias:
            attn_mask = attn_mask + (self._mask.float() * -1e9)
        if self.rel_bias:
            idx = (torch.arange(T, device=q.device).unsqueeze(1)
                   - torch.arange(Tkv, device=q.device).unsqueeze(0)
                   + Tkv - 1).clamp(0, 2 * self.max_seq_length - 1)
            attn_mask = attn_mask + self.rel_bias_table[:, idx].unsqueeze(0)
        if mem_bias is not None:
            # mem_bias 仅覆盖记忆段前 mem_cols 列，需扩到全宽 Tkv 再注入
            full_bias = torch.zeros(B, self.num_heads, T, Tkv, device=q.device)
            full_bias[..., :mem_cols] = mem_bias
            attn_mask = attn_mask + full_bias  # 注入检索门控 + 稀疏（前 mem_cols 列）
        if q.device.type in ('cuda', 'cpu'):
            if attn_mask.abs().max() > 0 or self.rel_bias:
                out = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            else:
                out = scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            extra = attn_mask if attn_mask.abs().max() > 0 else None
            if self.rel_bias and extra is None:
                extra = self._rbias.unsqueeze(0)
            out = self._manual_attention(q, k, v, extra)
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.proj(out), None

    def _manual_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # q,k,v: (B, H, Tq, D)；attn_mask: (1,1,Tq,Tkv) 或 None(纯因果)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale   # (B, H, Tq, Tkv)
        if attn_mask is not None:
            scores = scores + attn_mask
        else:
            Tq, Tk = q.size(2), k.size(2)
            causal = torch.triu(torch.ones(Tq, Tk, dtype=torch.bool, device=q.device), diagonal=1)
            scores = scores.masked_fill(causal, -1e9)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)                             # (B, H, Tq, D)


class MambaSSM(nn.Module):
    """Mamba-like 选择性状态空间模型（线性复杂度长序列建模）。
      门控 + 输入依赖的 Δ/B/C，零阶保持离散化后沿时间递推。
      选择性扫描已向量化（并行前缀扫描，log2(L) 步），消除逐时间步 Python for 循环：
      既显著加快 CPU 训练，也避免低功耗 iGPU 上 DML 因单次步内 kernel 过多触发 TDR 设备重置。
      支持增量推理：可传入 past_state (h_{t-1}) 并返回 present_state (h_t)。
      支持增量卷积状态：维护最后 conv_kernel-1 个输入用于因果卷积。
    """
    def __init__(self, dim: int, d_state: int = 16, d_inner_factor: int = 1, dt_rank: Optional[int] = None, conv_kernel: int = 3,
                 dt_proj_bias_init: float = 0.1, a_log_init_range: Tuple[float, float] = (-1.0, 1.0), D_init: float = 1.0):
        super().__init__()
        d_inner = dim * d_inner_factor
        dt_rank = dt_rank or max(1, math.ceil(dim / 16))
        self.dim = dim
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.conv_kernel = conv_kernel
        self.dt_proj_bias_init = dt_proj_bias_init
        self.a_log_init_range = a_log_init_range
        self.D_init = D_init
        self.norm = RMSNorm(dim)
        self.in_proj = nn.Linear(dim, 2 * d_inner, bias=False)
        # 因果卷积：左填充 conv_kernel-1 个零，输出取前 L 个位置（增量时取最后 1 个），
        # 保证位置 t 仅依赖 x[<=t]，避免居中窗口泄露未来 token
        self.conv = nn.Conv1d(d_inner, d_inner, kernel_size=conv_kernel,
                              padding=conv_kernel - 1, groups=d_inner, bias=False)
        self.act = nn.SiLU()
        # 从 conv 输出投影出 Δ 输入、B、C（选择性）
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        nn.init.constant_(self.dt_proj.bias, dt_proj_bias_init)
        # A 以对数形式存储，保证 A = -exp(A_log) 为负且稳定
        self.A_log = nn.Parameter(torch.empty(d_inner, d_state))
        nn.init.uniform_(self.A_log, a_log_init_range[0], a_log_init_range[1])
        self.D = nn.Parameter(torch.ones(d_inner) * D_init)   # 跳跃连接
        self.out_proj = nn.Linear(d_inner, dim, bias=False)
        self.proper_init()

    def proper_init(self):
        """SSM 专用初始化（避免被 TransformerModel._init_weights 的通用初始化覆盖）：
         - in/out/x_proj/dt_proj 权重用 Xavier（B/C 投影更稳）
         - dt_proj 偏置置 0.1（遗忘偏置，缓解早期不稳定）
         - A_log 用 uniform(a_log_init_range) -> A=-exp(A_log) 为负且稳定（Mamba 风格）
         - D 跳跃连接置 D_init
        """
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.xavier_uniform_(self.dt_proj.weight)
        nn.init.constant_(self.dt_proj.bias, 0.1)
        nn.init.uniform_(self.A_log, *self.a_log_init_range)
        nn.init.ones_(self.D)

    def forward(self, x: torch.Tensor, past_state: Optional[torch.Tensor] = None, past_conv_state: Optional[torch.Tensor] = None, use_cache: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x: (B, L, D) input tensor
            past_state: (B, d_inner, d_state) previous hidden state h_{t-1}
            past_conv_state: (B, d_inner, conv_kernel-1) previous conv inputs
            use_cache: whether to return present_state and present_conv_state for incremental decoding
        Returns:
            y: (B, L, D) output tensor
            present_state: (B, d_inner, d_state) if use_cache=True
            present_conv_state: (B, d_inner, conv_kernel-1) if use_cache=True
        """
        B, L, _ = x.shape
        x = self.norm(x)
        xz = self.in_proj(x)                          # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)
        
        # 因果卷积：conv 用左填充（padding=conv_kernel-1），输出取前 L 个位置（增量时取最后 1 个），
        # 确保仅依赖当前及历史 token，不泄露未来
        if past_conv_state is not None and past_conv_state.shape[0] == B and L == 1:
            # 增量解码：拼接历史 conv 窗口与当前 token，卷积后取最后一个位置即当前 token 特征
            conv_input = torch.cat([past_conv_state, x_in.transpose(1, 2)], dim=-1)  # (B, d_inner, conv_kernel)
            # 因果卷积后取索引 conv_kernel-1 的位置，即窗口 [past0, past1, current]（当前 token 特征）
            x_conv = self.conv(conv_input)[:, :, self.conv_kernel - 1].unsqueeze(1)  # (B, 1, d_inner)
            present_conv_state = conv_input[:, :, -(self.conv_kernel - 1):]  # (B, d_inner, conv_kernel-1)
        else:
            # 全量序列或 prefill：因果卷积后截断到前 L 个位置
            x_conv = self.conv(x_in.transpose(1, 2)).transpose(1, 2)[:, :L, :]  # (B, L, d_inner)
            if use_cache and L > 0:
                # 保存最后 conv_kernel-1 个 token 供下一步增量使用
                present_conv_state = x_in.transpose(1, 2)[:, :, -(self.conv_kernel - 1):]
            else:
                present_conv_state = None
        
        x_conv = self.act(x_conv)                    # (B, L, d_inner)
        ssm = self.x_proj(x_conv)                    # (B, L, dt_rank + 2*d_state)
        dt_in, Bp, Cp = ssm.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = torch.nn.functional.softplus(self.dt_proj(dt_in))   # (B, L, d_inner)
        A = -torch.exp(self.A_log)                   # (d_inner, d_state)
        dA = torch.exp(dt.unsqueeze(-1) * A)          # (B, L, d_inner, d_state)
        dB = dt.unsqueeze(-1) * Bp.unsqueeze(2)       # (B, L, d_inner, d_state)
        xb = dB * x_conv.unsqueeze(-1)               # (B, L, d_inner, d_state)
        C = Cp                                        # (B, L, d_state)
        
        if past_state is not None and L == 1:
            # 增量解码：逐 token 递推，并返回当前步 conv 状态供下一步拼接
            return self._forward_step(x, z, x_conv, dA, xb, C, past_state, use_cache, present_conv_state)
        
        # 全量序列处理（训练或 prefill）
        h = self._selective_scan(dA, xb, past_state)  # (B, L, d_inner, d_state)
        y = (h * C.unsqueeze(2)).sum(-1)             # (B, L, d_inner)
        y = y + self.D * x_conv                      # 跳跃连接
        y = y * self.act(z)                          # 门控
        y = self.out_proj(y)
        
        if use_cache:
            # 返回最后隐藏状态作为 present_state
            present_state = h[:, -1, :, :]  # (B, d_inner, d_state)
            return y, present_state, present_conv_state
        return y, None, present_conv_state

    def _forward_step(self, x: torch.Tensor, z: torch.Tensor, x_conv: torch.Tensor, dA: torch.Tensor, xb: torch.Tensor, C: torch.Tensor, past_state: torch.Tensor, use_cache: bool, present_conv_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Process a single token incrementally."""
        B = x.shape[0]
        # past_state: (B, d_inner, d_state)
        h_t = dA[:, 0] * past_state + xb[:, 0]  # (B, d_inner, d_state)
        y_t = (h_t * C[:, 0].unsqueeze(1)).sum(-1)  # (B, d_inner)
        y_t = y_t + self.D * x_conv[:, 0]  # skip connection（x_conv 为 (B,1,d_inner)，取位置 0 即当前 token）
        y_t = y_t * self.act(z[:, 0])  # gate
        y_t = self.out_proj(y_t).unsqueeze(1)  # (B, 1, dim)
        
        if use_cache:
            # 返回当前步的 conv 状态，供下一步增量解码拼接（不再丢弃）
            return y_t, h_t, present_conv_state
        return y_t, None, None

    def _selective_scan(self, a: torch.Tensor, b: torch.Tensor, past_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """并行前缀扫描计算 h_t = a_t * h_{t-1} + b_t（h_0=0 或 past_state）。

        a, b: (B, L, d_inner, d_state)。返回 h: (B, L, d_inner, d_state)。
        半群 (A, B)⊙(A', B') = (A·A', A'·B + B') 满足结合律；
        Hillis-Steele 含扫描：每轮把左邻 2^k 步的变换合并进来，offset 从 1 翻倍到 <L。
        单位元为 (A=1, B=0)，越界位置用单位元填充。
        
        如果提供 past_state (B, d_inner, d_state)，将其作为 h_{-1} 用于计算 h_0 = a_0 * past_state + b_0。
        """
        L = a.shape[1]
        # Use original a for prefix product calculation
        orig_A = a.clone()
        A = a.clone()
        B = b.clone()
        
        # Standard parallel prefix scan (Hillis-Steele) assuming h_0 = 0
        offset = 1
        while offset < L:
            # 左移 offset：位置 i 取 i-offset（越界填单位元 A=1, B=0）
            A_prev = torch.cat([torch.ones_like(A[:, :offset]), A[:, :-offset]], dim=1)
            B_prev = torch.cat([torch.zeros_like(B[:, :offset]), B[:, :-offset]], dim=1)
            A_new = A_prev * A
            B_new = A * B_prev + B
            A, B = A_new, B_new
            offset <<= 1
        
        # If we have past_state, we need to incorporate it
        # The scan result B assumes h_0 = b_0 (since h_{-1}=0)
        # With past_state: h_0 = a_0 * past_state + b_0
        # h_t = a_t * ... * a_0 * past_state + (scan result)
        if past_state is not None:
            # Compute prefix products of original a: prefix_A[t] = a_t * a_{t-1} * ... * a_0
            # Need to use original a, not the modified A from scan
            orig_A = a.clone()
            prefix_A = orig_A.clone()
            offset = 1
            while offset < L:
                # 左移 offset：位置 i 取 i-offset（越界填单位元 A=1）
                prev = torch.cat([torch.ones_like(prefix_A[:, :offset]), prefix_A[:, :-offset]], dim=1)
                prefix_A = prefix_A * prev
                offset <<= 1
            # prefix_A[:, t] = a_t * a_{t-1} * ... * a_0
            past_expanded = past_state.unsqueeze(1).expand(-1, L, -1, -1)
            B = B + prefix_A * past_expanded
        
        return B


class SwiGLU(nn.Module):
    """SwiGLU 前馈（LLaMA 风格门控 FFN，比 GELU MLP 更有表达力）。"""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """可配置混合块：attn / ssm / hybrid(attn+ssm 并行)。Pre-LN。"""
    def __init__(self, dim: int, num_heads: int, hidden_dim: int, block_type: str = 'attn',
                 dropout: float = 0.0, max_seq_length: int = 64,
                 ssm_kwargs: Optional[Dict[str, Any]] = None, attn_kwargs: Optional[Dict[str, Any]] = None,
                 residual_gate: bool = True, hybrid_gate: bool = True, gradient_checkpointing: bool = True):
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
            self.attn = SlidingWindowCausalSelfAttention(
                dim, num_heads, max_seq_length=max_seq_length, **attn_kwargs)
        if block_type in ('ssm', 'hybrid'):
            self.ssm = MambaSSM(dim, **ssm_kwargs)
        self.ln2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden_dim)
        # ②/⑥ 每层可学习残差门控：x = x + gate * f(x)（init 1.0，默认行为不变）
        if residual_gate:
            self.sub1_gate = nn.Parameter(torch.ones(1))   # 第一子层残差（attn 或 ssm）
            self.ffn_gate = nn.Parameter(torch.ones(1))    # FFN 子层残差
        # ⭐A 混合块内 attn/ssm 两路可学习门控（init 1.0）
        if hybrid_gate and block_type == 'hybrid':
            self.hybrid_attn_gate = nn.Parameter(torch.ones(1))
            self.hybrid_ssm_gate = nn.Parameter(torch.ones(1))

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
        gate1 = (self.sub1_gate if (self.residual_gate_enabled and self._rt["residual_gate"]) else None)
        gate2 = (self.ffn_gate if (self.residual_gate_enabled and self._rt["residual_gate"]) else None)

        if self.block_type == 'attn':
            if ckpt:
                q, k, v = self.attn.project_and_norm(self.ln1(x), start_pos)
                h, present = checkpoint(self.attn.attend, q, k, v, attn_past_kv, use_cache, start_pos, mem_kv, use_reentrant=False)
            else:
                h, present = self.attn(self.ln1(x), attn_past_kv, use_cache, start_pos, memory_kv=mem_kv)
            x = x + self.drop(gate1 * h if gate1 is not None else h)
        elif self.block_type == 'ssm':
            if ckpt:
                h, ssm_present_state, ssm_present_conv_state = checkpoint(self.ssm, self.ln1(x), ssm_past_state, ssm_past_conv_state, use_cache, use_reentrant=False)
            else:
                h, ssm_present_state, ssm_present_conv_state = self.ssm(self.ln1(x), past_state=ssm_past_state, past_conv_state=ssm_past_conv_state, use_cache=use_cache)
            x = x + self.drop(gate1 * h if gate1 is not None else h)
        elif self.block_type == 'hybrid':
            xn = self.ln1(x)
            if ckpt:
                q, k, v = self.attn.project_and_norm(xn, start_pos)
                h, attn_present = checkpoint(self.attn.attend, q, k, v, attn_past_kv, use_cache, start_pos, mem_kv, use_reentrant=False)
                ssm_h, ssm_present_state, ssm_present_conv_state = checkpoint(self.ssm, xn, ssm_past_state, ssm_past_conv_state, use_cache, use_reentrant=False)
            else:
                h, attn_present = self.attn(xn, attn_past_kv, use_cache, start_pos, memory_kv=mem_kv)
                ssm_h, ssm_present_state, ssm_present_conv_state = self.ssm(xn, past_state=ssm_past_state, past_conv_state=ssm_past_conv_state, use_cache=use_cache)
            if self.hybrid_gate_enabled and self._rt["hybrid_gate"]:
                # ⭐A 混合块：attn 与 ssm 两路各自可学习门控，让模型自决每层偏重
                x = x + self.drop(self.hybrid_attn_gate * h) + self.drop(self.hybrid_ssm_gate * ssm_h)
            else:
                x = x + self.drop(h) + self.drop(ssm_h)
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

    def __init__(self, vocab_size: int, embedding_dim: int, num_heads: int, num_layers: int,
                 hidden_dim: int, max_seq_length: int, dropout: float = 0.0, tie_weights: bool = True,
                 gradient_checkpointing: bool = True,
                 layer_plan: Optional[List[str] | str] = None,
                 ssm_d_state: int = 16, ssm_d_inner_factor: int = 1, ssm_dt_rank: Optional[int] = None,
                 ssm_conv_kernel: int = 3, ssm_dt_proj_bias_init: float = 0.1,
                 ssm_a_log_init_range: List[float] = [-1, 1],
                 ssm_D_init: float = 1.0,
                 attn_window: int = 0, attn_rel_bias: bool = False,
                 rope_base: float = 10000.0, rope_max_len: int = 4096,
                 mask_fill_value: float = -1e9,
                  qk_norm: bool = True, attn_temp: bool = True,
                   residual_gate: bool = True, hybrid_gate: bool = True,
                   char_merge: bool = False, char_merge_kernel: int = 3,
                   char_merge_dropout: float = 0.0,
                   memory_size: int = 0, memory_comp_dim: int = 32,
                   memory_retrieval: bool = False, memory_sparse_topk: int = 0):
        super(TransformerModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.gradient_checkpointing = gradient_checkpointing
        self.layer_plan = _parse_layer_plan(layer_plan, num_layers)
        self.rope_base = rope_base
        self.rope_max_len = rope_max_len
        self.mask_fill_value = mask_fill_value
        # 阶段2：可学习压缩记忆（memory_size>0 时启用），存压缩表示 + 可学门控选槽
        self.memory_enabled = memory_size > 0
        self.memory_size = memory_size
        self.memory_comp_dim = memory_comp_dim
        self.memory_retrieval = memory_retrieval
        self.memory_sparse_topk = memory_sparse_topk

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
                           qk_norm=qk_norm, attn_temp=attn_temp)
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, hidden_dim, block_type=bt,
                             dropout=dropout, max_seq_length=max_seq_length,
                             ssm_kwargs=ssm_kwargs, attn_kwargs=attn_kwargs,
                             residual_gate=residual_gate, hybrid_gate=hybrid_gate,
                             gradient_checkpointing=gradient_checkpointing)
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
                retrieval=memory_retrieval, sparse_topk=memory_sparse_topk)

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

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
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

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, past_key_values: Optional[List[Optional[Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]]]] = None, use_cache: bool = False) -> Tuple[torch.Tensor, Optional[List[Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]]]]:
        # src: (batch, seq_len)；RoPE 在注意力内部按位置旋转，无需外部 PE
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

        # 可学习压缩记忆：每个样本独立槽，前向首步重置（训练/推理皆按 batch 重建）
        memory = None
        if self.memory_enabled:
            memory = self.memory_bank
            memory.reset(x.size(0), x.device, x.dtype)

        for i, block in enumerate(self.blocks):
            ssm_past_state = ssm_states[i] if use_cache else None
            ssm_past_conv_state = ssm_conv_states[i] if use_cache else None
            # 检查点（仅重算力部分）已在 block 内部按 self.gradient_checkpointing 处理，此处直接调用
            x, present = block(x, past_key_values[i], use_cache, start_pos, ssm_past_state, ssm_past_conv_state, memory)
            presents.append(present)
        x = self.ln_f(x)
        if use_cache:
            return self.output_head(x), presents
        return self.output_head(x)

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
        generated = list(token_ids)
        max_seq_length = self.max_seq_length
        eos_token_id = eos_id
        pad_token_id = pad_id
        sep_token_id = sep_id
        # 现在支持混合架构的增量解码（SSM 也有增量状态）
        use_cache = True

        def sample_step(logits_t: torch.Tensor) -> Optional[int]:
            next_token_logits = logits_t / temperature
            for prev_token in set(generated):
                if 0 <= prev_token < next_token_logits.shape[0]:
                    # 符号感知的重复惩罚：正值除、负值乘（与 HF 一致），
                    # 避免负 logit 被“除”后反而更可能被采样
                    lt = next_token_logits[prev_token]
                    if lt > 0:
                        next_token_logits[prev_token] = lt / repetition_penalty
                    else:
                        next_token_logits[prev_token] = lt * repetition_penalty
            if ngram_fn is not None and ngram_weight != 0.0:
                next_token_logits = next_token_logits + ngram_weight * ngram_fn(generated, device)
            next_token_logits[pad_token_id] = float('-inf')
            next_token_logits[sep_token_id] = float('-inf')
            # min_length: 最小生成长度（不含 prompt），默认 3
            if len(generated) - len(token_ids) < min_length:
                next_token_logits[eos_token_id] = float('-inf')
            else:
                # eos_penalty: EOS 惩罚值，默认 -5.0（抑制 EOS）
                next_token_logits[eos_token_id] = next_token_logits[eos_token_id] + eos_penalty
            # top_k: <=0 禁用，>=vocab_size 视为全词表
            vocab_size = next_token_logits.shape[0]
            if top_k > 0 and top_k < vocab_size:
                top_k_vals = torch.topk(next_token_logits, min(top_k, vocab_size))[0]
                threshold = top_k_vals[..., -1]
                next_token_logits[next_token_logits < threshold] = float('-inf')
            if torch.isinf(next_token_logits).all():
                next_token_logits = logits_t / temperature
                next_token_logits[pad_token_id] = float('-inf')
            probs = torch.softmax(next_token_logits, dim=-1)
            if probs.max() < 0.01:
                return None
            return torch.multinomial(probs, num_samples=1).item()

        with torch.no_grad():
            past = None
            cur_pos = 0
            if use_cache:
                input_ids = torch.tensor([generated], dtype=torch.long, device=device)
                logits, past = self.forward(input_ids, past_key_values=None, use_cache=True)
                cur_pos = input_ids.size(1)
            else:
                input_ids = torch.tensor([generated], dtype=torch.long, device=device)
                logits = self.forward(input_ids)

            for _ in range(max_length):
                if cur_pos >= max_seq_length:
                    break
                next_token = sample_step(logits[0, -1, :])
                if next_token is None:
                    break
                generated.append(next_token)
                if next_token == eos_token_id and len(generated) - len(token_ids) >= min_length:
                    break
                if use_cache:
                    input_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
                    logits, past = self.forward(input_ids, past_key_values=past,
                                                use_cache=True)
                    cur_pos += 1
                else:
                    ctx = generated[-max_seq_length:] if len(generated) > max_seq_length else generated
                    input_ids = torch.tensor([ctx], dtype=torch.long, device=device)
                    logits = self.forward(input_ids)
        return generated