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

    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.0,
                 gate_bias_init: float = -1.0):
        super().__init__()
        self.dim = dim
        # 因果卷积：仅左侧填充（kernel-1 个零），位置 t 只看 t-(k-1)..t，不窥未来
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=True)
        # 门控偏置初始化：bias=-1 → sigmoid(-1)≈0.27，初期偏向字符表示（少融合），
        # 避免早期训练方差过大；随训练推进 gate 自适应调整融合比例。
        nn.init.constant_(self.gate.bias, gate_bias_init)
        self.norm = RMSNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        # 邻域聚合：因果左侧填充（仅 pad 左边），卷积后取前 T 个位置
        x_t = x.transpose(1, 2)  # (B, D, T)
        # 左侧零填充 pad 个位置（因果：位置 t 只看 t-pad..t）
        x_padded = F.pad(x_t, (self.pad, 0))  # (B, D, T+pad)
        agg = F.conv1d(x_padded, self.conv.weight, None, groups=D)  # (B, D, T)
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
                 retrieval: bool = False, sparse_topk: int = 0,
                 forget: bool = False, product_key: bool = False):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.comp_dim = comp_dim
        self.head_dim = head_dim or dim
        self.retrieval_enabled = retrieval
        self.sparse_topk = sparse_topk
        self.forget_enabled = forget
        # 阶段8.3：product-key 写路由——写入时按"新内容与各槽当前内容的相似度"分配
        # （而非纯位置相关的 write_gate），让"写什么到哪"由内容相似度驱动（可微、无硬选择）。
        # 与读路径的 query-槽相似度（attend 内 mlogits）天然对称：写用内容路由、读用查询路由。
        # 默认关（向后兼容旧权重：无此标志则不启用相似度路由）。
        self.product_key = product_key
        # 压缩 / 解压矩阵（可学）：把 D 维表示压到 comp_dim 再还原
        self.compress = nn.Linear(dim, comp_dim, bias=False)
        self.decompress = nn.Linear(comp_dim, dim, bias=False)
        # 写入门控：对当前表示打分，决定写入各槽的权重
        self.write_gate = nn.Linear(dim, num_slots, bias=True)
        # 阶段8.9：可学习遗忘门控升级为 per-slot——每个槽独立衰减率，
        # 模型自决"哪些槽保留历史、哪些槽快速更新"（如高频槽快忘、关键槽长存）。
        # 原标量 forget_gate 无法区分槽间差异，所有槽统一衰减。
        if forget:
            self.forget_gate = nn.Parameter(torch.zeros(num_slots))  # (M,)
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
        """新建 batch 大小的记忆（每个样本独立槽）。

        统一对齐到本模块权重所在设备（DML 的 privateuseone vs privateuseone:0
        别名不一致会导致后续 .to() 每步产生大量设备拷贝，故在此一次对齐到位，
        get_kv/write 热路径不再做 .to，消除拷贝开销。
        """
        dev = self.compress.weight.device
        self.slots = torch.zeros(batch, self.num_slots, self.comp_dim,
                                 device=dev, dtype=dtype)

    def write(self, x: torch.Tensor) -> None:
        """x: (B, T, D) 当前层表示，soft 写入记忆。"""
        B, T, D = x.shape
        # 仅在 batch 变化或设备确实不一致时重建（正常训练每步同设备，不触发拷贝）
        if self.slots.shape[0] != B or self.slots.device != x.device:
            if self.slots.device != x.device:
                x = x.to(self.slots.device)
            self.reset(B, self.slots.device, x.dtype)
        # 可学习遗忘（per-slot）：先按 forget_gate sigmoid→(0,1)^M 衰减各槽旧记忆，再叠加新信息。
        # 运行时开关 _forget_active=False 时跳过（恒等保留，向后兼容）。
        if self.forget_enabled and getattr(self, '_forget_active', True):
            f = torch.sigmoid(self.forget_gate)
            self.slots = f * self.slots
        # 压缩当前表示
        comp = self.compress(x)  # (B, T, comp_dim)
        # 写入权重：对每步表示，softmax 分配到 M 个槽。
        if self.product_key:
            # 阶段8.3：按内容相似度路由——新内容与各槽现有内容越相似，越写到该槽
            # （product-key 风格）。sim = comp·slots，softmax 后分配；可微、与读路径对称。
            # 注意：sim 依赖当前 slots，而 slots 会随写入更新 → 必须逐 token 顺序写，
            # 否则全量路径（一次算齐所有 token 的 gate）与增量 cache 路径（逐 token 更新
            # slots 后再算下一 token 的 gate）结果不一致（曾在 cache parity 测试暴露
            # max_diff=0.028）。顺序写让两条路径逐 token 等价。
            # 阶段8.8 优化：保留顺序语义（parity 关键），但消除每步 unsqueeze/squeeze 与
            # 重复的 norm 新建张量开销——用 F.normalize 原地式归一、comp_t 直接索引 (B,comp_dim)。
            slots = self.slots
            comp_t_all = comp  # (B, T, comp_dim)
            for t in range(T):
                ct = comp_t_all[:, t, :]                              # (B, comp_dim)
                sim = torch.einsum('bc,bmc->bm', ct, slots)          # (B, M)
                gate = torch.softmax(sim, dim=-1)                    # (B, M)
                update = torch.einsum('bm,bc->bmc', gate, ct)         # (B, M, comp_dim)
                slots = slots + update
                slots = F.normalize(slots, dim=-1, eps=1e-6)          # 替代 /(norm+1e-6)
            self.slots = slots
        else:
            # 原始：可学线性门控（位置相关分配，与 slots 无关，可向量化且全量/增量一致）
            gate = torch.softmax(self.write_gate(x), dim=-1)  # (B, T, M)
            # 按 gate 把压缩表示累加到槽（加权求和，可微）
            update = torch.einsum('btm,btc->bmc', gate, comp)  # (B, M, comp_dim)
            # 移动平均式软写入（保留历史记忆，新信息按 gate 权重叠加）
            self.slots = self.slots + update
            # 归一化防止数值膨胀
            self.slots = self.slots / (1e-6 + self.slots.norm(dim=-1, keepdim=True))

    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        """返回记忆的 K/V：(B, M, head_dim) 及检索元信息（门控/稀疏）。

        直接复用已对齐设备的 slots，不在热路径做 .to（避免 DML 设备别名导致的每步拷贝）。
        """
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

    def __init__(self, dim: int, base: float = 10000.0, learnable: bool = False):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.learnable = learnable
        # 阶段5：可学习 RoPE 频率缩放——让模型调整各频率尺度，更好适应长度/尺度。
        # inv_freq 实际 = buffer * exp(log_scale)，log_scale 每维可学（init 0 = 不变）。
        if learnable:
            self.rope_log_scale = nn.Parameter(torch.zeros(dim // 2))
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
        inv_freq = self.inv_freq
        if self.learnable:
            # 可学习频率缩放：inv_freq_eff = inv_freq * exp(log_scale)（per-dim 可学）
            inv_freq = inv_freq * torch.exp(self.rope_log_scale).to(inv_freq.device)
        freqs = torch.outer(t, inv_freq)
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
        # 标准旋转公式：x1*cos - x2*sin, x1*sin + x2*cos（无 cat、无 neg 临时张量）
        return torch.cat([
            x1 * cos[..., :d] - x2 * sin[..., :d],
            x1 * sin[..., :d] + x2 * cos[..., d:]
        ], dim=-1)

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
                 qk_norm: bool = True, attn_temp: bool = True, mask_fill_value: float = -1e9,
                 rope_learnable: bool = False, alibi: bool = False, retrieval_full: bool = False,
                 retrieval_topk: int = 32, learn_window: bool = False, window_base: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window = int(window or 0)
        self.rel_bias = rel_bias
        self.max_seq_length = max_seq_length
        self.mask_fill_value = float(mask_fill_value)
        self.alibi = alibi
        self.retrieval_full = retrieval_full
        self.retrieval_topk = retrieval_topk
        # 阶段6：可学习滑动窗口——每层一个 log_window 参数，实际窗口 = round(exp(log_window))*base，
        # 范围 clamp 到 [1, window_base]（或更宽），模型自决每层看多远。默认关（向后兼容）。
        self.learn_window = learn_window
        self.window_base = window_base
        if learn_window:
            # 初始化使初始窗口 = 配置 window（window=0 → 退化为 1，即仅相邻局部）
            init_w = max(1, self.window) if self.window > 0 else 1
            self.log_window = nn.Parameter(torch.tensor(math.log(max(init_w, 1) / max(window_base, 1))))
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, learnable=rope_learnable)
        if self.rel_bias:
            # T5 风格相对位置偏置表：(heads, 2T-1)
            self.rel_bias_table = nn.Parameter(torch.zeros(num_heads, 2 * max_seq_length - 1))
        # 阶段5：ALiBi 线性位置偏置——对距离线性惩罚，长度外推极稳，与 RoPE 互补。
        # 每个头一个斜率 m_h = 2^(-h/H * 8)，bias = -m_h * |i-j|（注入 attn scores 前）。
        if alibi:
            # 头斜率（固定、不可学，符合 ALiBi 原设计）；短序列也安全
            m = torch.tensor([2.0 ** (-(h + 1) / num_heads * 8.0) for h in range(num_heads)])
            self.register_buffer('alibi_slopes', m, persistent=False)
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
        # 训练路径静态偏置掩码缓存（仅依赖 T/Tkv/mem_cols，逐层逐步重建代价高）：
        # 避免每步每头重复 arange/torch.zeros/cat 造成的海量分配与 DML 拷贝开销
        self._bias_key: Optional[tuple] = None
        self._bias_cache: Optional[torch.Tensor] = None

    def _sync_window(self):
        """阶段6：从可学习 log_window 重算实际窗口尺寸（每步前向同步，训练时随参数变化）。

        log_window 初始化为 log(init_w / window_base)，故还原须乘回 window_base，
        否则 exp 后丢失 base 缩放、任意 window<32 都会被 round 成 1（窗口无声退化）。
        """
        if self.learn_window:
            w = int(round(math.exp(float(self.log_window)) * self.window_base))
            w = max(1, min(w, max(self.window_base, 1) * 4))
            if w != self.window:
                self.window = w
                self._bias_key = None  # 窗口变化 → 掩码缓存失效

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

    def _alibi_bias(self, Tq: int, Tkv: int, device: torch.device, start_pos: int = 0) -> Optional[torch.Tensor]:
        """ALiBi 线性位置偏置：(1, H, Tq, Tkv)，bias[h,i,j] = -m_h * |i-j|。

        start_pos 为增量解码时当前窗口首 token 的绝对位置，必须传入，
        否则缓存路径会把每个查询当成序列第 0 位、造成训练-推理位置偏移。
        """
        if not self.alibi:
            return None
        qpos = torch.arange(start_pos, start_pos + Tq, device=device).unsqueeze(1)
        kpos = torch.arange(0, Tkv, device=device).unsqueeze(0)
        dist = (qpos - kpos).abs().to(device)
        # slopes: (H,) -> (1,H,1,1)，乘以距离 -> (1,H,Tq,Tkv)
        bias = -self.alibi_slopes.view(1, self.num_heads, 1, 1).to(device) * dist.unsqueeze(0).unsqueeze(0)
        return bias

    def _full_retrieval_bias(self, q: torch.Tensor, k_full: torch.Tensor, Treal: int, mem_cols: int,
                             gate: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
        """全上下文检索（阶段3 扩展）：对真实序列 KV（窗口外远端）做相似性检索，
        仅保留 top-k 最相关位置（局部窗口恒保留），注入为注意力正偏置。
        返回 (B,H,Tq,Tkv_all)，记忆段与局部窗口位置为 0（不额外奖励），远端仅保留检索到的少数槽。"""
        if not self.retrieval_full or Treal <= 0:
            return None
        # 真实 KV 段：k_full[..., mem_cols:mem_cols+Treal, :]
        k_real = k_full[:, :, mem_cols:mem_cols + Treal, :]  # (B,H,Treal,D)
        rlogits = torch.einsum('bhqd,bhkd->bhqk', q, k_real)  # (B,H,Tq,Treal)
        if gate is not None:
            rlogits = rlogits * gate
        # 局部窗口恒保留：对每个 query q，保留其因果窗口 [q-window, q] 内的 key 位置，
        # 防止这些本应可见的位置被 top-k 稀疏误丢。原实现仅保留全局末尾 window+1 个位置，
        # 导致早期 query 的窗口内 key 无 +1e9 保护，被 top-k 丢弃后 retrieval bias 叠加
        # -1e9 到基础掩码的 0 上 → 静默遮蔽本应可见的位置。
        if self.window > 0:
            Tq = q.size(2)
            qpos_q = torch.arange(Tq, device=device).unsqueeze(1)  # (Tq, 1)
            kpos = torch.arange(Treal, device=device).unsqueeze(0)  # (1, Treal)
            keep = ((qpos_q - kpos) <= self.window) & (kpos <= qpos_q)  # (Tq, Treal)
            rlogits = rlogits + keep.unsqueeze(0).unsqueeze(0).float() * 1e9
        # 因果：未来位置本就被 attn_mask 屏蔽，这里也压到 -inf 不参与检索
        causal = torch.triu(torch.ones(Treal, Treal, dtype=torch.bool, device=device), diagonal=1)
        rlogits = rlogits.masked_fill(causal.unsqueeze(0).unsqueeze(0), self.mask_fill_value)
        # top-k 稀疏（保留最相关 k 个），余下压 -inf
        k_keep = max(1, min(self.retrieval_topk, Treal))
        kvals, _ = torch.topk(rlogits, k_keep, dim=-1)
        thr = kvals[..., -1:]
        drop = (rlogits < thr).to(device)
        rlogits = rlogits.masked_fill(drop, self.mask_fill_value)
        # 拼回完整 Tkv（记忆段前缀补 0）
        if mem_cols > 0:
            rlogits = torch.cat([torch.zeros(rlogits.size(0), rlogits.size(1), rlogits.size(2), mem_cols,
                                          device=device, dtype=rlogits.dtype), rlogits], dim=-1)
        return rlogits

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
        self._sync_window()
        # DML 设备别名不一致（privateuseone vs privateuseone:0）：以本模块权重所在设备为权威，
        # 所有掩码/缓存构建都用它，避免 q.device 被剥索引导致 _build_masks/_bias_cache 每步重建
        dev = self.qkv.weight.device
        # 阶段3 可学习检索：先计算记忆段偏置（门控缩放 + top-k 稀疏），注入后续 attn_mask
        mem_bias: Optional[torch.Tensor] = None
        mem_cols = 0
        meta = None
        mem_bias = None
        if memory_kv is not None:
            mk, mv, meta = memory_kv
            # 记忆列数恒由记忆 KV 张量维度决定（与 meta 是否为 None 无关），
            # 否则 memory_retrieval=False 时 mem_cols 停留 0、记忆段被因果掩码静默遮蔽。
            mem_cols = mk.size(1)
            # 记忆查询相似度（每槽点积）：(B,H,Tq,M)，廉价（M 小）
            mlogits = torch.einsum('bhqd,bhmd->bhqm', q,
                                   mk.unsqueeze(1).expand(-1, self.num_heads, -1, -1))
            if meta is not None:
                # 仅当开启检索/稀疏时才加偏置；否则记忆仅作为全局 KV 参与注意力（不加额外 bias）
                if meta.get('retrieval_gate') is not None:
                    # 可学门控：sigmoid → (0,1) 软增强/抑制记忆召回，受 LM loss 监督
                    gate = torch.sigmoid(meta['retrieval_gate']).view(1, 1, 1, 1).to(mlogits.device)
                    mlogits = mlogits * gate
                if meta.get('sparse_topk', 0) and meta['sparse_topk'] < mem_cols:
                    k_keep = meta['sparse_topk']
                    # 每查询保留 top-k 记忆槽，余下压到 -inf（可微稀疏，降低无关记忆干扰）
                    # 用 topk 计算阈值（DML 下 kthvalue 的 CPU fallback 会返回异常形状，topk 更稳）
                    kvals, _ = torch.topk(mlogits, k_keep, dim=-1)  # (B,H,Tq,k_keep)
                    thr = kvals[..., -1:]  # 第 k 大的值作为阈值 (B,H,Tq,1)
                    drop = (mlogits < thr).to(mlogits.device)
                    mlogits = mlogits.masked_fill(drop, self.mask_fill_value)
                mem_bias = mlogits  # (B,H,Tq,M)，作为 scores 的可加偏置

        if use_cache:
            # 增量解码：拼接待拼接的 K/V 缓存，仅对当前 token 做注意力
            if past_kv is not None:
                # past_kv 可能为 (k, v) 或混合 mixer 的 (k, v, linear_S)，仅取前两项
                pk, pv = past_kv[0], past_kv[1]
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)
            # present 存累积的 token KV（past+token，不含 memory），作为下一步的 past_kv；
            # memory 只在注意力计算时临时拼接，不进入缓存，避免序列长度膨胀。
            present = (k, v)
            if memory_kv is not None:
                mk, mv = memory_kv[0], memory_kv[1]
                # (B,M,D) -> (B,H,M,D) 各头共享记忆 KV
                mk = mk.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
                mv = mv.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
                k = torch.cat([mk, k], dim=2)
                v = torch.cat([mv, v], dim=2)
            Tkv = k.size(2)
            qpos = torch.arange(start_pos, start_pos + Tq, device=q.device).unsqueeze(1)
            kpos = torch.arange(0, Tkv, device=q.device).unsqueeze(0)
            # 与全量路径 line 543 对齐：掩码用「合并坐标」（记忆段占前 mem_cols 列、真实序列紧随其后），
            # 因果 = 真实键列 > 查询绝对位置 + mem_cols（记忆列恒被该式排除、再显式清零双保险），
            # 保证 memory+window>0 时训练/推理一致性（否则推理期记忆按位置被部分遮蔽、静默质量退化）。
            causal_mask = kpos > (qpos + mem_cols)
            if self.window > 0:
                causal_mask = causal_mask | (qpos - kpos > self.window)
            if mem_cols > 0:
                causal_mask[..., :mem_cols] = False
            attn_mask = (causal_mask.float() * self.mask_fill_value).unsqueeze(0)   # (1,1,T,Tkv)
            # 记忆槽位置在窗口 KV 之前（seq 起点之前），永远不被因果遮蔽，
            # 但也不参与"未来"泄露：记忆是历史压缩，视为已发生，不施加 causal 惩罚
            if self.rel_bias:
                idx = (qpos - kpos + Tkv - 1).clamp(0, 2 * self.max_seq_length - 1)
                attn_mask = attn_mask + self.rel_bias_table[:, idx].unsqueeze(0)
            if mem_bias is not None:
                # mem_bias: (B,H,Tq,mem_cols)，右侧补零到 Tkv 再与 attn_mask 广播相加
                padded = torch.nn.functional.pad(mem_bias, (0, Tkv - mem_cols))
                attn_mask = attn_mask + padded
            alibi_b = self._alibi_bias(Tq, Tkv, q.device, start_pos)
            if alibi_b is not None:
                attn_mask = attn_mask + alibi_b
            # 全上下文检索（与全量路径 line 577-583 对齐）：cache 路径同样注入对真实 KV 远端的 top-k 稀疏偏置，
            # 否则开启 retrieval_full 时训练-推理系统性不一致（生成质量偏离训练行为）。
            if self.retrieval_full and mem_cols < Tkv:
                rgate = None
                if memory_kv is not None and memory_kv[2] is not None and memory_kv[2].get('retrieval_gate') is not None:
                    rgate = torch.sigmoid(memory_kv[2]['retrieval_gate']).view(1, 1, 1, 1).to(q.device)
                rbias = self._full_retrieval_bias(q, k, Tkv - mem_cols, mem_cols, rgate, q.device)
                if rbias is not None:
                    attn_mask = attn_mask + rbias
            # 与全量（非缓存）路径走同一后端：cuda/cpu 用 fused SDPA、DML(privateuseone) 用 manual，
            # 保证训练-推理在带偏置（alibi/rel_bias/mem_bias）时数值一致。
            if q.device.type in ('cuda', 'cpu'):
                out = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            else:
                out = self._manual_attention(q, k, v, attn_mask)
            out = out.transpose(1, 2).reshape(B, Tq, self.num_heads * self.head_dim)
            return self.proj(out), present

        # —— 非缓存（训练 / 含 SSM 模型全量重算）路径 ——
        T = q.size(2)
        self._build_masks(T, dev)
        Tkv = k.size(2)
        # 训练路径把记忆拼到 K/V 之前（记忆在前，窗口/全量在后）
        Treal = Tkv  # 真实序列 KV 长度（记忆拼接前）
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
        # 静态部分（窗口/因果掩码）仅依赖 (T, Tkv, mem_cols)，缓存复用避免每步每头重建
        if memory_kv is not None:
            cache_key = (T, Tkv, mem_cols)
        else:
            cache_key = (T, Tkv, 0)
        if self._bias_key != cache_key or self._bias_cache is None or self._bias_cache.device != dev:
            base = torch.zeros(1, 1, T, Tkv, device=dev)
            if self.window > 0:
                qpos = torch.arange(0, T, device=dev).unsqueeze(1)
                kpos = torch.arange(0, Tkv, device=dev).unsqueeze(0)
                # 滑动窗口因果掩码：遮蔽过远的过去 AND 遮蔽未来（与 cache 路径 line 383-386 一致）
                m = ((qpos - kpos > self.window) | (kpos > qpos + mem_cols)).float() * self.mask_fill_value
                m = m.unsqueeze(0).unsqueeze(0)  # (1,1,T,Tkv)
                if memory_kv is not None:
                    m[:, :, :, :mem_cols] = 0.0  # 记忆段不受窗口/因果限制
                base = base + m
            elif self.rel_bias:
                base = base + (self._mask.float() * self.mask_fill_value)
            elif memory_kv is not None or self.alibi:
                # 记忆开启但 window==0 且非 rel_bias，或仅开启 ALiBi（window==0）：
                # 主序列段仍需因果遮蔽，否则记忆/ALiBi 路径下未来 token 会泄漏
                # （记忆段本身全 0，不受因果限制）。
                qpos = torch.arange(0, T, device=dev).unsqueeze(1)
                kpos = torch.arange(0, Tkv, device=dev).unsqueeze(0)
                # 主序列因果：K/V 前 mem_cols 列是记忆（全局可检索，不施加因果），
                # 主序列段（全局索引 >= mem_cols）的相对位置 = kpos - mem_cols，
                # 故因果条件为 kpos > qpos + mem_cols（与 window 分支 line 582、
                # cache 路径 line 498 一致）。原 kpos > qpos 会多遮 16 列合法过去 token。
                cm = (kpos > (qpos + mem_cols)).float() * self.mask_fill_value  # 主序列因果
                cm = cm.unsqueeze(0).unsqueeze(0)  # (1,1,T,Tkv)
                if mem_cols > 0:
                    cm[:, :, :, :mem_cols] = 0.0  # 记忆段全 0（全局可检索）
                base = base + cm
            if self.rel_bias:
                idx = (torch.arange(T, device=dev).unsqueeze(1)
                       - torch.arange(Tkv, device=dev).unsqueeze(0)
                       + Tkv - 1).clamp(0, 2 * self.max_seq_length - 1)
                base = base + self.rel_bias_table[:, idx].unsqueeze(0)
            self._bias_key = cache_key
            self._bias_cache = base
        attn_mask = self._bias_cache
        if mem_bias is not None:
            # mem_bias: (B,H,T,mem_cols)，右侧补零到 Tkv 再与 attn_mask 广播相加
            padded = torch.nn.functional.pad(mem_bias, (0, Tkv - mem_cols))  # (B,H,T,Tkv)
            attn_mask = attn_mask + padded
        alibi_b = self._alibi_bias(T, Tkv, dev, start_pos)
        if alibi_b is not None:
            attn_mask = attn_mask + alibi_b
        # 全上下文检索（阶段3 扩展）：对真实 KV 远端做稀疏检索，注入正偏置
        if self.retrieval_full:
            rgate = None
            if memory_kv is not None and memory_kv[2] is not None and memory_kv[2].get('retrieval_gate') is not None:
                rgate = torch.sigmoid(memory_kv[2]['retrieval_gate']).view(1, 1, 1, 1).to(dev)
            rbias = self._full_retrieval_bias(q, k, Treal, mem_cols, rgate, dev)
            if rbias is not None:
                attn_mask = attn_mask + rbias
        if q.device.type in ('cuda', 'cpu'):
            # 静态条件：无自定义掩码时用 fused is_causal（避免运行时 abs().max() sync）
            _use_causal = (not self.rel_bias) and (memory_kv is None) and (self.window == 0) and (not self.alibi)
            if _use_causal:
                out = scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                out = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            # DML/其他：直接传 mask（all-zeros 时 scores+zeros 是 no-op）
            # 消除 6 次/步的 host-device sync（abs().max() → __bool__() → .item()）
            out = self._manual_attention(q, k, v, attn_mask)
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
            scores = scores.masked_fill(causal, self.mask_fill_value)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)                             # (B, H, Tq, D)


class LinearAttention(nn.Module):
    """线性注意力（线性复杂度 token mixer，O(N) 推理，天然兼容 KV-cache）。

    特征映射 φ=elu(x)+1 后，注意力写为 S = Σ φ(K)⊗V 的递推（因果：按时间累积），
    较 softmax 注意力省去 O(N²) 的 scores 矩阵，长序列/小 iGPU 下显著省算力。
    与 SlidingWindowCausalSelfAttention 同接口（project_and_norm + attend），便于混合门控。
    """

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, attn_temp: bool = True,
                 max_seq_length: int = 64, feature: str = 'elu'):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_seq_length = max_seq_length
        self.feature = feature
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self.qk_norm_enabled = qk_norm
        if qk_norm:
            self.qk_norm = RMSNorm(self.head_dim)
        self.attn_temp_enabled = attn_temp
        if attn_temp:
            self.log_temp = nn.Parameter(torch.zeros(1))
        self._rt: Dict[str, bool] = {"qk_norm": True, "attn_temp": True}

    def _feat(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature == 'elu':
            return torch.nn.functional.elu(x) + 1.0
        return torch.nn.functional.relu(x) + 1e-6

    def project_and_norm(self, x: torch.Tensor, start_pos: int = 0):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm_enabled and self._rt["qk_norm"]:
            q = self.qk_norm(q)
            k = self.qk_norm(k)
        if self.attn_temp_enabled and self._rt["attn_temp"]:
            s = torch.exp(-0.5 * self.log_temp)
            q = q * s
            k = k * s
        q, k = self.rope(q, k, start_pos=start_pos, max_len=self.max_seq_length)
        return q, k, v

    def forward(self, x: torch.Tensor, past_kv=None, use_cache: bool = False, start_pos: int = 0,
                memory_kv=None):
        # 简化实现：全量因果线性注意力（增量解码复用 past_kv 累积状态 S）。
        # past_kv 可来自 attn_kv 元组第 3 位（混合 mixer 时）：(k, v, linear_S)。
        # 阶段8.9 向量化：用 cumsum 替代 for-t 循环，消除 O(T) Python 步。
        q, k, v = self.project_and_norm(x, start_pos)
        B, H, T, D = q.shape
        qf = self._feat(q)  # (B,H,T,D)
        kf = self._feat(k)  # (B,H,T,D)
        # 外积: (B,H,T,D,D)
        kv_all = torch.einsum('bhtd,bhte->bhtde', kf, v)
        # 累积状态: S_t = Σ_{i=0}^{t} kv_i —— cumsum 沿 T 维
        S_all = torch.cumsum(kv_all, dim=2)  # (B,H,T,D,D)
        # 分母累积: z_t = Σ_{i=0}^{t} φ(k_i)，den_t = φ(q_t)·z_t
        z_all = torch.cumsum(kf, dim=2)  # (B,H,T,D)
        # 若有历史状态，偏移累积和
        S_prev = None
        if past_kv is not None and len(past_kv) >= 3 and past_kv[2] is not None:
            S_prev = past_kv[2]  # (B,H,D,D)
            S_all = S_all + S_prev.unsqueeze(2)  # 广播加到每个位置
            # 历史分母状态（z_prev: (B,H,D)），也需要偏移
            if len(past_kv) >= 4 and past_kv[3] is not None:
                z_all = z_all + past_kv[3].unsqueeze(2)
        # 分子: q·S（每个位置查对应累积状态）
        num = torch.einsum('bhtd,bhtde->bhte', qf, S_all)  # (B,H,T,D)
        # 分母: φ(q_t)·z_t（累积分母，保证归一化正确）
        den = torch.einsum('bhtd,bhtd->bht', qf, z_all).unsqueeze(-1).clamp_min(1e-6)  # (B,H,T,1)
        out = (num / den).transpose(1, 2).reshape(B, T, H * D)
        # present: 末尾累积状态供增量解码（S_final + z_final）
        S_final = S_all[:, :, -1, :, :]
        z_final = z_all[:, :, -1, :]
        present = (k, v, S_final, z_final) if use_cache else None
        return self.proj(out), present


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
         - D 跳跃连接置 D_init（构造函数传入值，非固定 1.0）
        """
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.xavier_uniform_(self.dt_proj.weight)
        nn.init.constant_(self.dt_proj.bias, 0.1)
        nn.init.uniform_(self.A_log, *self.a_log_init_range)
        nn.init.ones_(self.D)
        self.D.data.mul_(self.D_init)

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
        
        # If we have past_state, incorporate it: A is already the prefix product
        if past_state is not None:
            # A[:, t] = a_t * a_{t-1} * ... * a_0（标准扫描已得出，无需重算）
            past_expanded = past_state.unsqueeze(1).expand(-1, L, -1, -1)
            B = B + A * past_expanded
        
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
            self.mixer = mixer
            if mixer == 'linear':
                # 阶段7：纯线性注意力（O(N) token mixer）
                self.attn = LinearAttention(dim, num_heads, max_seq_length=max_seq_length,
                                           qk_norm=attn_kwargs.get('qk_norm', True),
                                           attn_temp=attn_kwargs.get('attn_temp', True))
                self.linear_attn = None
            elif mixer == 'hybrid':
                # 阶段7：attn + 线性注意力 两路并行，可学习 mixer_gate 自选择用多少
                self.attn = SlidingWindowCausalSelfAttention(
                    dim, num_heads, max_seq_length=max_seq_length, **attn_kwargs)
                self.linear_attn = LinearAttention(dim, num_heads, max_seq_length=max_seq_length,
                                                  qk_norm=attn_kwargs.get('qk_norm', True),
                                                  attn_temp=attn_kwargs.get('attn_temp', True))
                self.mixer_gate = nn.Parameter(torch.ones(1))  # init 1.0 → 偏 attn
            else:
                self.attn = SlidingWindowCausalSelfAttention(
                    dim, num_heads, max_seq_length=max_seq_length, **attn_kwargs)
                self.linear_attn = None
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
            if ckpt and hasattr(self.attn, 'attend'):
                # SlidingWindowCausalSelfAttention：拆分 project_and_norm / attend 以缩小检查点重算区
                q, k, v = self.attn.project_and_norm(self.ln1(x), start_pos)
                h, present = checkpoint(self.attn.attend, q, k, v, attn_past_kv, use_cache, start_pos, mem_kv, use_reentrant=False)
            else:
                # LinearAttention 等无 attend 接口的 mixer：直接对整层前向做检查点
                h, present = checkpoint(self.attn, self.ln1(x), attn_past_kv, use_cache, start_pos, mem_kv, use_reentrant=False) if ckpt else \
                    self.attn(self.ln1(x), attn_past_kv, use_cache, start_pos, memory_kv=mem_kv)
            if self.linear_attn is not None:
                # 阶段7：混合 mixer（attn + 线性注意力并行），mixer_gate 自选择比例
                lh, lpresent = self.linear_attn(self.ln1(x), attn_past_kv, use_cache, start_pos, memory_kv=mem_kv)
                mg = torch.sigmoid(self.mixer_gate).to(x.device)
                h = mg * h + (1.0 - mg) * lh
                if use_cache and lpresent is not None:
                    # 两路 KV 缓存合并：把线性注意力状态 S 和分母累积 z 塞进 attn_kv 元组
                    # (k, v, linear_S, z_final)，保持块级 (attn_kv, ssm_state, ssm_conv_state) 三元组不变。
                    present = ((present[0], present[1], lpresent[2], lpresent[3] if len(lpresent) > 3 else None), None, None)
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
            if ckpt:
                q, k, v = self.attn.project_and_norm(xn, start_pos)
                h, attn_present = checkpoint(self.attn.attend, q, k, v, attn_past_kv, use_cache, start_pos, mem_kv, use_reentrant=False)
                ssm_h, ssm_present_state, ssm_present_conv_state = checkpoint(self.ssm, xn, ssm_past_state, ssm_past_conv_state, use_cache, use_reentrant=False)
            else:
                h, attn_present = self.attn(xn, attn_past_kv, use_cache, start_pos, memory_kv=mem_kv)
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
                           learn_window=learn_window, window_base=window_base)
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

    def forward(self, src: torch.Tensor, past_key_values: Optional[List[Optional[Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]]]] = None, use_cache: bool = False, intuition: Optional[torch.Tensor] = None, igmcg_force_off: bool = False) -> Tuple[torch.Tensor, Optional[List[Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]]]]:
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
            logp = F.log_softmax(z, dim=-1)                                 # (B,T,V) 同尺度 log 概率
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
        self._ngram_last_ids = None
        generated = list(token_ids)
        max_seq_length = self.max_seq_length
        eos_token_id = eos_id
        pad_token_id = pad_id
        sep_token_id = sep_id
        # 现在支持混合架构的增量解码（SSM 也有增量状态）
        use_cache = True

        def sample_step(logits_t: torch.Tensor) -> Optional[int]:
            next_token_logits = logits_t / temperature
            # 向量化重复惩罚：一次性处理所有已生成 token，消除 Python 循环
            vocab_size = next_token_logits.shape[0]
            prev_tokens = torch.tensor(
                [t for t in set(generated) if 0 <= t < vocab_size],
                dtype=torch.long, device=device
            )
            if prev_tokens.numel() > 0:
                # 加性频率惩罚（统一 IGMCG 路径与 sample_step 路径）：
                # 对已出现 token 按出现次数减去 penalty，对称稳定无正负不对称问题。
                from collections import Counter
                freq = Counter(generated)
                prev_toks = torch.tensor(list(freq.keys()), dtype=torch.long, device=device)
                prev_counts = torch.tensor(list(freq.values()), dtype=torch.float, device=device)
                valid = (prev_toks >= 0) & (prev_toks < vocab_size)
                next_token_logits[prev_toks[valid]] -= repetition_penalty * prev_counts[valid]
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