from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self._forget_active: bool = True

    def _init_slots(self):
        # 记忆槽以压缩空间零初始化（forward 首步由 reset 填充）
        self.register_buffer('slots', torch.zeros(1, self.num_slots, self.comp_dim),
                             persistent=False)
        # K/V 缓存（slots 未变时复用，避免多块重复解压）
        self._kv_cache = None
        self._kv_cache_slots = None

    def reset(self, batch: int, device: torch.device, dtype: torch.dtype):
        """新建 batch 大小的记忆（每个样本独立槽）。

        统一对齐到本模块权重所在设备（DML 的 privateuseone vs privateuseone:0
        别名不一致会导致后续 .to() 每步产生大量设备拷贝，故在此一次对齐到位，
        get_kv/write 热路径不再做 .to，消除拷贝开销。
        """
        dev = self.compress.weight.device if device is None else device
        self.slots = torch.zeros(batch, self.num_slots, self.comp_dim,
                                 device=dev, dtype=dtype)
        # slots 重建 → 缓存失效
        self._kv_cache = None
        self._kv_cache_slots = None
        self._kv_dirty = True

    def write(self, x: torch.Tensor) -> None:
        """x: (B, T, D) 当前层表示，soft 写入记忆。

        forget_gate parity 修复：原实现每次 write() 调用施加一次 forget 衰减。
        训练时每块 1 次 write(T 个 token) = 1 次衰减；增量解码每块 T 次 write(1 token) = T 次衰减。
        衰减次数不同导致 train/infer divergence。修复：把 forget 衰减移入逐 token 循环，
        按 token 数衰减（训练 1 次 write T token → 衰减 T 次；增量 T 次 write → 共 T 次），保证一致。

        product_key 跨块顺序注意：product_key 模式下 gate 依赖 slots，slots 随写入变化。
        训练是 block-major（block0 写全部 T token → block1 写），增量是 token-major（各 block 逐 token）。
        两者 slots 演化路径不同 → 产生 divergence。这是已知架构限制，product_key 模式建议
        仅在单层或 train/infer 同序时使用；多层 + 增量解码场景请用非 product_key 路径。
        """
        B, T, D = x.shape
        # slots 即将变更 → 失效 K/V 缓存（get_kv 下次重算）
        self._kv_cache = None
        self._kv_cache_slots = None
        # 仅在 batch 变化或设备确实不一致时重建（正常训练每步同设备，不触发拷贝）
        if self.slots.shape[0] != B or self.slots.device != x.device:
            if self.slots.device != x.device:
                x = x.to(self.slots.device)
            self.reset(B, self.slots.device, x.dtype)
        # 压缩当前表示
        comp = self.compress(x)  # (B, T, comp_dim)
        # forget 衰减因子（per-slot，逐 token 施加以保证 train/infer parity）
        f_per_slot = None
        if self.forget_enabled and getattr(self, '_forget_active', True):
            f_per_slot = torch.sigmoid(self.forget_gate).view(1, -1, 1)  # (1, M, 1)
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
                # forget 逐 token 施加（与增量解码逐 token write 一致）
                if f_per_slot is not None:
                    slots = f_per_slot * slots
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
            # forget 逐 token 衰减：训练 1 次 write(T) → 衰减 T 次后累加 T 个 update；
            # 增量 T 次 write(1) → 每次衰减 1 次后累加 1 个 update，共 T 次。两者等价。
            if f_per_slot is not None:
                # 逐 token 语义：slots_T = f^T * slots_0 + Σ_{t=0}^{T-1} f^{T-1-t} * update_t
                # 向量化：f_per_slot (1,M,1)；衰减权重 (M,T) 其中 [m,t]=f[m]^(T-1-t)
                # gate (B,T,M) 需按 (m,t) 衰减 → decay (M,T) 转置成 (T,M) 后广播乘
                f_vec = f_per_slot.squeeze(0).squeeze(-1)  # (M,)
                t_idx = torch.arange(T, device=f_per_slot.device, dtype=f_per_slot.dtype)
                decay_mt = f_vec.unsqueeze(1) ** (T - 1 - t_idx).unsqueeze(0)  # (M, T)
                decayed_gate = gate * decay_mt.t().unsqueeze(0)  # (B, T, M)
                weighted_update = torch.einsum('btm,btc->bmc', decayed_gate, comp)  # (B, M, comp_dim)
                f_pow = f_per_slot ** T  # (1, M, 1) = f^T
                self.slots = f_pow * self.slots + weighted_update
            else:
                # 移动平均式软写入（保留历史记忆，新信息按 gate 权重叠加）
                # 关键：此处【不】对 slots 做逐写归一化。原实现每次 write 后都 L2 归一化，
                # 导致全量前向（一次写 T 个 token，归一化 1 次）与增量解码（逐 token 写，
                # 每次归一化）的累加结果不一致（norm(a+b) ≠ norm(norm(a)+b)），产生训练-推理
                # 记忆 divergence（slots 差 ~0.6）。归一化改为在读取时（_recompute_kv_cache）
                # 统一做一次，与写入粒度无关，保证全量/增量记忆槽逐位一致。
                self.slots = self.slots + update
        # write 已更新 slots → 标记缓存脏，延迟到下次 get_kv 时按需重算（惰性重算）。
        # 避免：(1) 最后一层 write 后的无用重算（写完不读）；(2) 多层重复解压开销。
        self._kv_dirty = True

    def _recompute_kv_cache(self):
        # 读取时统一 L2 归一化（与写入粒度无关，保证全量/增量记忆一致；幂等于已归一化情形）
        normed = self.slots / (1e-6 + self.slots.norm(dim=-1, keepdim=True))
        decomp = self.decompress(normed)  # (B, M, D)
        self._kv_cache = (self.mem_k(decomp), self.mem_v(decomp))
        self._kv_cache_slots = self.slots

    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        """返回记忆的 K/V：(B, M, head_dim) 及检索元信息（门控/稀疏）。

        惰性重算：write() 标记 _kv_dirty=True，此处按需重算（slots 未变且非脏时直接复用缓存）。
        避免：(1) 最后一层 write 后无用重算；(2) 多层重复解压。
        """
        if getattr(self, '_kv_dirty', True) or getattr(self, '_kv_cache', None) is None or self._kv_cache_slots is not self.slots:
            self._recompute_kv_cache()
            self._kv_dirty = False
        k, v = self._kv_cache
        meta = None
        if self.retrieval_enabled or self.sparse_topk > 0:
            meta = {
                'retrieval_gate': self.retrieval_gate if self.retrieval_enabled else None,
                'sparse_topk': self.sparse_topk,
                'num_slots': self.num_slots,
            }
        return k, v, meta

    @staticmethod
    def inject_memory(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      mk: torch.Tensor, mv: torch.Tensor, meta: Optional[Dict[str, Any]],
                      mask_fill: float) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """把可学习压缩记忆注入注意力 K/V 并返回记忆段偏置（B 项核心收敛点）。

        纯函数（无实例状态），统一取代 attend 全量/cache 两条路径里重复的"记忆 K/V
        拼接 + retrieval_gate 稀疏门控"逻辑，消除两路径因各自实现而漂移的风险
        （阶段3/13 的 cache-parity bug 正源于此）。全上下文检索偏置由 attend 调用方
        经 `_full_retrieval_bias`（依赖实例 window/topk 开关）另行计算。

        Args:
            q: (B,H,Tq,D) 查询
            k,v: 主序列 K/V（B,H,Tkv,D），记忆将拼到其前面
            mk,mv: 记忆 K/V（B,M,D）
            meta: 检索元信息（retrieval_gate / sparse_topk），可为 None
            mask_fill: 掩码填充值（-1e9）
        Returns:
            k_aug: 记忆拼接后的 K (B,H,M+Tkv,D)
            v_aug: 记忆拼接后的 V (B,H,M+Tkv,D)
            mem_bias: 记忆段可加偏置 (B,H,Tq,M) 或 None
        """
        mem_cols = mk.size(1)
        # 各头共享记忆 K/V：升维到 (B,H,M,D) 供点积与拼接复用（避免重复 expand）
        mk_e = mk.unsqueeze(1).expand(-1, q.size(1), -1, -1)
        mv_e = mv.unsqueeze(1).expand(-1, q.size(1), -1, -1)
        # 记忆查询相似度（每槽点积）：(B,H,Tq,M)，廉价（M 小）
        mlogits = torch.einsum('bhqd,bhmd->bhqm', q, mk_e)
        if meta is not None:
            # 仅当开启检索/稀疏时才加偏置；否则记忆仅作为全局 KV 参与注意力（不加额外 bias）
            if meta.get('retrieval_gate') is not None:
                # 可学门控：sigmoid → (0,1) 软增强/抑制记忆召回，受 LM loss 监督
                gate = torch.sigmoid(meta['retrieval_gate']).view(1, 1, 1, 1).to(mlogits.device)
                mlogits = mlogits * gate
            if meta.get('sparse_topk', 0) and 0 < meta['sparse_topk'] < mem_cols:
                k_keep = meta['sparse_topk']
                # 每查询保留 top-k 记忆槽，余下压到 -inf（可微稀疏，降低无关记忆干扰）
                kvals, _ = torch.topk(mlogits, k_keep, dim=-1)  # (B,H,Tq,k_keep)
                thr = kvals[..., -1:]  # 第 k 大的值作为阈值 (B,H,Tq,1)
                drop = (mlogits < thr).to(mlogits.device)
                mlogits = mlogits.masked_fill(drop, mask_fill)
        mem_bias = mlogits  # (B,H,Tq,M)，作为 scores 的可加偏置

        # 记忆拼到 K/V 之前（记忆在前，窗口/全量在后）；各头共享记忆 K/V
        k_aug = torch.cat([mk_e, k], dim=2)
        v_aug = torch.cat([mv_e, v], dim=2)
        return k_aug, v_aug, mem_bias
