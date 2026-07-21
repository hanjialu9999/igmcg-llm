from __future__ import annotations
import math
import threading
from typing import Optional, List, Tuple, Any, Dict, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch.utils.checkpoint import checkpoint
from models.constants import MASK_FILL_VALUE, ROPE_BASE
from models.norms import RMSNorm
from models.rope import RotaryEmbedding
from models.memory import MemoryBank


def apply_qk_norm_and_temp(q: torch.Tensor, k: torch.Tensor,
                            rt: Dict[str, bool], qk_norm: Optional[nn.Module],
                            log_temp: Optional[nn.Parameter]) -> Tuple[torch.Tensor, torch.Tensor]:
    """QK-Norm + 可学习温度 + RoPE 之前的共享预处理（额外2 去重）。

    SlidingWindowCausalSelfAttention 与 LinearAttention 的 project_and_norm 中三段逻辑
    几乎逐字重复，统一到此处：
      - ① QK-Norm：投影后、RoPE 前对 Q/K 各自归一化（运行时开关 _rt 可跳过）；
      - ⑤ 可学习温度：温度恒正（T=exp(log_temp)），直接缩放 Q/K 幅值
        （等价 softmax(score/T)，融合为单次标量乘法 q*=exp(-0.5*log_temp)，免额外 sqrt）。
    返回处理后的 (q, k)。"""
    if qk_norm is not None and rt.get("qk_norm", True):
        q = qk_norm(q)
        k = qk_norm(k)
    if log_temp is not None and rt.get("attn_temp", True):
        scale = torch.exp(-0.5 * log_temp)
        q = q * scale
        k = k * scale
    return q, k

class SlidingWindowCausalSelfAttention(nn.Module):
    """因果自注意力，可选滑动窗口 + 可学习相对位置偏置。
     CUDA/CPU 用原生 fused SDPA；AMD DirectML 的 fused 内核会触发原生崩溃，
     故 DML(及其他后端)走手动 matmul+softmax+因果掩码 以规避该 bug。
    """
    def __init__(self, dim: int, num_heads: int, window: int = 0, rel_bias: bool = False, max_seq_length: int = 64,
                 qk_norm: bool = True, attn_temp: bool = True, mask_fill_value: float = MASK_FILL_VALUE,
                 rope_learnable: bool = False, alibi: bool = False, retrieval_full: bool = False,
                 retrieval_topk: int = 32, learn_window: bool = False, window_base: int = 64,
                 shared_qkv: Optional[nn.Linear] = None, shared_proj: Optional[nn.Linear] = None):
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
        self.learn_window = learn_window
        self.window_base = window_base
        if learn_window:
            init_w = max(1, self.window) if self.window > 0 else 1
            self.log_window = nn.Parameter(torch.tensor(math.log(max(init_w, 1) / max(window_base, 1))))
        # 层间共享：传入 shared_qkv/shared_proj 时复用外部投影，不在本层创建新参数
        self.qkv = shared_qkv if shared_qkv is not None else nn.Linear(dim, 3 * dim, bias=False)
        self.proj = shared_proj if shared_proj is not None else nn.Linear(dim, dim, bias=False)
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
        # 增量解码（cache 路径）纯因果掩码缓存：掩码仅依赖 (Tq, Tkv)，确定性，
        # 逐 token 解码时 Tkv 单调增长，缓存避免每步 arange + (1,1,Tq,Tkv) 张量分配
        # （DML 小算子启动税敏感）。仅 attn_mask 为 None（无窗口/记忆/alibi/rel_bias）时命中。
        self._causal_key: Optional[tuple] = None
        self._causal_cache: Optional[torch.Tensor] = None

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

    def _alibi_bias(self, Tq: int, Tkv: int, device: torch.device, start_pos: int = 0,
                    mem_cols: int = 0) -> Optional[torch.Tensor]:
        """ALiBi 线性位置偏置：(1, H, Tq, Tkv)，bias[h,i,j] = -m_h * |i-j|。

        start_pos 为增量解码时当前窗口首 token 的绝对位置，必须传入，
        否则缓存路径会把每个查询当成序列第 0 位、造成训练-推理位置偏移。

        mem_cols：记忆列（前 mem_cols 列）是位置无关的压缩历史，不受位置距离偏置影响；
        显式清零，避免生成时 start_pos 增长使记忆列被强负偏置逐步压制（训练-推理不一致）。
        """
        if not self.alibi:
            return None
        qpos = torch.arange(start_pos, start_pos + Tq, device=device).unsqueeze(1)
        kpos = torch.arange(0, Tkv, device=device).unsqueeze(0)
        dist = (qpos - kpos).abs().to(device)
        # slopes: (H,) -> (1,H,1,1)，乘以距离 -> (1,H,Tq,Tkv)
        bias = -self.alibi_slopes.view(1, self.num_heads, 1, 1).to(device) * dist.unsqueeze(0).unsqueeze(0)
        if mem_cols > 0:
            bias = bias.clone()
            bias[..., :mem_cols] = 0
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
            # 与 inject_memory 的 mem_bias 一致：retrieval_gate 经 sigmoid→(0,1) 软门控，
            # 不能把原始 Parameter（含 0 初始化）直接乘（会令全上下文检索偏置整体清零、与
            # 记忆段偏置语义/尺度脱节）。
            gate = torch.sigmoid(gate)
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
        # 因果：未来位置本就被 attn_mask 屏蔽，这里也压到 -inf 不参与检索。
        # 注意因果掩码须是 (Tq, Treal) 的逐查询掩码（query i 只看 key j<=i），不能写成
        # (Treal,Treal) 的方阵——当 Tq≠Treal（如增量/不同长度）会形状不符崩溃；统一用
        # qpos/kpos 构造与上方 keep 掩码同源，避免维度假设。
        qpos_q = torch.arange(q.size(2), device=device).unsqueeze(1)  # (Tq, 1)
        kpos_r = torch.arange(Treal, device=device).unsqueeze(0)      # (1, Treal)
        causal = (kpos_r > qpos_q)                                    # (Tq, Treal)
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
        # ① QK-Norm + ⑤ 可学习温度（共享预处理，见 apply_qk_norm_and_temp）
        q, k = apply_qk_norm_and_temp(
            q, k, self._rt,
            self.qk_norm if self.qk_norm_enabled else None,
            self.log_temp if self.attn_temp_enabled else None)
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
        # 阶段3 可学习检索：统一经 MemoryBank.inject_memory 注入记忆 K/V + 检索偏置，
        # 取代 cache/全量两条路径各自重复的"记忆拼接 + 稀疏门控 + 全上下文检索"逻辑（B 项收敛）。
        mem_cols = 0
        mem_bias: Optional[torch.Tensor] = None
        rbias_full: Optional[torch.Tensor] = None
        k_orig_cols = k.size(2)  # 记忆拼接前的真实序列 KV 长度
        if memory_kv is not None:
            mk, mv, meta = memory_kv
            mem_cols = mk.size(1)
            k, v, mem_bias = MemoryBank.inject_memory(
                q, k, v, mk, mv, meta, self.mask_fill_value)
            # 全上下文检索偏置（阶段3 扩展）：对真实 KV 远端做稀疏检索，注入为注意力正偏置。
            # 由实例方法计算以复用本层的 window/topk 开关（与两路径历史上各自实现同源一致）。
            rbias_full = self._full_retrieval_bias(q, k, k_orig_cols, mem_cols,
                                                   meta.get('retrieval_gate') if meta else None,
                                                   dev)

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
            Tkv = k.size(2)
            # 与全量路径共用基础因果/窗口掩码（额外1），保证 memory+window>0 时
            # 训练/推理一致性（否则推理期记忆按位置被部分遮蔽、静默质量退化）。
            attn_mask = self._build_causal_window_mask(Tq, Tkv, mem_cols, dev, start_pos)
            # cache 路径始终需要显式掩码（单步/整段解码都靠它施加因果，不能用 is_causal 快捷）：
            # 纯因果（无窗口/记忆/alibi）时退化为主序列因果掩码。
            if attn_mask is None:
                attn_mask = self._cached_causal_mask(Tq, Tkv, dev, start_pos)
            # 记忆槽位置在窗口 KV 之前（seq 起点之前），永远不被因果遮蔽，
            # 但也不参与"未来"泄露：记忆是历史压缩，视为已发生，不施加 causal 惩罚
            if self.rel_bias:
                # qpos/kpos 必须在本作用域构造（非缓存路径它们在 _build_causal_window_mask
                # 内，缓存在 attend 作用域外不可见）；增量解码 qpos 从 start_pos 起。
                qpos = torch.arange(start_pos, start_pos + Tq, device=dev).unsqueeze(1)
                kpos = torch.arange(0, Tkv, device=dev).unsqueeze(0)
                idx = (qpos - kpos + Tkv - 1).clamp(0, 2 * self.max_seq_length - 1)
                attn_mask = attn_mask + self.rel_bias_table[:, idx].unsqueeze(0)
            if mem_bias is not None:
                # mem_bias: (B,H,Tq,mem_cols)，右侧补零到 Tkv 再与 attn_mask 广播相加
                padded = torch.nn.functional.pad(mem_bias, (0, Tkv - mem_cols))
                attn_mask = attn_mask + padded
            alibi_b = self._alibi_bias(Tq, Tkv, dev, start_pos, mem_cols=mem_cols)
            if alibi_b is not None:
                attn_mask = attn_mask + alibi_b
            # 全上下文检索：inject_memory 已统一算好 rbias_full（cache 与全量路径同源），
            # 否则开启 retrieval_full 时训练-推理系统性不一致（生成质量偏离训练行为）。
            if rbias_full is not None:
                attn_mask = attn_mask + rbias_full
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
        Treal = k_orig_cols  # 真实序列 KV 长度（记忆已在上方面经 inject_memory 拼接，此处仅作语义记录）
        # 统一构造 (1,1,T,Tkv) 注意力掩码：记忆段全 0（全局可检索），
        # 主序列段按 causal / window / rel_bias 遮蔽
        # 静态部分（窗口/因果掩码）仅依赖 (T, Tkv, mem_cols)，缓存复用避免每步每头重建
        # （基础因果/窗口掩码经 _build_causal_window_mask 与 cache 路径共用，额外1）
        cache_key = (T, Tkv, mem_cols)
        if self._bias_key != cache_key or self._bias_cache is None or self._bias_cache.device != dev:
            raw_mask = self._build_causal_window_mask(T, Tkv, mem_cols, dev, 0)
            base = raw_mask if raw_mask is not None else torch.zeros(1, 1, T, Tkv, device=dev)
            if self.rel_bias:
                # 绝对位置相对偏置表（rel_bias 路径必须显式带因果掩码，不能退回 is_causal 快捷）。
                # 注意 KV 长度 Tkv = T + mem_cols（记忆列已拼到前面），self._mask 是 (T,T) 与
                # base (1,1,T,Tkv) 维度不符（记忆开启时越界崩溃），故此处直接用 _build_causal_window_mask
                # 构造含记忆列的基础掩码（记忆列恒 0，全局可检索），再叠加相对偏置表。
                if raw_mask is None:
                    # 纯因果（无窗口/记忆/alibi）：显式构造因果掩码，保证 rel_bias 开启时仍有因果
                    qp = torch.arange(0, T, device=dev).unsqueeze(1)
                    kp = torch.arange(0, Tkv, device=dev).unsqueeze(0)
                    base = ((kp > qp).float() * self.mask_fill_value).unsqueeze(0).unsqueeze(0)
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
        alibi_b = self._alibi_bias(T, Tkv, dev, start_pos, mem_cols=mem_cols)
        if alibi_b is not None:
            attn_mask = attn_mask + alibi_b
        # 全上下文检索：inject_memory 已统一算好 rbias_full（与 cache 路径同源一致）
        if rbias_full is not None:
            attn_mask = attn_mask + rbias_full
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

    def _cached_causal_mask(self, Tq: int, Tkv: int, dev: torch.device,
                             start_pos: int) -> torch.Tensor:
        """纯因果（无窗口/记忆/alibi/rel_bias）增量解码掩码 (1,1,Tq,Tkv)，带缓存。

        掩码仅依赖 (Tq, Tkv)（确定性），逐 token 解码 Tkv 单调增长；缓存避免每步
        重建 arange + (1,1,Tq,Tkv) 张量（DML 小算子启动税敏感）。语义与原始
        `(kpos > qpos) * mask_fill` 完全一致。"""
        key = (Tq, Tkv)
        if self._causal_key == key and self._causal_cache is not None \
                and self._causal_cache.device == dev:
            return self._causal_cache
        qpos = torch.arange(start_pos, start_pos + Tq, device=dev).unsqueeze(1)
        kpos = torch.arange(0, Tkv, device=dev).unsqueeze(0)
        causal = (kpos > qpos).float() * self.mask_fill_value
        self._causal_cache = causal.unsqueeze(0).unsqueeze(0)  # (1,1,Tq,Tkv)
        self._causal_key = key
        return self._causal_cache

    def _build_causal_window_mask(self, T: int, Tkv: int, mem_cols: int,
                                   dev: torch.device, start_pos: int) -> Optional[torch.Tensor]:
        """构造因果 + 滑动窗口基础掩码 (1,1,T,Tkv)，记忆段（前 mem_cols 列）恒全 0（全局可检索）。

        供 attend 的 cache / 全量两条路径共用，消除两路径各自重复实现而漂移的风险
        （额外1）。rel_bias/ALiBi/mem_bias/rbias 等附加偏置由各路径在返回后单独叠加。
        纯因果（window==0 且 mem_cols==0 且非 alibi）返回 None，交给 SDPA is_causal / manual 兜底。
        """
        if self.window > 0:
            qpos = torch.arange(start_pos, start_pos + T, device=dev).unsqueeze(1)
            kpos = torch.arange(0, Tkv, device=dev).unsqueeze(0)
            mask = (kpos > (qpos + mem_cols)) | (qpos - kpos > self.window)
            if mem_cols > 0:
                mask[..., :mem_cols] = False
            return (mask.float() * self.mask_fill_value).unsqueeze(0).unsqueeze(0)
        if mem_cols > 0 or self.alibi:
            qpos = torch.arange(start_pos, start_pos + T, device=dev).unsqueeze(1)
            kpos = torch.arange(0, Tkv, device=dev).unsqueeze(0)
            mask = (kpos > (qpos + mem_cols))
            if mem_cols > 0:
                mask[..., :mem_cols] = False
            return (mask.float() * self.mask_fill_value).unsqueeze(0).unsqueeze(0)
        return None

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
                 max_seq_length: int = 64, feature: str = 'relu', head_dim: Optional[int] = None,
                 rope_learnable: bool = False,
                 shared_qkv: Optional[nn.Linear] = None, shared_proj: Optional[nn.Linear] = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.max_seq_length = max_seq_length
        self.feature = feature
        # 层间共享：传入 shared_qkv/shared_proj 时复用外部投影
        self.qkv = shared_qkv if shared_qkv is not None else nn.Linear(dim, 3 * self.num_heads * self.head_dim, bias=False)
        self.proj = shared_proj if shared_proj is not None else nn.Linear(self.num_heads * self.head_dim, dim, bias=False)
        # 与 attn 分支的 RotaryEmbedding 保持同一 rope_learnable 配置，
        # 避免 attn_linear 混合块内两路 RoPE 静默不一致（仅 rope_learnable=True 时显式分叉）。
        self.rope = RotaryEmbedding(self.head_dim, learnable=rope_learnable)
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
        # ① QK-Norm + ⑤ 可学习温度（与 SlidingWindowCausalSelfAttention 共享预处理）
        q, k = apply_qk_norm_and_temp(
            q, k, self._rt,
            self.qk_norm if self.qk_norm_enabled else None,
            self.log_temp if self.attn_temp_enabled else None)
        q, k = self.rope(q, k, start_pos=start_pos, max_len=self.max_seq_length)
        return q, k, v

    def forward(self, x: torch.Tensor, past_kv=None, use_cache: bool = False, start_pos: int = 0,
                memory_kv=None):
        # 线性注意力：全量路径用 cumsum 向量化（O(T·D²) 内存，T≤64 安全），
        # 增量解码路径用 RNN 逐 token 累积（O(D²) 内存）。
        q, k, v = self.project_and_norm(x, start_pos)
        B, H, T, D = q.shape
        qf = self._feat(q)
        kf = self._feat(k)

        if use_cache and past_kv is not None and len(past_kv) >= 4 and past_kv[2] is not None:
            # 增量解码：RNN 逐 token（T=1）
            S = past_kv[2]
            z = past_kv[3]
            kf_t = kf[:, :, 0, :]
            v_t = v[:, :, 0, :]
            S = S + torch.einsum('bhd,bhe->bhde', kf_t, v_t)
            z = z + kf_t
            num_t = torch.einsum('bhd,bhde->bhe', qf[:, :, 0, :], S)
            den_t = torch.einsum('bhd,bhd->bh', qf[:, :, 0, :], z).unsqueeze(-1).clamp_min(1e-6)
            out = self.proj((num_t / den_t).transpose(1, 2).reshape(B, 1, H * D))
            present = (k, v, S, z)
            return out, present

        # 全量路径：cumsum 向量化
        kv_all = torch.einsum('bhtd,bhte->bhtde', kf, v)  # (B,H,T,D,D)
        S_all = torch.cumsum(kv_all, dim=2)                 # (B,H,T,D,D)
        z_all = torch.cumsum(kf, dim=2)                     # (B,H,T,D)
        num = torch.einsum('bhtd,bhtde->bhte', qf, S_all)  # (B,H,T,D)
        den = torch.einsum('bhtd,bhtd->bht', qf, z_all).unsqueeze(-1).clamp_min(1e-6)
        out = num / den                                     # (B,H,T,D)
        out = out.transpose(1, 2).reshape(B, T, H * D)
        present = (k, v, S_all[:, :, -1], z_all[:, :, -1]) if use_cache else None
        return self.proj(out), present


class AxialLinearAttention(nn.Module):
    """2D 轴向线性注意力：将 1D 序列视为 row×col 网格，先行后列各做线性注意力，
    输出加权融合。复杂度 O(T·√T)，兼顾效率与 2D 空间归纳偏置。

    与 LinearAttention 同接口（project_and_norm + attend），可直接替换为 mixer='linear2d'。
    支持 KV-cache：增量解码时仅处理单 token，不展开网格（退化为等效 1D 线性注意力）。

    参数:
        grid_size: (row, col) 网格尺寸。为 None 时自动取最接近的整数平方根。
        gate_init: 行/列融合门控初始偏置（>0 偏向列，<0 偏向行）。
    """

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, attn_temp: bool = True,
                 max_seq_length: int = 64, feature: str = 'relu', head_dim: Optional[int] = None,
                 rope_learnable: bool = False, grid_size: Optional[Tuple[int, int]] = None,
                 gate_init: float = 0.0,
                 shared_qkv: Optional[nn.Linear] = None, shared_proj: Optional[nn.Linear] = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.max_seq_length = max_seq_length
        self.feature = feature
        self.grid_size = grid_size  # (row, col) or None for auto
        # 共享/独立 QKV/Output 投影：main + row 共享（4 个 Linear，而非 6 个）
        # main（增量解码）和 row 轴共享同一组投影；col 轴独立投影。
        self.qkv = shared_qkv if shared_qkv is not None else nn.Linear(dim, 3 * self.num_heads * self.head_dim, bias=False)
        self.proj = shared_proj if shared_proj is not None else nn.Linear(self.num_heads * self.head_dim, dim, bias=False)
        # row 轴复用 main 投影；col 轴独立投影
        self.qkv_col = nn.Linear(dim, 3 * self.num_heads * self.head_dim, bias=False)
        self.proj_col = nn.Linear(self.num_heads * self.head_dim, dim, bias=False)
        # 行/列融合门控：sigmoid(gate) * row_out + (1-sigmoid(gate)) * col_out
        self.gate = nn.Parameter(torch.tensor(gate_init))
        # RoPE / QK-Norm / Temp（与 LinearAttention 一致）
        self.rope = RotaryEmbedding(self.head_dim, learnable=rope_learnable)
        self.qk_norm_enabled = qk_norm
        if qk_norm:
            self.qk_norm = RMSNorm(self.head_dim)
            self.qk_norm_col = RMSNorm(self.head_dim)
        self.attn_temp_enabled = attn_temp
        if attn_temp:
            self.log_temp = nn.Parameter(torch.zeros(1))
            self.log_temp_col = nn.Parameter(torch.zeros(1))
        self._rt: Dict[str, bool] = {"qk_norm": True, "attn_temp": True}

    def _feat(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature == 'elu':
            return torch.nn.functional.elu(x) + 1.0
        return torch.nn.functional.relu(x) + 1e-6

    def _infer_grid(self, T: int) -> Tuple[int, int]:
        if self.grid_size is not None:
            return self.grid_size
        import math
        col = int(math.ceil(math.sqrt(T)))
        row = math.ceil(T / col)
        return row, col

    def _linear_attn_1d(self, q, k, v, qk_norm_mod=None, log_temp_mod=None):
        """单轴线性注意力：cumsum 向量化（全量路径），返回 (B,H,T,D)。"""
        B, H, T, D = q.shape
        if qk_norm_mod is not None:
            q = qk_norm_mod(q)
            k = qk_norm_mod(k)
        if log_temp_mod is not None:
            q = q * torch.exp(log_temp_mod)
            k = k * torch.exp(log_temp_mod)
        qf = self._feat(q)
        kf = self._feat(k)
        kv_all = torch.einsum('bhtd,bhte->bhtde', kf, v)
        S_all = torch.cumsum(kv_all, dim=2)
        z_all = torch.cumsum(kf, dim=2)
        num = torch.einsum('bhtd,bhtde->bhte', qf, S_all)
        den = torch.einsum('bhtd,bhtd->bht', qf, z_all).unsqueeze(-1).clamp_min(1e-6)
        return num / den  # (B,H,T,D)

    def _linear_attn_1d_rnn(self, q_t, k_t, v_t, S, z, qk_norm_mod=None, log_temp_mod=None):
        """增量解码（T=1）RNN 路径。"""
        if qk_norm_mod is not None:
            q_t = qk_norm_mod(q_t)
            k_t = qk_norm_mod(k_t)
        if log_temp_mod is not None:
            q_t = q_t * torch.exp(log_temp_mod)
            k_t = k_t * torch.exp(log_temp_mod)
        qf = self._feat(q_t)
        kf = self._feat(k_t)
        S = S + torch.einsum('bhd,bhe->bhde', kf[:, :, 0, :], v_t[:, :, 0, :])
        z = z + kf[:, :, 0, :]
        num = torch.einsum('bhd,bhde->bhe', qf[:, :, 0, :], S)
        den = torch.einsum('bhd,bhd->bh', qf[:, :, 0, :], z).unsqueeze(-1).clamp_min(1e-6)
        return num / den, S, z  # (B,H,D), S, z

    def project_and_norm(self, x: torch.Tensor, start_pos: int = 0):
        # 主投影（用于增量解码或非轴向模式），轴向模式下由 _axial_forward 内部使用
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_qk_norm_and_temp(
            q, k, self._rt,
            self.qk_norm if self.qk_norm_enabled else None,
            self.log_temp if self.attn_temp_enabled else None)
        q, k = self.rope(q, k, start_pos=start_pos, max_len=self.max_seq_length)
        return q, k, v

    def _axial_forward(self, x: torch.Tensor, use_cache: bool = False, start_pos: int = 0):
        """轴向 2D 线性注意力：reshpe 1D → 2D → 行注意力 → 列注意力 → 融合 → reshape → proj。"""
        B, T, C = x.shape
        row, col = self._infer_grid(T)
        pad_len = row * col - T
        if pad_len > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))  # 补零到 row*col
        x2d = x.reshape(B, row, col, C)

        # ── 行注意力：沿 col 维度做线性注意力（复用 main 投影） ──
        x_row = x2d.reshape(B * row, col, C)
        qkv_r = self.qkv(x_row).reshape(B * row, col, 3, self.num_heads, self.head_dim)
        qkv_r = qkv_r.permute(2, 0, 3, 1, 4)
        qr, kr, vr = qkv_r[0], qkv_r[1], qkv_r[2]
        qr, kr = apply_qk_norm_and_temp(
            qr, kr, self._rt,
            self.qk_norm if self.qk_norm_enabled else None,
            self.log_temp if self.attn_temp_enabled else None)
        hr = self._linear_attn_1d(qr, kr, vr,
                                   qk_norm_mod=None, log_temp_mod=None)  # 已在上面处理
        out_row = self.proj(hr.transpose(1, 2).reshape(B * row, col, self.num_heads * self.head_dim))
        out_row = out_row.reshape(B, row, col, C)

        # ── 列注意力：沿 row 维度做线性注意力（独立投影） ──
        x_col = x2d.permute(0, 2, 1, 3).reshape(B * col, row, C)  # (B*col, row, C)
        qkv_c = self.qkv_col(x_col).reshape(B * col, row, 3, self.num_heads, self.head_dim)
        qkv_c = qkv_c.permute(2, 0, 3, 1, 4)
        qc, kc, vc = qkv_c[0], qkv_c[1], qkv_c[2]
        qc, kc = apply_qk_norm_and_temp(
            qc, kc, self._rt,
            self.qk_norm_col if self.qk_norm_enabled else None,
            self.log_temp_col if self.attn_temp_enabled else None)
        hc = self._linear_attn_1d(qc, kc, vc,
                                   qk_norm_mod=None, log_temp_mod=None)
        out_col = self.proj_col(hc.transpose(1, 2).reshape(B * col, row, self.num_heads * self.head_dim))
        out_col = out_col.reshape(B, col, row, C).permute(0, 2, 1, 3)  # → (B, row, col, C)

        # ── 融合 ──
        g = torch.sigmoid(self.gate)
        fused = g * out_row + (1 - g) * out_col  # (B, row, col, C)
        fused = fused.reshape(B, row * col, C)[:, :T, :]  # 去掉 padding
        return fused  # (B, T, C)

    def forward(self, x: torch.Tensor, past_kv=None, use_cache: bool = False, start_pos: int = 0,
                memory_kv=None):
        B, T, C = x.shape
        # 增量解码（T=1）：退化为等效 1D 线性注意力（不展开网格）
        if use_cache and T == 1:
            q, k, v = self.project_and_norm(x, start_pos)
            H, D = self.num_heads, self.head_dim
            if past_kv is not None and len(past_kv) >= 4 and past_kv[2] is not None:
                S, z = past_kv[2], past_kv[3]
                kf_t = self._feat(k[:, :, 0, :])
                v_t = v[:, :, 0, :]
                S = S + torch.einsum('bhd,bhe->bhde', kf_t, v_t)
                z = z + kf_t
                qf_t = self._feat(q[:, :, 0, :])
                num = torch.einsum('bhd,bhde->bhe', qf_t, S)
                den = torch.einsum('bhd,bhd->bh', qf_t, z).unsqueeze(-1).clamp_min(1e-6)
                out = self.proj((num / den).transpose(1, 2).reshape(B, 1, H * D))
                return out, (k, v, S, z)
            # 首步：初始化 S/z
            qf = self._feat(q)
            kf = self._feat(k)
            S = torch.einsum('bhd,bhe->bhde', kf[:, :, 0, :], v[:, :, 0, :])
            z = kf[:, :, 0, :]
            num = torch.einsum('bhd,bhde->bhe', qf[:, :, 0, :], S)
            den = torch.einsum('bhd,bhd->bh', qf[:, :, 0, :], z).unsqueeze(-1).clamp_min(1e-6)
            out = self.proj((num / den).transpose(1, 2).reshape(B, 1, H * D))
            return out, (k, v, S, z)
        # 全量路径：轴向 2D 线性注意力
        fused = self._axial_forward(x, use_cache=False, start_pos=start_pos)
        return self.proj(fused), None


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
        # 融合 dB * x_conv 为单次运算，避免 (B,L,d_inner,d_state) 中间张量分配
        xb = (dt.unsqueeze(-1) * Bp.unsqueeze(2)) * x_conv.unsqueeze(-1)  # (B, L, d_inner, d_state)
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
        # Standard parallel prefix scan (Hillis-Steele) assuming h_0 = 0
        # 训练时用普通运算（支持 autograd），推理时用预分配缓冲区（减少分配开销）
        if a.requires_grad:
            A, B = a, b
            offset = 1
            while offset < L:
                A_prev = A.roll(offset, dims=1)
                A_prev[:, :offset] = 1.0
                B_prev = B.roll(offset, dims=1)
                B_prev[:, :offset] = 0.0
                A, B = A_prev * A, A * B_prev + B
                offset <<= 1
        else:
            A = a.clone()
            B = b.clone()
            A_prev = torch.empty_like(A)
            B_prev = torch.empty_like(B)
            A_new = torch.empty_like(A)
            B_new = torch.empty_like(B)
            offset = 1
            while offset < L:
                A_prev.copy_(A.roll(offset, dims=1))
                A_prev[:, :offset] = 1.0
                B_prev.copy_(B.roll(offset, dims=1))
                B_prev[:, :offset] = 0.0
                torch.mul(A_prev, A, out=A_new)
                torch.mul(A, B_prev, out=B_new)
                B_new.add_(B)
                A.copy_(A_new)
                B.copy_(B_new)
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
