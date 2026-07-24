from __future__ import annotations
import math
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


class EnhancementsMixin:
    """运行时增强开关 mixin（set_enhancements_active 的单一事实来源）。

    SlidingWindowCausalSelfAttention / LinearMixerBase / DifferentialAttention
    三类原各自重复实现同一逻辑，统一到此处消除漂移风险。要求宿主类初始化 self._rt: Dict[str, bool]。
    """

    def set_enhancements_active(self, spec):
        """`spec=True/False` 全开/全关；`spec=dict` 仅更新存在的键。
        用于"交替/分段增强"训练，关闭时跳过对应 QK-Norm/可学习温度（恒等）。"""
        if isinstance(spec, bool):
            on = spec
            self._rt = {"qk_norm": on, "attn_temp": on}
        elif isinstance(spec, dict):
            for k, v in spec.items():
                if k in self._rt:
                    self._rt[k] = bool(v)
        else:
            raise TypeError(f"set_enhancements_active 期望 bool 或 dict，收到 {type(spec)}")


def _pad_mem_bias(mem_bias: torch.Tensor, Tkv: int, mem_cols: int) -> torch.Tensor:
    """记忆段偏置右补零到完整 KV 长度，供与 scores/attn_mask 广播相加。

    mem_bias: (B,H,Tq,mem_cols) → (B,H,Tq,Tkv)。记忆段保留原值，主序列段补 0。
    4 处调用点（attend cache/全量 + DifferentialAttention 双路）共用，消除 F.pad 散落。"""
    return torch.nn.functional.pad(mem_bias, (0, Tkv - mem_cols))


def _parallel_prefix_scan(
    a: torch.Tensor, b: torch.Tensor,
    past_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Hillis-Steele 并行前缀扫描：h_t = a_t * h_{t-1} + b_t。

    a, b: (B, L, d_inner, d_state)。返回 h: (B, L, d_inner, d_state)。
    半群 (A, B)⊙(A', B') = (A·A', A'·B + B') 满足结合律；
    Hillis-Steele 含扫描：每轮把左邻 2^k 步的变换合并进来，offset 从 1 翻倍到 <L。
    单位元为 (A=1, B=0)，越界位置用单位元填充。

    如果提供 past_state (B, d_inner, d_state)，将其作为 h_{-1} 用于计算 h_0 = a_0 * past_state + b_0。
    """
    L = a.shape[1]
    # DML 兼容：roll 回退 CPU（aten::roll 不支持 DML），切片赋值触发 scatter 错误。
    # 用 cat 替代 roll（DML 原生）+ torch.where 替代切片赋值（DML 原生，无 scatter）。
    pos_idx = torch.arange(L, device=a.device)
    if a.requires_grad:
        A, B = a, b
        offset = 1
        while offset < L:
            mask = (pos_idx < offset).view(1, L, 1, 1)
            # roll(offset, dims=1) 等价于 cat([后 offset 位, 前 L-offset 位], dim=1)
            A_prev = torch.where(mask, torch.ones_like(A),
                                 torch.cat([A[:, -offset:], A[:, :-offset]], dim=1))
            B_prev = torch.where(mask, torch.zeros_like(B),
                                 torch.cat([B[:, -offset:], B[:, :-offset]], dim=1))
            A, B = A_prev * A, A * B_prev + B
            offset <<= 1
    else:
        A = a.clone()
        B = b.clone()
        offset = 1
        while offset < L:
            mask = (pos_idx < offset).view(1, L, 1, 1)
            A_prev = torch.where(mask, torch.ones_like(A),
                                 torch.cat([A[:, -offset:], A[:, :-offset]], dim=1))
            B_prev = torch.where(mask, torch.zeros_like(B),
                                 torch.cat([B[:, -offset:], B[:, :-offset]], dim=1))
            A, B = A_prev * A, A * B_prev + B
            offset <<= 1
    if past_state is not None:
        past_expanded = past_state.unsqueeze(1).expand(-1, L, -1, -1)
        B = B + A * past_expanded
    return B


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
        # 第二十一轮：per-head 温度——log_temp (num_heads,) 时 scale 需 view (1,H,1,1) 广播
        # (1,) 全局标量时保持标量广播（向后兼容）
        if scale.dim() == 1 and scale.numel() > 1:
            scale = scale.view(1, -1, 1, 1)
        q = q * scale
        k = k * scale
    return q, k

class SlidingWindowCausalSelfAttention(nn.Module, EnhancementsMixin):
    """因果自注意力，可选滑动窗口 + 可学习相对位置偏置。
     全后端统一用 fused SDPA（DML fused SDPA 现已稳定且比 manual 快 ~2.5x）；
     DML 的 bool attn_mask 语义与 PyTorch 标准相反（True=允许≠禁止），故全路径用 float attn_mask
     （_build_causal_window_mask 返回 mask.float()*fill_value）；纯因果路径用 is_causal=True 更高效。
    """
    def __init__(self, dim: int, num_heads: int, window: int = 0, rel_bias: bool = False, max_seq_length: int = 64,
                 qk_norm: bool = True, attn_temp: bool = True, mask_fill_value: float = MASK_FILL_VALUE,
                 rope_learnable: bool = False, alibi: bool = False, retrieval_full: bool = False,
                 retrieval_topk: int = 32, learn_window: bool = False, window_base: int = 64,
                 shared_qkv: Optional[nn.Linear] = None, shared_proj: Optional[nn.Linear] = None,
                 pe_gate: bool = False, rope_dim_fraction: float = 1.0,
                 output_gate: bool = False,
                 use_mla_kv: bool = False, kv_latent_dim: Optional[int] = None,
                 use_rope: bool = True,
                 yarn_scale: float = 1.0, yarn_beta: float = 0.1, yarn_orig_max_seq_length: int = 0,
                 dim_wise_rope: bool = False,
                 head_temp: bool = False,
                 value_relative_coding: bool = False,
                 intra_hybrid_rope: bool = False,
                 intra_hybrid_ratio: float = 0.5,
                 alibi_learnable: bool = False):
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
        self.rope = RotaryEmbedding(self.head_dim, learnable=rope_learnable, dim_fraction=rope_dim_fraction,
                                    yarn_scale=yarn_scale, yarn_beta=yarn_beta,
                                    yarn_orig_max_seq_length=yarn_orig_max_seq_length,
                                    dim_wise=dim_wise_rope)
        if self.rel_bias:
            # T5 风格相对位置偏置表：(heads, 2T-1)
            self.rel_bias_table = nn.Parameter(torch.zeros(num_heads, 2 * max_seq_length - 1))
        # 阶段5：ALiBi 线性位置偏置——对距离线性惩罚，长度外推极稳，与 RoPE 互补。
        # 每个头一个斜率 m_h = 2^(-h/H * 8)，bias = -m_h * |i-j|（注入 attn scores 前）。
        if alibi:
            # 头斜率：默认固定（buffer，符合 ALiBi 原设计）；alibi_learnable=True 时升级为
            # 可学 Parameter（per-head 自由学习最优位置衰减模式，打破原几何级数先验）。
            # 初始值仍为 m_h = 2^(-h/H * 8)，精确向后兼容；与 shared_alibi 正交兼容
            # （shared_alibi 共享第一层 Parameter 对象，所有层共用同一组可学斜率，减参+一致）。
            # 注意：alibi_slopes 既是 buffer 又可能是 Parameter，下游代码须用 .view() 等只读操作
            # （不可 in-place 修改，会破坏 Parameter 的 autograd）。
            m = torch.tensor([2.0 ** (-(h + 1) / num_heads * 8.0) for h in range(num_heads)])
            self.alibi_learnable = alibi_learnable
            if alibi_learnable:
                self.alibi_slopes = nn.Parameter(m)
            else:
                self.register_buffer('alibi_slopes', m, persistent=False)
        else:
            self.alibi_learnable = False
        # 第十一轮：位置编码选择性门控——per-head 可学强度控制 ALiBi 位置偏置。
        # pe_strength = 1.0 + tanh(log_pe_gate)，init 0 → 1.0（精确向后兼容），范围 (0,2)。
        # 让模型自决每个头对位置信息的依赖：某些头更靠内容、某些头更靠位置。
        # 当前仅作用于 ALiBi（每次前向重算，梯度正确回流）；rel_bias 因掩码
        # 缓存机制暂不支持 pe_gate（避免缓存导致梯度截断）。
        self.pe_gate_enabled = pe_gate and alibi
        if self.pe_gate_enabled:
            self.log_pe_gate = nn.Parameter(torch.zeros(num_heads))
        # 第十五轮：Output Gating——注意力输出后加门控 sigmoid(W·out)，消除 Attention Sink。
        # 灵感：Qwen3-Next Output Gating。init W=0, b=0 → sigmoid=0.5（半通），训练中自决。
        # 默认关（向后兼容），config 显式开启。
        self.output_gate_enabled = output_gate
        if self.output_gate_enabled:
            self.output_gate = nn.Linear(dim, dim, bias=True)
            nn.init.zeros_(self.output_gate.weight)
            nn.init.zeros_(self.output_gate.bias)
        # 第十七轮：MLA 风格 KV 潜空间压缩——K/V 共享下投影到潜空间 c_kv，cache 只存潜向量，
        # attend 时上投影还原 K/V。长序列 KV cache 内存降 2*dim/kv_latent_dim 倍。
        # 灵感：DeepSeek-V3 MLA。结合项目：与 MemoryBank 的 compress/decompress 抽象思路一致
        # （MemoryBank 压缩固定槽，MLA 压缩序列 KV），两者正交可叠加。
        # RoPE 处理：MLA 时 q 在 project_and_norm 应用 RoPE（位置 start_pos），
        # k 在 attend 内部还原后应用 RoPE（位置 0..T_total-1，拼接后），保证 RoPE 旋转信息
        # 不被压缩-还原破坏（与 DeepSeek-V3 decoupled RoPE 思路一致，但简化为单段压缩）。
        # 默认关（向后兼容），config 显式开启。
        self.mla_kv_enabled = use_mla_kv
        # 第十八轮：iRoPE 交错 NoPE 层——use_rope=False 时跳过 RoPE 应用，
        # 位置信号由 ALiBi 提供（NoPE 层须强制 alibi=True）。
        # 灵感：LLaMA 4 iRoPE（3:1 交错 RoPE/NoPE）。NoPE 层理论上能学到任意长度外推。
        # 默认 True（向后兼容），config 通过 nope_layers 指定哪些层关闭 RoPE。
        self.use_rope = use_rope
        if self.mla_kv_enabled:
            # kv_latent_dim 默认 = dim（压缩 2x：2*dim K/V → dim 潜向量）
            _kv_latent = kv_latent_dim if kv_latent_dim is not None else dim
            self.kv_latent_dim = _kv_latent
            # 下投影：cat([k, v]) (2*dim) → 潜空间 (kv_latent_dim)
            self.kv_compress = nn.Linear(2 * dim, _kv_latent, bias=False)
            # 上投影：潜空间 → K+V (2*num_heads*head_dim = 2*dim)，单次 GEMM 合并
            # （原 kv_decompress_k/kv_decompress_v 两次 GEMM → 合并为一次，DML 启动税敏感）
            self.kv_decompress = nn.Linear(_kv_latent, 2 * num_heads * self.head_dim, bias=False)
        # ① QK-Norm：对 Q/K 各自做 RMSNorm 后再进注意力，与 RoPE 互补、稳定训练（默认开）
        self.qk_norm_enabled = qk_norm
        if qk_norm:
            self.qk_norm = RMSNorm(self.head_dim)
        # ⑤ 可学习注意力温度：softmax(score / T)，T=exp(log_temp) 恒正（默认开）
        self.temp_enabled = attn_temp
        # 第二十一轮：per-head 可学注意力温度——log_temp 从 (1,) 升级为 (num_heads,)
        # 灵感：NoPE 长度外推（arXiv:2404.12224）——NoPE 层注意力熵分散致外推失败，
        # per-head 温度让每个头独立控制 softmax 聚焦度。init 0 → 温度=1（向后兼容）。
        # value_relative_coding：v += tanh(λ)·v_{t-1}，轻量相对位置信号（init λ=0 → 不编码）
        self.head_temp_enabled = head_temp and attn_temp
        if attn_temp:
            self.log_temp = nn.Parameter(torch.zeros(num_heads if head_temp else 1))
        self.value_relative_coding_enabled = value_relative_coding
        if value_relative_coding:
            # tanh 限制 λ∈(-1,1)，init 0 → v 不变；cache 存原始 v 保证 train/infer parity
            self.value_rel_lambda = nn.Parameter(torch.zeros(1))
        # 第二十二轮：层内 head 拆半 RoPE/NoPE——同一层内前半 head 用 RoPE（位置精确匹配），
        # 后半 head 用 NoPE（内容语义+长度外推，靠 ALiBi 获位置信号）。
        # 灵感：LLaMA 4 iRoPE 层间交错的层内版 + HARoPE head-wise PE。
        # 与 nope_layers（层间交错）正交：nope_layers 整层关 RoPE，intra_hybrid 层内拆半。
        # DML 零额外开销（仅 split+cat，比 RoPE 本身的三角函数计算还轻）。
        # 默认关（向后兼容），config 显式开启。NoPE half 须与 alibi=True 组合。
        self.intra_hybrid_rope_enabled = intra_hybrid_rope
        if intra_hybrid_rope:
            if num_heads < 2:
                raise ValueError(
                    f"intra_hybrid_rope=True 须 num_heads >= 2（当前 num_heads={num_heads}，无法拆半）")
            self.intra_hybrid_nope_heads = max(1, int(num_heads * intra_hybrid_ratio))
            if self.intra_hybrid_nope_heads >= num_heads:
                raise ValueError(
                    f"intra_hybrid_ratio={intra_hybrid_ratio} 致 nope_heads="
                    f"{self.intra_hybrid_nope_heads} >= num_heads={num_heads}，无法拆半"
                    f"（请减小 ratio 或增加 num_heads）")
        else:
            self.intra_hybrid_nope_heads = 0
        # 运行时增强开关（按开关粒度，用于“交替/分段增强”训练）：默认全开
        self._rt: Dict[str, bool] = {"qk_norm": True, "attn_temp": True}
        # 训练路径静态偏置掩码缓存（仅依赖 T/Tkv/mem_cols，逐层逐步重建代价高）：
        # 避免每步每头重复 arange/torch.zeros/cat 造成的海量分配与 DML 拷贝开销
        self._bias_key: Optional[tuple] = None
        self._bias_cache: Optional[torch.Tensor] = None
        # 增量解码（cache 路径）纯因果掩码缓存：掩码仅依赖 (Tq, Tkv)，确定性，
        # 逐 token 解码时 Tkv 单调增长，缓存避免每步 arange + (1,1,Tq,Tkv) 张量分配
        # （DML 小算子启动税敏感）。仅 attn_mask 为 None（无窗口/记忆/alibi/rel_bias）时命中。
        self._causal_key: Optional[tuple] = None
        self._causal_cache: Optional[torch.Tensor] = None
        # _sync_window 推理期跳过标志（首次同步后置 True，避免每步 DML CPU 同步税）
        self._window_synced: bool = False
        # ALiBi 距离矩阵缓存（第二十一轮性能优化）：训练期 start_pos=0 固定、T/Tkv 确定性，
        # dist=|qpos-kpos| 每步重算 arange×2+abs 有 DML 启动税。缓存按 (Tq,Tkv,start_pos,device)。
        # pe_gate_enabled 时 pe_strength 每步变（log_pe_gate 是 Parameter），但 dist 不变仍可缓存。
        self._alibi_dist_cache: Dict[Tuple[int, int, int, str], torch.Tensor] = {}

    def _sync_window(self):
        """阶段6：从可学习 log_window 重算实际窗口尺寸（每步前向同步，训练时随参数变化）。

        log_window 初始化为 log(init_w / window_base)，故还原须乘回 window_base，
        否则 exp 后丢失 base 缩放、任意 window<32 都会被 round 成 1（窗口无声退化）。

        性能：float(self.log_window) 在 DML 上触发 GPU→CPU 同步税。推理期参数冻结，
        窗口不变，首次同步后跳过（_window_synced 标志）；训练期每步同步（参数在变）。
        """
        if not self.learn_window:
            return
        # 推理期参数不变，首次同步后跳过，避免每步 DML→CPU 同步税
        if not self.training and self._window_synced:
            return
        w = int(round(math.exp(float(self.log_window)) * self.window_base))
        w = max(1, min(w, max(self.window_base, 1) * 4))
        if w != self.window:
            self.window = w
            self._bias_key = None  # 窗口变化 → 掩码缓存失效
        self._window_synced = True

    def forward(self, x: torch.Tensor, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False, start_pos: int = 0,
                memory_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        q, k, v, c_kv = self.project_and_norm(x, start_pos)
        out, present = self.attend(q, k, v, past_kv, use_cache, start_pos, memory_kv, c_kv=c_kv)
        # 第十八轮：Gated Attention（NeurIPS 2025 Best Paper）——门源从 out 升级为 query。
        # 论文关键：门必须由 query 产生才触发 query-dependent 稀疏 + value 通路非线性，
        # BOS 注意力下沉从 46.7%→4.8%。init W=0/b=0 → sigmoid=0.5 半通起步（向后兼容）。
        if self.output_gate_enabled:
            B_, _, T_, _ = q.shape
            q_flat = q.transpose(1, 2).reshape(B_, T_, -1)  # (B,H,T,D) → (B,T,dim)
            out = out * torch.sigmoid(self.output_gate(q_flat))
        return out, present

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
        # 距离矩阵 dist=|qpos-kpos| 仅依赖 (Tq,Tkv,start_pos,device)，确定性，可缓存
        # （第二十一轮性能优化：避免每步 arange×2+abs 的 DML 启动税，~100-200μs/层）
        # 回审修复：增量解码每步 start_pos/Tkv 唯一，缓存键永不复用，无限增长致 OOM。
        # 限制缓存大小——训练期（start_pos=0, T 固定）仅 1 条不会触发清理；
        # 增量解码超过阈值时清空，代价是首步重算 1 次（~100μs，可忽略）。
        if len(self._alibi_dist_cache) > 8:
            self._alibi_dist_cache.clear()
        cache_key = (Tq, Tkv, start_pos, str(device))
        dist = self._alibi_dist_cache.get(cache_key)
        if dist is None:
            qpos = torch.arange(start_pos, start_pos + Tq, device=device).unsqueeze(1)
            kpos = torch.arange(0, Tkv, device=device).unsqueeze(0)
            dist = (qpos - kpos).abs()
            self._alibi_dist_cache[cache_key] = dist
        # slopes: (H,) -> (1,H,1,1)，乘以距离 -> (1,H,Tq,Tkv)
        # alibi_slopes 为 buffer（alibi_learnable=False）或 Parameter（alibi_learnable=True）；
        # 两者皆在 model.to(device) 时已移动到目标设备，无需再 .to(device)。
        # Parameter 时梯度通过 .view() (非 in-place) 正确回流到斜率本身。
        bias = -self.alibi_slopes.view(1, self.num_heads, 1, 1) * dist.unsqueeze(0).unsqueeze(0)
        if mem_cols > 0:
            bias = bias.clone()
            bias[..., :mem_cols] = 0
        if self.pe_gate_enabled:
            # per-head 位置编码强度门控：1.0+tanh(log_pe_gate)，init 1.0（不变）
            # log_pe_gate 是 Parameter 已在设备上，无需 .to(bias.device)
            pe_strength = (1.0 + torch.tanh(self.log_pe_gate)).view(1, -1, 1, 1)
            bias = bias * pe_strength
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
        # rlogits 已在 device 上，比较结果也在 device 上，无需 .to(device)（DML 同步税）
        drop = (rlogits < thr)
        rlogits = rlogits.masked_fill(drop, self.mask_fill_value)
        # 拼回完整 Tkv（记忆段前缀补 0）
        if mem_cols > 0:
            rlogits = torch.cat([torch.zeros(rlogits.size(0), rlogits.size(1), rlogits.size(2), mem_cols,
                                          device=device, dtype=rlogits.dtype), rlogits], dim=-1)
        return rlogits

    def project_and_norm(self, x: torch.Tensor, start_pos: int = 0):
        """廉价部分（在梯度检查点重算区域之外执行，避免被反向重算放大）：
        QKV 投影 + ①QK-Norm + ⑤可学习温度 + RoPE。返回 (q, k, v, c_kv)。

        第十七轮 MLA：开启 use_mla_kv 时，k 不在此处应用 RoPE（保持压缩前无 RoPE），
        而是 cat([k, v]) 压缩到潜空间 c_kv 返回；q 仍应用 RoPE（位置 start_pos）。
        k 的 RoPE 延后到 attend 内部还原后应用（位置 0..T_total-1，拼接后），
        保证 RoPE 旋转信息不被压缩-还原破坏。c_kv=None 表示未开启 MLA。
        """
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # ① QK-Norm + ⑤ 可学习温度（共享预处理，见 apply_qk_norm_and_temp）
        q, k = apply_qk_norm_and_temp(
            q, k, self._rt,
            self.qk_norm if self.qk_norm_enabled else None,
            self.log_temp if self.temp_enabled else None)
        if self.mla_kv_enabled:
            # MLA：q 应用 RoPE（位置 start_pos），k 不应用 RoPE（压缩前保持无 RoPE）
            q = self.rope.apply_to_single(q, start_pos=start_pos, max_len=self.max_seq_length) if self.use_rope else q
            # k, v: (B,H,T,D) → (B,T,H,D) → (B,T,dim)，cat 后压缩到潜空间
            _dim = self.num_heads * self.head_dim
            k_flat = k.permute(0, 2, 1, 3).reshape(B, T, _dim)
            v_flat = v.permute(0, 2, 1, 3).reshape(B, T, _dim)
            c_kv = self.kv_compress(torch.cat([k_flat, v_flat], dim=-1))  # (B,T,kv_latent_dim)
            return q, k, v, c_kv
        if self.use_rope:
            if self.intra_hybrid_rope_enabled and self.intra_hybrid_nope_heads < self.num_heads:
                # 层内 head 拆半：前 H_rope 个 head 应用 RoPE，后 nope_heads 个 head 跳过（NoPE）
                # NoPE half 靠 ALiBi 提供位置信号（config 须同时开 alibi=True）
                H_rope = self.num_heads - self.intra_hybrid_nope_heads
                q_rope, q_nope = q[:, :H_rope], q[:, H_rope:]
                k_rope, k_nope = k[:, :H_rope], k[:, H_rope:]
                q_rope, k_rope = self.rope(q_rope, k_rope, start_pos=start_pos, max_len=self.max_seq_length)
                q = torch.cat([q_rope, q_nope], dim=1)
                k = torch.cat([k_rope, k_nope], dim=1)
            else:
                q, k = self.rope(q, k, start_pos=start_pos, max_len=self.max_seq_length)
        return q, k, v, None

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False,
                start_pos: int = 0,
                memory_kv: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]] = None,
                c_kv: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """重算力部分（在梯度检查点重算区域内执行）：scores/softmax/proj。
        大幅激活（scores 张量）不落盘、反向时重算，保留大模型显存收益。
        memory_kv: (mk, mv, meta) 可学习压缩记忆的 K/V + 检索元信息（门控/稀疏）。
        c_kv: 第十七轮 MLA 当前步潜向量 (B, T_current, kv_latent_dim)。MLA 开启时，
              attend 内部用 c_kv 还原 K/V（拼接 past 后上投影 + RoPE），传入的 k/v 被覆盖。"""
        B, _, Tq, _ = q.shape
        self._sync_window()
        # DML 设备别名不一致（privateuseone vs privateuseone:0）：以本模块权重所在设备为权威，
        # 所有掩码/缓存构建都用它，避免 q.device 被剥索引导致 _bias_cache 每步重建
        dev = self.qkv.weight.device
        # 第十七轮 MLA：还原 K/V 并应用 RoPE（在 memory 注入前，因 inject_memory 期望完整 K/V）
        if self.mla_kv_enabled:
            # c_kv: (B, T_current, kv_latent_dim)
            if use_cache and past_kv is not None:
                # 增量解码：拼接 past 潜向量 + 当前潜向量
                c_kv_past = past_kv[0]  # (B, T_past, kv_latent_dim)
                c_kv_full = torch.cat([c_kv_past, c_kv], dim=1)  # (B, T_total, kv_latent_dim)
            else:
                # 全量路径：直接用当前潜向量
                c_kv_full = c_kv
            B_c, T_total, _ = c_kv_full.shape
            # 上投影还原 K/V：单次 GEMM 输出 2*dim，chunk 拆分为 K/V（view 零拷贝）
            _nh, _hd = self.num_heads, self.head_dim
            kv = self.kv_decompress(c_kv_full)  # (B, T_total, 2*dim)
            k_flat, v_flat = kv.chunk(2, dim=-1)
            k = k_flat.reshape(B_c, T_total, _nh, _hd).permute(0, 2, 1, 3)
            v = v_flat.reshape(B_c, T_total, _nh, _hd).permute(0, 2, 1, 3)
            # 还原后应用 RoPE（位置 0..T_total-1，与非 MLA 路径的"每 token 在对应位置旋转"等价）
            # 第十八轮审查修复：NoPE 层（use_rope=False）k 不可旋转，否则 q 无 RoPE/k 有 RoPE 破坏语义
            k = self.rope.apply_to_single(k, start_pos=0, max_len=self.max_seq_length) if self.use_rope else k
            # MLA 已处理 cache 拼接，后续 cache 分支不再拼接 past_kv
            past_kv = None
        # 第二十一轮：value-side 相对编码——递推 v_encoded[t] = v[t] + λ·v_encoded[t-1]
        # 增量解码：v 仅当前 token，从 past_kv 取 v_{t-1}（cache 存编码后 v）
        # 全量前向：必须用递推（非 shift）以与增量解码一致——cache 存编码后 v，
        # 下一步用 encoded v_{t-1}，若全量用 shift(原 v) 会导致 train/infer parity 崩溃。
        # init λ=0 → tanh=0 → v 不变（向后兼容）；灵感：ViPE value-side relative coding
        if self.value_relative_coding_enabled:
            _lam = torch.tanh(self.value_rel_lambda).view(1, 1, 1, 1)
            if v.size(2) == 1 and use_cache and past_kv is not None and past_kv[1] is not None:
                v = v + _lam * past_kv[1][:, :, -1:, :]
            elif v.size(2) > 1:
                # 递推滤波：v_encoded[t] = v[t] + λ * v_encoded[t-1]
                # 与增量解码路径一致（cache 存 encoded v，下一步复用）
                # 性能优化（第二十五轮）：原 Python for 循环逐 token 做 mul+add，
                # T=64 时产生 ~189 次 DML kernel 启动（实测 28.6ms/层，占前向 17%）。
                # 改用 _parallel_prefix_scan 向量化：a=λ（常数衰减），b=v[t]，
                # h_t = a·h_{t-1} + b_t = λ·v_encoded[t-1] + v[t]（数学等价）。
                # Hillis-Steele 扫描 O(log T) 轮，T=64 仅 6 轮 ~42 次启动（4.5x 减少）。
                # 浮点舍入差异在 atol=1e-4 内（cache parity 测试已验证）。
                T = v.size(2)
                B_, H_, _, D_ = v.shape
                # (B,H,T,D) → (B,T,H*D,1) 适配 _parallel_prefix_scan 的 (B,L,d_inner,d_state)
                v_2d = v.permute(0, 2, 1, 3).reshape(B_, T, H_ * D_, 1)
                a_const = _lam.expand(B_, T, H_ * D_, 1).contiguous()
                v_enc = _parallel_prefix_scan(a_const, v_2d)
                v = v_enc.reshape(B_, T, H_, D_).permute(0, 2, 1, 3).contiguous()
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
            # 第十七轮 MLA：present 存累积潜向量 c_kv_full（past+current 拼接后，而非仅当前 token），
            #   cache 内存仍降 2*dim/kv_latent_dim 倍（存潜向量而非完整 K/V）。
            #   用 (c_kv_full, None) 双元素保持与 (k, v) 格式一致，hybrid 块合并 present[0]/present[1] 时
            #   MLA 路径 past_kv[0]=c_kv_full, past_kv[1]=None（attend 内部检测 mla_kv_enabled 用 [0]）。
            #   修复 bug：原存 (c_kv, None) 只含当前 token，导致下一步 past 只有 1 token 而非全部历史。
            present = (c_kv_full, None) if self.mla_kv_enabled else (k, v)
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
                padded = _pad_mem_bias(mem_bias, Tkv, mem_cols)
                attn_mask = attn_mask + padded
            alibi_b = self._alibi_bias(Tq, Tkv, dev, start_pos, mem_cols=mem_cols)
            if alibi_b is not None:
                attn_mask = attn_mask + alibi_b
            # 全上下文检索：inject_memory 已统一算好 rbias_full（cache 与全量路径同源），
            # 否则开启 retrieval_full 时训练-推理系统性不一致（生成质量偏离训练行为）。
            if rbias_full is not None:
                attn_mask = attn_mask + rbias_full
            # 统一用 fused SDPA（float mask）——DML fused SDPA 现已稳定且比 manual 快 ~2.5x。
            # 关键：DML 的 bool attn_mask 语义与 PyTorch 标准相反（True=允许≠禁止），
            # 但 float attn_mask 正确（加到 scores 上）。本路径的 attn_mask 始终是 float
            # （_build_causal_window_mask 返回 mask.float()*fill_value，后续 +mem_bias 等加法也保持 float）。
            out = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            out = out.transpose(1, 2).reshape(B, Tq, self.num_heads * self.head_dim)
            return self.proj(out), present

        # —— 非缓存（训练 / 含 SSM 模型全量重算）路径 ——
        T = q.size(2)
        Tkv = k.size(2)
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
                # 注意 KV 长度 Tkv = T + mem_cols（记忆列已拼到前面），此处直接用
                # _build_causal_window_mask 构造含记忆列的基础掩码（记忆列恒 0，全局可检索），
                # 再叠加相对偏置表。
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
            padded = _pad_mem_bias(mem_bias, Tkv, mem_cols)  # (B,H,T,Tkv)
            attn_mask = attn_mask + padded
        alibi_b = self._alibi_bias(T, Tkv, dev, start_pos, mem_cols=mem_cols)
        if alibi_b is not None:
            attn_mask = attn_mask + alibi_b
        # 全上下文检索：inject_memory 已统一算好 rbias_full（与 cache 路径同源一致）
        if rbias_full is not None:
            attn_mask = attn_mask + rbias_full
        # 统一用 fused SDPA——DML fused SDPA 比 manual 快 ~2.5x（训练规模实测）。
        # 纯因果（无自定义偏置）时用 is_causal=True（比传全零 mask 更高效，DML 上实测 is_causal 略慢
        # 但 CPU/CUDA 上更快，且保证因果性正确）；有偏置时用 float attn_mask（DML 上 float mask 正确，
        # bool mask 语义反了故不用）。
        # 关键：_use_causal=True 时 _build_causal_window_mask 返回 None → base=torch.zeros → _bias_cache 恒全零，
        # 故无需每步 .item() 同步检查（旧实现的 .abs().max().item() 是 DML→CPU 同步税，已删）。
        _use_causal = (not self.rel_bias) and (memory_kv is None) and (self.window == 0) and (not self.alibi)
        if _use_causal:
            out = scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
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


def _accum_kv(past_kv, k, v):
    """累积 k/v 仅供纯 mixer 场景的 start_pos 推断。

    hybrid 场景下 past_kv[0]/[1] 来自 attn 分支（MLA 时为 3D c_kv / None），
    维度不匹配则跳过累积——hybrid 的 start_pos 由 attn cache 推断，
    linear 的 k/v 被 _run_attn_mixer 合并逻辑（present[0:2] 取 attn）丢弃。
    """
    p0, p1 = past_kv[0], past_kv[1]
    if p0 is not None and p0.dim() == k.dim() and p1 is not None and p1.dim() == v.dim():
        return torch.cat([p0, k], dim=2), torch.cat([p1, v], dim=2)
    return k, v


class LinearMixerBase(nn.Module, EnhancementsMixin):
    """线性注意力系 mixer 的共享基础设施（LinearAttention / AxialLinearAttention）。

    提供 __init__ 初始化、_feat、project_and_norm、_rt 开关，子类只需实现 forward 核心逻辑。
    """

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, attn_temp: bool = True,
                 max_seq_length: int = 64, feature: str = 'relu', head_dim: Optional[int] = None,
                 rope_learnable: bool = False, rope_dim_fraction: float = 1.0,
                 shared_qkv: Optional[nn.Linear] = None, shared_proj: Optional[nn.Linear] = None,
                 yarn_scale: float = 1.0, yarn_beta: float = 0.1, yarn_orig_max_seq_length: int = 0,
                 dim_wise_rope: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.max_seq_length = max_seq_length
        self.feature = feature
        self.qkv = shared_qkv if shared_qkv is not None else nn.Linear(dim, 3 * self.num_heads * self.head_dim, bias=False)
        self.proj = shared_proj if shared_proj is not None else nn.Linear(self.num_heads * self.head_dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, learnable=rope_learnable, dim_fraction=rope_dim_fraction,
                                    yarn_scale=yarn_scale, yarn_beta=yarn_beta,
                                    yarn_orig_max_seq_length=yarn_orig_max_seq_length,
                                    dim_wise=dim_wise_rope)
        self.qk_norm_enabled = qk_norm
        if qk_norm:
            self.qk_norm = RMSNorm(self.head_dim)
        self.temp_enabled = attn_temp
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
        q, k = apply_qk_norm_and_temp(
            q, k, self._rt,
            self.qk_norm if self.qk_norm_enabled else None,
            self.log_temp if self.temp_enabled else None)
        q, k = self.rope(q, k, start_pos=start_pos, max_len=self.max_seq_length)
        return q, k, v


class LinearAttention(LinearMixerBase):
    """线性注意力（线性复杂度 token mixer，O(N) 推理，天然兼容 KV-cache）。

    特征映射 φ=elu(x)+1 后，注意力写为 S = Σ φ(K)⊗V 的递推（因果：按时间累积），
    较 softmax 注意力省去 O(N²) 的 scores 矩阵，长序列/小 iGPU 下显著省算力。
    与 SlidingWindowCausalSelfAttention 同接口（project_and_norm + attend），便于混合门控。
    """

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
            out = self.proj((num_t / den_t).reshape(B, 1, H * D))
            present = (*_accum_kv(past_kv, k, v), S, z)
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


class GatedDeltaNet(LinearMixerBase):
    """Gated DeltaNet：门控 delta rule 线性注意力（ICLR 2025，Qwen3-Next 采纳）。

    用门控 delta rule 替换 LinearAttention 的简单累加：
        S_t = α_t · S_{t-1} + β_t · (v_t - S_{t-1}·k_t) ⊗ k_t
        z_t = α_t · z_{t-1} + β_t · k_t
        o_t = (S_t · q_t) / (z_t · q_t + ε)

    α_t（衰减门）控制遗忘，β_t（输入门）控制写入；delta rule 让新 (k,v) 覆盖旧记忆中
    相似 key 的关联，而非简单叠加——长程检索更精确，对"key 冲突"场景（同义改写、
    指代消解）表现优于朴素线性注意力。

    灵感（不可复用源码，仅参考思路，结合本项目重写）：
      - Qwen3-Next：Gated DeltaNet + Gated Attention 3:1 混合（验证 delta rule 在 LLM 上的有效性）
      - Mamba2/SSD：α/β 门控统一 SSM 与线性注意力递推
      - DeltaNet (ICLR 2025)：delta rule 替代 simple write，提升检索精度

    与项目内 LinearAttention 同接口（project_and_norm + forward），mixer='gated_delta' 启用，
    默认关（AttnConfig.mixer 默认 'attn'，向后兼容）。

    训练全量路径：T≤64 时用 for 循环递推（D·D 矩阵-向量乘开销小，循环开销可接受）。
    增量解码：T=1 单步 delta 更新，O(D²) 内存。
    TODO: 后续可优化为 chunk-wise parallel（DeltaNet 原论文做法）以支持更长训练序列。

    参数:
        alpha_init: 衰减门偏置初值（init -2.0 → sigmoid≈0.12，弱遗忘起步，保留长程信息）
        beta_init:  输入门偏置初值（init 2.0 → sigmoid≈0.88，强写入起步，确保新信息注入）
    """

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, attn_temp: bool = True,
                 max_seq_length: int = 64, feature: str = 'relu', head_dim: Optional[int] = None,
                 rope_learnable: bool = False, rope_dim_fraction: float = 1.0,
                 alpha_init: float = -2.0, beta_init: float = 2.0,
                 shared_qkv: Optional[nn.Linear] = None, shared_proj: Optional[nn.Linear] = None,
                 channel_wise: bool = False,
                 yarn_scale: float = 1.0, yarn_beta: float = 0.1, yarn_orig_max_seq_length: int = 0,
                 dim_wise_rope: bool = False,
                 rwkv7: bool = False):
        super().__init__(dim, num_heads, qk_norm, attn_temp, max_seq_length, feature,
                         head_dim, rope_learnable, rope_dim_fraction,
                         shared_qkv, shared_proj,
                         yarn_scale, yarn_beta, yarn_orig_max_seq_length,
                         dim_wise_rope=dim_wise_rope)
        # 第十九轮 KDA（Kimi Delta Attention）：逐通道衰减 α/β（per-channel 向量而非标量）
        # 灵感：Kimi K3 KDA（arXiv:2510.26692）—— Diag(α_t)·S 让每个通道独立遗忘率，
        # 模型自决"哪些通道保留长程信息、哪些快速更新"。与 MemoryBank per-slot forget 对称。
        # channel_wise=False（默认）：标量 α/β (B,H,T,1)，向后兼容
        # channel_wise=True：向量 α/β (B,H,T,D)，逐通道衰减
        self.channel_wise = channel_wise
        _gate_out = num_heads * (self.head_dim if channel_wise else 1)
        # 门控投影：per-head per-token 的 α/β，输入 x 经线性 + sigmoid
        # init W=0 使门控仅由 bias 决定（α_init/beta_init），训练中 W 学习 x-dependent 调制
        self.alpha_proj = nn.Linear(dim, _gate_out, bias=True)
        self.beta_proj = nn.Linear(dim, _gate_out, bias=True)
        nn.init.zeros_(self.alpha_proj.weight)
        nn.init.constant_(self.alpha_proj.bias, alpha_init)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias, beta_init)
        # 第二十一轮：RWKV-7 广义 Delta Rule——rank-1 状态扰动项 z_t·b_t⊗(b_t^T·S)
        # 灵感：RWKV-7 "Goose" (arXiv:2503.14456)——状态门 z_t 控制扰动强度，b_t 为扰动方向。
        # 递推：S_t = α·S + β·(v-S·k)⊗k + z_t·b_t⊗(b_t^T·S)，rank-1 更新让状态沿 b_t 方向调整。
        # init z bias=-3 → sigmoid≈0.05（弱扰动起步）；b_proj 用通用 N(0,0.02) 初始化
        # （不可归零，否则 b=0→bTS=0→rank-1=0→梯度=0 数学死锁，b_proj 永不更新）。
        # 谱半径由 z_gate≈0.05 控制（rank-1 系数 = z_gate·|b|²，init 时 b 小且 z_gate 小）。
        self.rwkv7_enabled = rwkv7
        if rwkv7:
            self.z_proj = nn.Linear(dim, num_heads, bias=True)  # per-head 标量状态门
            self.b_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)  # 扰动方向
            nn.init.zeros_(self.z_proj.weight)
            nn.init.constant_(self.z_proj.bias, -3.0)

    def _compute_gates(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 per-head per-token 门控 α/β。

        标量模式：返回 (B,H,T,1)；通道模式：返回 (B,H,T,D)。
        """
        # x: (B,T,C) → proj → (B,T,H) 或 (B,T,H*D) → transpose → (B,H,T) 或 (B,H*D,T)
        alpha = torch.sigmoid(self.alpha_proj(x).transpose(1, 2))
        beta = torch.sigmoid(self.beta_proj(x).transpose(1, 2))
        if self.channel_wise:
            # (B, H*D, T) → (B, H, D, T) → (B, H, T, D)
            B, _, T = alpha.shape
            alpha = alpha.reshape(B, self.num_heads, self.head_dim, T).permute(0, 1, 3, 2)
            beta = beta.reshape(B, self.num_heads, self.head_dim, T).permute(0, 1, 3, 2)
        else:
            # (B, H, T) → (B, H, T, 1)
            alpha = alpha.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
        return alpha, beta

    def _compute_rwkv7_gates(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 RWKV-7 状态门 z_t 和扰动方向 b_t。

        z_t: (B,H,T) per-head 标量门控，sigmoid 限制 ∈(0,1)
        b_t: (B,H,T,D) 扰动方向（不归一化，谱半径由 z_gate 控制）
        """
        # z: (B,T,H) → (B,H,T)
        z = torch.sigmoid(self.z_proj(x).transpose(1, 2))
        # b: (B,T,H*D) → (B,H,T,D)
        B, T, _ = x.shape
        b = self.b_proj(x).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        return z, b

    def forward(self, x: torch.Tensor, past_kv=None, use_cache: bool = False, start_pos: int = 0,
                memory_kv=None):
        q, k, v = self.project_and_norm(x, start_pos)
        B, H, T, D = q.shape
        qf = self._feat(q)
        kf = self._feat(k)
        # delta rule 数值稳定要求：k L2 归一化，使 (I - β·k·k^T) 谱半径 < 1 防爆炸
        kf = kf / (kf.norm(dim=-1, keepdim=True) + 1e-6)
        alpha, beta = self._compute_gates(x)  # 各 (B,H,T,1)
        # 第二十一轮：RWKV-7 rank-1 状态扰动项
        # z_gate (B,H,T) 状态门，b_dir (B,H,T,D) 扰动方向
        rwkv7 = self.rwkv7_enabled
        if rwkv7:
            z_gate, b_dir = self._compute_rwkv7_gates(x)

        if use_cache and past_kv is not None and len(past_kv) >= 4 and past_kv[2] is not None:
            # 增量解码：T=1，单步 delta 更新
            S = past_kv[2]  # (B,H,D,D)
            z = past_kv[3]  # (B,H,D)
            kf_t = kf[:, :, 0, :]      # (B,H,D)
            v_t = v[:, :, 0, :]        # (B,H,D)
            alpha_t = alpha[:, :, 0, :].unsqueeze(-1)  # (B,H,1,1) 与 S (B,H,D,D) 广播
            beta_t = beta[:, :, 0, :].unsqueeze(-1)
            # S·k_t：当前 key 与旧状态的关联
            Sk = torch.einsum('bhd,bhde->bhe', kf_t, S)  # (B,H,D)
            # delta 更新：S = α·S + β·(v - S·k)⊗k
            S = alpha_t * S + beta_t * (v_t - Sk).unsqueeze(-1) * kf_t.unsqueeze(-2)
            if rwkv7:
                # rank-1 扰动：S += z_t·b_t⊗(b_t^T·S)
                zt = z_gate[:, :, 0]  # (B,H)
                bt = b_dir[:, :, 0, :]  # (B,H,D)
                bTS = torch.einsum('bhd,bhde->bhe', bt, S)  # (B,H,D)
                S = S + zt.unsqueeze(-1).unsqueeze(-1) * torch.einsum('bhd,bhe->bhde', bt, bTS)
            z = alpha_t.squeeze(-1) * z + beta_t.squeeze(-1) * kf_t
            num = torch.einsum('bhd,bhde->bhe', qf[:, :, 0, :], S)  # (B,H,D)
            den = torch.einsum('bhd,bhd->bh', qf[:, :, 0, :], z).unsqueeze(-1).clamp_min(1e-6)
            out = (num / den).reshape(B, 1, H * D)
            return self.proj(out), (*_accum_kv(past_kv, k, v), S, z)

        # 全量训练：for 循环递推（T≤64 开销可控；后续可优化为 chunk-wise parallel）
        S = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
        z = torch.zeros(B, H, D, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            kf_t = kf[:, :, t, :]
            v_t = v[:, :, t, :]
            alpha_t = alpha[:, :, t, :].unsqueeze(-1)   # (B,H,1,1) 与 S (B,H,D,D) 广播
            beta_t = beta[:, :, t, :].unsqueeze(-1)
            Sk = torch.einsum('bhd,bhde->bhe', kf_t, S)
            S = alpha_t * S + beta_t * (v_t - Sk).unsqueeze(-1) * kf_t.unsqueeze(-2)
            if rwkv7:
                # rank-1 扰动：S += z_t·b_t⊗(b_t^T·S)
                zt = z_gate[:, :, t]  # (B,H)
                bt = b_dir[:, :, t, :]  # (B,H,D)
                bTS = torch.einsum('bhd,bhde->bhe', bt, S)  # (B,H,D)
                S = S + zt.unsqueeze(-1).unsqueeze(-1) * torch.einsum('bhd,bhe->bhde', bt, bTS)
            z = alpha_t.squeeze(-1) * z + beta_t.squeeze(-1) * kf_t
            num = torch.einsum('bhd,bhde->bhe', qf[:, :, t, :], S)
            den = torch.einsum('bhd,bhd->bh', qf[:, :, t, :], z).unsqueeze(-1).clamp_min(1e-6)
            outs.append(num / den)
        out = torch.stack(outs, dim=2)  # (B,H,T,D)
        out = out.transpose(1, 2).reshape(B, T, H * D)
        present = (k, v, S, z) if use_cache else None
        return self.proj(out), present


class AxialLinearAttention(LinearMixerBase):
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
        super().__init__(dim, num_heads, qk_norm=qk_norm, attn_temp=attn_temp,
                         max_seq_length=max_seq_length, feature=feature, head_dim=head_dim,
                         rope_learnable=rope_learnable, shared_qkv=shared_qkv, shared_proj=shared_proj)
        self.grid_size = grid_size
        # col 轴独立投影（row 轴复用基类的 qkv/proj）
        self.qkv_col = nn.Linear(dim, 3 * self.num_heads * self.head_dim, bias=False)
        self.proj_col = nn.Linear(self.num_heads * self.head_dim, dim, bias=False)
        self.gate = nn.Parameter(torch.tensor(gate_init))
        # col 轴独立的 QK-Norm / Temp
        if qk_norm:
            self.qk_norm_col = RMSNorm(self.head_dim)
        if attn_temp:
            self.log_temp_col = nn.Parameter(torch.zeros(1))

    def _infer_grid(self, T: int) -> Tuple[int, int]:
        if self.grid_size is not None:
            return self.grid_size
        import math
        # 优先最接近正方形（空间局部性最强），用 isqrt 作为基准
        s = math.isqrt(T)
        row, col = s, math.ceil(T / s)
        # 如果 row < col，交换使 row ≥ col（行数≥列数，列注意力覆盖距离更远）
        if row < col:
            row, col = col, row
        return row, col

    def _linear_attn_1d(self, q, k, v, qk_norm_mod=None, log_temp_mod=None):
        """单轴线性注意力：cumsum 向量化（全量路径），返回 (B,H,T,D)。"""
        B, H, T, D = q.shape
        if qk_norm_mod is not None:
            q = qk_norm_mod(q)
            k = qk_norm_mod(k)
        if log_temp_mod is not None:
            # 与 apply_qk_norm_and_temp 一致：exp(-0.5*log_temp) 分别作用于 q,k
            scale = torch.exp(-0.5 * log_temp_mod)
            q = q * scale
            k = k * scale
        qf = self._feat(q)
        kf = self._feat(k)
        kv_all = torch.einsum('bhtd,bhte->bhtde', kf, v)
        S_all = torch.cumsum(kv_all, dim=2)
        z_all = torch.cumsum(kf, dim=2)
        num = torch.einsum('bhtd,bhtde->bhte', qf, S_all)
        den = torch.einsum('bhtd,bhtd->bht', qf, z_all).unsqueeze(-1).clamp_min(1e-6)
        return num / den  # (B,H,T,D)

    def _axial_forward(self, x: torch.Tensor, start_pos: int = 0):
        """轴向 2D 线性注意力：reshpe 1D → 2D → 行注意力 → 列注意力 → 融合 → reshape → proj。"""
        B, T, C = x.shape
        row, col = self._infer_grid(T)
        pad_len = row * col - T
        if pad_len > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))  # 补零到 row*col
        x2d = x.reshape(B, row, col, C)

        # padding mask：True 表示 padded 位置，v 设为 0 防止信息泄漏
        pad_mask = None
        if pad_len > 0:
            pad_mask = torch.zeros(B, row * col, dtype=x.dtype, device=x.device)
            pad_mask[:, T:] = float('-inf')  # 用于屏蔽 padded v 的贡献

        # ── 行注意力：沿 col 维度做线性注意力（复用 main 投影） ──
        x_row = x2d.reshape(B * row, col, C)
        qkv_r = self.qkv(x_row).reshape(B * row, col, 3, self.num_heads, self.head_dim)
        qkv_r = qkv_r.permute(2, 0, 3, 1, 4)
        qr, kr, vr = qkv_r[0], qkv_r[1], qkv_r[2]
        qr, kr = apply_qk_norm_and_temp(
            qr, kr, self._rt,
            self.qk_norm if self.qk_norm_enabled else None,
            self.log_temp if self.temp_enabled else None)
        # 屏蔽 padded v：reshape 为 (B*row, col) 后 expand 到 (B*row, H, col, D)
        if pad_mask is not None:
            row_mask = pad_mask.reshape(B * row, col).unsqueeze(1).unsqueeze(-1)
            vr = vr * (row_mask.exp())  # exp(-inf)=0 → padded v 变为 0
        hr = self._linear_attn_1d(qr, kr, vr,
                                   qk_norm_mod=None, log_temp_mod=None)
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
            self.log_temp_col if self.temp_enabled else None)
        # 列注意力的 padding mask：按列重排后，padded 位置在同一列的不同行
        if pad_mask is not None:
            col_mask = pad_mask.reshape(B, row, col).permute(0, 2, 1).reshape(B * col, row).unsqueeze(1).unsqueeze(-1)
            vc = vc * (col_mask.exp())
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
        if use_cache:
            # 推理期：统一用 1D 线性注意力（保持 cache parity）
            # 训练用 2D 轴向注意力（行+列融合），推理用 1D cumsum/RNN 递推；
            # 2D 无法增量递推（需维护 per-row/per-col 独立状态），故推理退化为 1D
            q, k, v = self.project_and_norm(x, start_pos)
            H, D = self.num_heads, self.head_dim
            qf = self._feat(q)
            kf = self._feat(k)
            if past_kv is not None and len(past_kv) >= 4 and past_kv[2] is not None and T == 1:
                # 增量解码（T=1）：RNN 逐 token 累积
                S, z = past_kv[2], past_kv[3]
                kf_t = kf[:, :, 0, :]
                v_t = v[:, :, 0, :]
                S = S + torch.einsum('bhd,bhe->bhde', kf_t, v_t)
                z = z + kf_t
                qf_t = qf[:, :, 0, :]
                num = torch.einsum('bhd,bhde->bhe', qf_t, S)
                den = torch.einsum('bhd,bhd->bh', qf_t, z).unsqueeze(-1).clamp_min(1e-6)
                out = self.proj((num / den).reshape(B, 1, H * D))
                return out, (*_accum_kv(past_kv, k, v), S, z)
            # prefill/首步（T>=1）：cumsum 向量化 1D 路径
            kv_all = torch.einsum('bhtd,bhte->bhtde', kf, v)
            S_all = torch.cumsum(kv_all, dim=2)
            z_all = torch.cumsum(kf, dim=2)
            num = torch.einsum('bhtd,bhtde->bhte', qf, S_all)
            den = torch.einsum('bhtd,bhtd->bht', qf, z_all).unsqueeze(-1).clamp_min(1e-6)
            out = (num / den).transpose(1, 2).reshape(B, T, H * D)
            present = (k, v, S_all[:, :, -1], z_all[:, :, -1])
            return self.proj(out), present
        # 训练期（use_cache=False）：2D 轴向线性注意力
        if memory_kv is not None:
            import warnings
            warnings.warn(
                "AxialLinearAttention 全量路径暂不支持 memory_kv 注入，记忆信息被忽略。",
                stacklevel=2,
            )
        fused = self._axial_forward(x, start_pos=start_pos)
        return self.proj(fused), None


class DifferentialAttention(nn.Module, EnhancementsMixin):
    """差分注意力（Differential Attention，CVPR 2025）。

    核心思想：用两组注意力的差值来消除噪声，增强关键信息。
    out = (softmax(Q1·K1^T) - λ · softmax(Q2·K2^T)) · V
    其中 λ 是可学习标量，两组 Q/K 共享投影但独立计算注意力。

    复杂度 O(T²)（标准注意力），但差分机制比标准注意力更关注显著特征。
    支持增量解码（use_cache）和层间共享投影。
    """
    def __init__(self, dim: int, num_heads: int, max_seq_length: int = 64,
                 qk_norm: bool = True, attn_temp: bool = True,
                 mask_fill_value: float = MASK_FILL_VALUE,
                 shared_qkv: Optional[nn.Linear] = None,
                 shared_proj: Optional[nn.Linear] = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_seq_length = max_seq_length
        self.mask_fill_value = float(mask_fill_value)
        # 层间共享：传入 shared_qkv/shared_proj 时复用外部投影
        # 差分注意力需要 2 组独立 Q/K（共享 V），额外创建一组 QKV 投影
        self.qkv = shared_qkv if shared_qkv is not None else nn.Linear(dim, 3 * dim, bias=False)
        self.qkv2 = nn.Linear(dim, 2 * dim, bias=False)  # 第二组 Q/K 投影
        self.proj = shared_proj if shared_proj is not None else nn.Linear(dim, dim, bias=False)
        # QK-Norm + 温度（与 SlidingWindowCausalSelfAttention 一致）
        self.qk_norm_enabled = qk_norm
        if qk_norm:
            self.qk_norm = RMSNorm(self.head_dim)
        self.temp_enabled = attn_temp
        if attn_temp:
            self.log_temp = nn.Parameter(torch.zeros(1))
        # 可学习差分权重 λ（初始化为 0.5，介于完全差分和完全平均之间）
        self.diff_lambda = nn.Parameter(torch.tensor(0.5))
        # 因果掩码缓冲区
        self.register_buffer('_causal_mask', torch.triu(torch.ones(1, 1, max_seq_length, max_seq_length, dtype=torch.bool), diagonal=1))
        self._cached_T = None
        # 增强调度运行时开关（与 SlidingWindowCausalSelfAttention 一致）
        self._rt = {"qk_norm": True, "attn_temp": True}

    def forward(self, x: torch.Tensor, past_kv=None, use_cache: bool = False,
                start_pos: int = 0, memory_kv=None):
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim
        # 第一组 Q/K/V
        qkv = self.qkv(x)  # (B, T, 3*C)
        qkv = qkv.reshape(B, T, 3, H, D)
        q1, k1, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # 各 (B, T, H, D)
        # 第二组 Q/K（独立投影）
        qkv2 = self.qkv2(x)  # (B, T, 2*C)
        q2, k2 = qkv2.reshape(B, T, 2, H, D).unbind(dim=2)
        # QK-Norm（受 SEL 运行时开关 _rt['qk_norm'] 控制）
        if self.qk_norm_enabled and self._rt.get('qk_norm', True):
            q1 = self.qk_norm(q1)
            k1 = self.qk_norm(k1)
            q2 = self.qk_norm(q2)
            k2 = self.qk_norm(k2)
        # 温度缩放（受 SEL 运行时开关 _rt['attn_temp'] 控制）
        if self.temp_enabled and self._rt.get('attn_temp', True):
            scale = torch.exp(-0.5 * self.log_temp)
            q1, q2 = q1 * scale, q2 * scale
            k1, k2 = k1 * scale, k2 * scale
        # 因果掩码
        causal = self._causal_mask[:, :, :T, :T]  # (1, 1, T, T)
        if use_cache and past_kv is not None:
            # past_kv 由 TransformerBlock 传入，结构为 attn_kv=(k1,k2,v) 或完整元组
            # cache 统一 (B,H,T,D) 布局（与其他 mixer 一致，BlockState.start_pos 取 size(2)）
            if isinstance(past_kv, tuple) and len(past_kv) == 3 and isinstance(past_kv[0], torch.Tensor):
                pk1, pk2, pv = past_kv
            elif isinstance(past_kv, tuple) and len(past_kv) == 3 and isinstance(past_kv[0], tuple):
                pk1, pk2, pv = past_kv[0]
            else:
                pk1, pk2, pv = None, None, None
            if pk1 is not None:
                # pk1/pk2/pv 为 (B,H,T_past,D)，当前 k1/k2/v 为 (B,T,H,D)，转置后 cat dim=2
                k1_full = torch.cat([pk1, k1.transpose(1, 2)], dim=2)  # (B,H,T_total,D)
                k2_full = torch.cat([pk2, k2.transpose(1, 2)], dim=2)
                v_full = torch.cat([pv, v.transpose(1, 2)], dim=2)
                Tkv = k1_full.size(2)
                # 增量掩码：当前 token 可 attend 所有历史
                causal = torch.zeros(1, 1, T, Tkv, dtype=torch.bool, device=x.device)
                # 转回 (B,T,H,D) 保持后续 transpose 逻辑统一（transpose 是 view 零开销）
                k1_full = k1_full.transpose(1, 2)
                k2_full = k2_full.transpose(1, 2)
                v_full = v_full.transpose(1, 2)
            else:
                k1_full, k2_full, v_full = k1, k2, v
                Tkv = T
        else:
            k1_full, k2_full, v_full = k1, k2, v
            Tkv = T
        # transpose 到 (B, H, T, D) 供 attention 与 memory 注入使用
        q1t = q1.transpose(1, 2)
        q2t = q2.transpose(1, 2)
        k1t = k1_full.transpose(1, 2)  # (B, H, Tkv, D)
        k2t = k2_full.transpose(1, 2)
        vt = v_full.transpose(1, 2)    # (B, H, Tkv, D)
        # 记忆注入（若有）：先扩展 K/V（前缀拼记忆列），再算 scores，避免 scores 形状与
        # 增长后的 K/V 不符（与 SlidingWindowCausalSelfAttention 顺序一致，原实现 scores 先算
        # 导致 (B,H,T,Tkv) + (B,H,T,M) 形状崩溃）。
        mem_cols = 0
        if memory_kv is not None:
            mk, mv, mem_meta = memory_kv
            mem_cols = mk.size(1)
            k1t, v_aug, mem_bias1 = self.__class__.inject_mem(
                q1t, k1t, vt, mk, mv, mem_meta, self.mask_fill_value)
            k2t, _, mem_bias2 = self.__class__.inject_mem(
                q2t, k2t, vt, mk, mv, mem_meta, self.mask_fill_value)
            vt = v_aug
            # K/V 现在长度 Tkv+mem_cols，掩码需同步扩展（记忆列恒不遮蔽）
            if mem_cols > 0:
                Tkv_new = k1t.size(2)
                if causal.shape[-1] < Tkv_new:
                    # 原 causal 为 (1,1,T,Tkv)，左侧补 False（记忆列不遮蔽）
                    pad = torch.zeros(1, 1, T, mem_cols, dtype=causal.dtype, device=causal.device)
                    causal = torch.cat([pad, causal], dim=-1)
        else:
            mem_bias1 = mem_bias2 = None
        # 注意力计算：两组独立 softmax
        scores1 = torch.matmul(q1t, k1t.transpose(-2, -1)) / math.sqrt(D)
        scores2 = torch.matmul(q2t, k2t.transpose(-2, -1)) / math.sqrt(D)
        scores1 = scores1.masked_fill(causal, self.mask_fill_value)
        scores2 = scores2.masked_fill(causal, self.mask_fill_value)
        # 记忆段偏置（仅前 mem_cols 列），右侧补零到 Tkv_full 后广播相加
        if mem_bias1 is not None:
            Tkv_full = scores1.size(-1)
            scores1 = scores1 + _pad_mem_bias(mem_bias1, Tkv_full, mem_cols)
            scores2 = scores2 + _pad_mem_bias(mem_bias2, Tkv_full, mem_cols)
        attn1 = torch.softmax(scores1, dim=-1)
        attn2 = torch.softmax(scores2, dim=-1)
        # 差分注意力
        lam = torch.sigmoid(self.diff_lambda)
        attn_diff = attn1 - lam * attn2  # (B, H, T, Tkv)
        out = torch.matmul(attn_diff, vt)  # (B, H, T, D)
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        if use_cache:
            # present 存累积的 k1_full/k2_full/v_full（cache 路径含全部历史，非 cache 路径仅当前 token），
            # 转置到 (B,H,T,D) 布局（与其他 mixer 一致，BlockState.start_pos 取 size(2)）
            return out, (k1_full.transpose(1, 2), k2_full.transpose(1, 2), v_full.transpose(1, 2))
        return out, None

    @staticmethod
    def inject_mem(q, k, v, mk, mv, meta, mask_fill):
        """复用 MemoryBank.inject_memory 统一逻辑，避免与 SlidingWindowCausalSelfAttention 漂移。"""
        from models.memory import MemoryBank
        k_aug, v_aug, mem_bias = MemoryBank.inject_memory(q, k, v, mk, mv, meta, mask_fill)
        return k_aug, v_aug, mem_bias


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

    def _compute_dA_and_xb(self, x_conv: torch.Tensor, dt: torch.Tensor,
                            Bp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算离散化转移矩阵 dA 和融合输入 xb。

        标准 Mamba：A 静态（A = -exp(A_log)），dA = exp(dt * A)。
        MambaSSMWithCAST 覆盖此方法添加上下文调制 A_delta。

        Returns:
            dA: (B, L, d_inner, d_state)
            xb: (B, L, d_inner, d_state) 融合 dB * x_conv
        """
        A = -torch.exp(self.A_log)                   # (d_inner, d_state)
        dA = torch.exp(dt.unsqueeze(-1) * A)          # (B, L, d_inner, d_state)
        # 融合 dB * x_conv 为单次运算，避免 (B,L,d_inner,d_state) 中间张量分配
        xb = (dt.unsqueeze(-1) * Bp.unsqueeze(2)) * x_conv.unsqueeze(-1)
        return dA, xb

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
                # 保存最后 conv_kernel-1 个 token 供下一步增量使用。
                # max(.,0) 防御 conv_kernel=1 时 -(1-1)=-0 等价于 0 导致返回全长序列（应空切片）
                keep = max(self.conv_kernel - 1, 0)
                if keep > 0:
                    conv_state = x_in.transpose(1, 2)[:, :, -keep:]
                    # 首步 L < keep（如 pure_incremental L=1, keep=2）时切片不足 keep 个元素，
                    # 左补零对齐（因果卷积左填充语义：缺失位置等价于零输入）
                    if conv_state.size(-1) < keep:
                        conv_state = torch.nn.functional.pad(conv_state, (keep - conv_state.size(-1), 0))
                    present_conv_state = conv_state
                else:
                    present_conv_state = x_in.new_zeros(x_in.size(0), x_in.transpose(1, 2).size(1), 0)
            else:
                present_conv_state = None

        x_conv = self.act(x_conv)                    # (B, L, d_inner)
        ssm = self.x_proj(x_conv)                    # (B, L, dt_rank + 2*d_state)
        dt_in, Bp, C = ssm.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = torch.nn.functional.softplus(self.dt_proj(dt_in))   # (B, L, d_inner)
        dA, xb = self._compute_dA_and_xb(x_conv, dt, Bp)

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
        """并行前缀扫描计算 h_t = a_t * h_{t-1} + b_t（h_0=0 或 past_state）。"""
        return _parallel_prefix_scan(a, b, past_state)


class SwiGLU(nn.Module):
    """SwiGLU 前馈（LLaMA 风格门控 FFN，比 GELU MLP 更有表达力）。

    fuse_swiglu=True 时将 w1/w3 合并为单个 w13 Linear（2*hidden_dim 输出），
    前向时 chunk 拆分为两路。减少一次 GEMM 调用，DML 上小算子启动税敏感时有益。
    默认关（向后兼容旧 checkpoint 的 w1/w2/w3 三参数格式）。
    """
    def __init__(self, dim: int, hidden_dim: int, fuse_swiglu: bool = False):
        super().__init__()
        self.fuse_swiglu = fuse_swiglu
        if fuse_swiglu:
            self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        else:
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, dim, bias=False)
            self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fuse_swiglu:
            x1, x3 = self.w13(x).chunk(2, dim=-1)
            return self.w2(torch.nn.functional.silu(x1) * x3)
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))

    @staticmethod
    def convert_legacy_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """将旧格式 state_dict（w1/w3 分离）转换为新格式（w13 合并）。

        供 fuse_swiglu=True 模型加载旧 checkpoint 时使用：
        w13.weight = cat([w1.weight, w3.weight], dim=0)
        """
        new_sd: Dict[str, torch.Tensor] = {}
        # 预扫描：先标记所有将被转换的 w3_key，避免 w3 在 w1 之前迭代时残留
        skip = set()
        for k in state_dict:
            parts = k.split('.')
            if len(parts) >= 2 and parts[-2:] == ['w1', 'weight']:
                prefix = '.'.join(parts[:-2])
                w3_key = f'{prefix}.w3.weight' if prefix else 'w3.weight'
                if w3_key in state_dict:
                    skip.add(w3_key)
        for k, v in state_dict.items():
            if k in skip:
                continue
            parts = k.split('.')
            if len(parts) >= 2 and parts[-2:] == ['w1', 'weight']:
                prefix = '.'.join(parts[:-2])
                w3_key = f'{prefix}.w3.weight' if prefix else 'w3.weight'
                w13_key = f'{prefix}.w13.weight' if prefix else 'w13.weight'
                if w3_key in state_dict:
                    new_sd[w13_key] = torch.cat([v, state_dict[w3_key]], dim=0)
                    continue
            new_sd[k] = v
        return new_sd


class MambaSSMWithCAST(MambaSSM):
    """MambaSSM + CAST（Context-Adaptive State Transition）。

    核心思想：状态转移矩阵 A 由局部上下文动态调制。
    - 基态矩阵 A_base = -exp(A_log)（与标准 MambaSSM 相同）
    - 上下文残差 A_delta：从输入 x 的局部统计身份（均值/方差/范数）推导
    - 有效转移矩阵 A_t = A_base + A_delta（每个位置不同）

    与标准 MambaSSM 的区别：
    - 标准 Mamba：A 是静态的（与输入无关）
    - CAST 版：A 由上下文动态调制（更灵活的选择性）

    兼容 MambaSSM 的接口（past_state/past_conv_state/use_cache）。
    """
    def __init__(self, dim: int, d_state: int = 16, d_inner_factor: int = 1,
                 dt_rank: Optional[int] = None, conv_kernel: int = 3,
                 dt_proj_bias_init: float = 0.1,
                 a_log_init_range: Tuple[float, float] = (-1.0, 1.0),
                 D_init: float = 1.0,
                 cast_hidden: int = 32):
        super().__init__(dim, d_state, d_inner_factor, dt_rank, conv_kernel,
                         dt_proj_bias_init, a_log_init_range, D_init)
        # CAST 组件：从上下文推导 A_delta
        self.cast_stat_proj = nn.Linear(3, cast_hidden, bias=False)
        self.cast_delta_proj = nn.Linear(cast_hidden, self.d_inner * d_state, bias=False)
        nn.init.zeros_(self.cast_delta_proj.weight)  # 零初始化→初始行为等价于标准 Mamba

    def _compute_cast_delta(self, x: torch.Tensor) -> torch.Tensor:
        """从输入 x 的局部统计身份推导 A_delta。"""
        # x: (B, L, d_inner)
        # 计算逐位置统计量（用全局统计作为近似，避免引入额外注意力）
        mean = x.mean(dim=-1, keepdim=True)      # (B, L, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # (B, L, 1)
        norm = x.norm(dim=-1, keepdim=True)       # (B, L, 1)
        stats = torch.cat([mean, var, norm], dim=-1)  # (B, L, 3)
        # 投影到 A_delta
        h = self.cast_stat_proj(stats)            # (B, L, cast_hidden)
        h = F.silu(h)
        delta = self.cast_delta_proj(h)           # (B, L, d_inner * d_state)
        return delta.view(x.size(0), x.size(1), self.d_inner, self.d_state)

    def _compute_dA_and_xb(self, x_conv: torch.Tensor, dt: torch.Tensor,
                            Bp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """CAST 覆盖：上下文调制 A（A_base + A_delta）再离散化。

        继承基类 forward 的所有 conv/x_proj/selective_scan/output 逻辑，
        仅差异在 dA 计算：标准 Mamba 用静态 A，CAST 用 A_base + A_delta。
        同时修复基类 conv_kernel=1 防御（CAST 旧 forward 缺失此检查）。
        """
        A_base = -torch.exp(self.A_log)
        A_delta = self._compute_cast_delta(x_conv)
        A_effective = A_base.unsqueeze(0).unsqueeze(0) + A_delta
        dA = torch.exp(dt.unsqueeze(-1) * A_effective)
        xb = (dt.unsqueeze(-1) * Bp.unsqueeze(2)) * x_conv.unsqueeze(-1)
        return dA, xb
