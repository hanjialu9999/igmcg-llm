from __future__ import annotations

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.functional import scaled_dot_product_attention
import threading
from typing import Optional, List, Tuple, Any, Dict, Callable


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
    def __init__(self, dim: int, num_heads: int, window: int = 0, rel_bias: bool = False, max_seq_length: int = 64):
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
        self._cached_T = T

    def forward(self, x: torch.Tensor, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False, start_pos: int = 0) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.rope(q, k, start_pos=start_pos, max_len=self.max_seq_length)

        if use_cache:
            # 增量解码：拼接待拼接的 K/V 缓存，仅对当前 token 做注意力
            if past_kv is not None:
                pk, pv = past_kv
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)
            present = (k, v)
            Tkv = k.size(2)
            qpos = torch.arange(start_pos, start_pos + T, device=q.device).unsqueeze(1)
            kpos = torch.arange(0, Tkv, device=q.device).unsqueeze(0)
            causal_mask = kpos > qpos
            if self.window > 0:
                causal_mask = causal_mask | (qpos - kpos > self.window)
            attn_mask = (causal_mask.float() * -1e9).unsqueeze(0)   # (1,1,T,Tkv)
            if self.rel_bias:
                idx = (qpos - kpos + Tkv - 1).clamp(0, 2 * self.max_seq_length - 1)
                attn_mask = attn_mask + self.rel_bias_table[:, idx].unsqueeze(0)
            # 增量解码单 token 查询时，SDPA 在 CPU 上的 per-call 开销远大于一次显式 matmul，
            # 故 CPU 也走手动注意力；CUDA 仍用 fused SDPA（显存带宽充足、kernel 更优）。
            if x.device.type == 'cuda':
                out = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            else:
                out = self._manual_attention(q, k, v, attn_mask)
            out = out.transpose(1, 2).reshape(B, T, C)
            return self.proj(out), present

        # —— 非缓存（训练 / 含 SSM 模型全量重算）路径 ——
        self._build_masks(T, q.device)
        if x.device.type in ('cuda', 'cpu'):
            if self.rel_bias or self.window > 0:
                attn_mask = self._mask.float() * -1e9
                if self.rel_bias:
                    attn_mask = attn_mask + self._rbias.unsqueeze(0)
                out = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            else:
                out = scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            extra = self._mask.float() * -1e9 if (self.rel_bias or self.window > 0) else None
            if self.rel_bias and extra is not None:
                extra = extra + self._rbias.unsqueeze(0)
            out = self._manual_attention(q, k, v, extra)
        out = out.transpose(1, 2).reshape(B, T, C)
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
                 ssm_kwargs: Optional[Dict[str, Any]] = None, attn_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.block_type = block_type
        self.drop = nn.Dropout(dropout)
        ssm_kwargs = ssm_kwargs or {}
        attn_kwargs = attn_kwargs or {}
        # Both attn and ssm blocks need a pre-norm layer
        self.ln1 = RMSNorm(dim)
        if block_type in ('attn', 'hybrid'):
            self.attn = SlidingWindowCausalSelfAttention(
                dim, num_heads, max_seq_length=max_seq_length, **attn_kwargs)
        if block_type in ('ssm', 'hybrid'):
            self.ssm = MambaSSM(dim, **ssm_kwargs)
        self.ln2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden_dim)

    def forward(self, x: torch.Tensor, past_kv: Optional[Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]] = None, use_cache: bool = False, start_pos: int = 0, ssm_past_state: Optional[torch.Tensor] = None, ssm_past_conv_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]]]:
        present: Optional[Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]] = None
        ssm_present_state: Optional[torch.Tensor] = None
        ssm_present_conv_state: Optional[torch.Tensor] = None
        # Extract attention KV from past_kv tuple (attn_kv, ssm_state, ssm_conv_state)
        attn_past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if past_kv is not None:
            attn_past_kv = past_kv[0]
        if self.block_type == 'attn':
            h, present = self.attn(self.ln1(x), attn_past_kv, use_cache, start_pos)
            x = x + self.drop(h)
        elif self.block_type == 'ssm':
            h, ssm_present_state, ssm_present_conv_state = self.ssm(self.ln1(x), past_state=ssm_past_state, past_conv_state=ssm_past_conv_state, use_cache=use_cache)
            x = x + self.drop(h)
        elif self.block_type == 'hybrid':
            h, attn_present = self.attn(self.ln1(x), attn_past_kv, use_cache, start_pos)
            ssm_h, ssm_present_state, ssm_present_conv_state = self.ssm(self.ln1(x), past_state=ssm_past_state, past_conv_state=ssm_past_conv_state, use_cache=use_cache)
            x = x + self.drop(h) + self.drop(ssm_h)
            if use_cache:
                present = (attn_present, ssm_present_state, ssm_present_conv_state)
        x = x + self.drop(self.ffn(self.ln2(x)))
        # Combine attn KV cache and SSM state
        if use_cache:
            if self.block_type == 'attn':
                present = (present, None, None)  # (attn_kv, ssm_state, ssm_conv_state)
            elif self.block_type == 'ssm':
                present = (None, ssm_present_state, ssm_present_conv_state)
            elif self.block_type == 'hybrid':
                present = (attn_present, ssm_present_state, ssm_present_conv_state)
        return x, present


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
                 mask_fill_value: float = -1e9):
        super(TransformerModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.gradient_checkpointing = gradient_checkpointing
        self.layer_plan = _parse_layer_plan(layer_plan, num_layers)
        self.rope_base = rope_base
        self.rope_max_len = rope_max_len
        self.mask_fill_value = mask_fill_value

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.drop = nn.Dropout(dropout)
        ssm_kwargs = dict(
            d_state=ssm_d_state,
            d_inner_factor=ssm_d_inner_factor,
            dt_rank=ssm_dt_rank,
            conv_kernel=ssm_conv_kernel,
            dt_proj_bias_init=ssm_dt_proj_bias_init,
            a_log_init_range=ssm_a_log_init_range,
            D_init=ssm_D_init,
        )
        attn_kwargs = dict(window=attn_window, rel_bias=attn_rel_bias)
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, hidden_dim, block_type=bt,
                             dropout=dropout, max_seq_length=max_seq_length,
                             ssm_kwargs=ssm_kwargs, attn_kwargs=attn_kwargs)
            for bt in self.layer_plan
        ])
        self.ln_f = RMSNorm(embedding_dim)
        self.output_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self._tie_weights = tie_weights
        if tie_weights:
            self.output_head.weight = self.embedding.weight

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

        for i, block in enumerate(self.blocks):
            ssm_past_state = ssm_states[i] if use_cache else None
            ssm_past_conv_state = ssm_conv_states[i] if use_cache else None
            if self.training and self.gradient_checkpointing:
                x, present = checkpoint(block, x, past_key_values[i], use_cache, start_pos, ssm_past_state, ssm_past_conv_state,
                                        use_reentrant=False)
            else:
                x, present = block(x, past_key_values[i], use_cache, start_pos, ssm_past_state, ssm_past_conv_state)
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