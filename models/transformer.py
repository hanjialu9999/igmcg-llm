import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.functional import scaled_dot_product_attention


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization（LLaMA 风格，比 LayerNorm 更省且更稳定）。"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


_ROPE_CACHE = {}


def _rope_cos_sin(inv_freq, start_pos, seq_len, device, dtype, max_len=2048):
    """按 (device, head_dim) 缓存整张位置表后按需切片，跨层与跨生成步共享。

    原实现按 (start_pos, seq_len) 缓存单段：KV 缓存逐 token 生成时 start_pos 每步
    都变化，导致缓存每步必未命中、反复重算 RoPE。改为缓存整表并切片后，训练时多层
    与生成时每步单 token 都能命中同一张表。
    """
    key = (str(device), inv_freq.shape[0])
    need = start_pos + seq_len
    cached = _ROPE_CACHE.get(key)
    if cached is not None and cached[0].size(2) >= need:
        cos_full, sin_full = cached
        return cos_full[:, :, start_pos:need, :].to(dtype), sin_full[:, :, start_pos:need, :].to(dtype)
    L = max(need, min(max_len, 4096))
    t = torch.arange(0, L, device=device).type_as(inv_freq)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_full = emb.cos()[None, None, :, :].to(dtype)
    sin_full = emb.sin()[None, None, :, :].to(dtype)
    _ROPE_CACHE[key] = (cos_full, sin_full)
    return cos_full[:, :, start_pos:need, :].to(dtype), sin_full[:, :, start_pos:need, :].to(dtype)


class RotaryEmbedding(nn.Module):
    """旋转位置编码 RoPE：对 Q/K 按位置旋转，天然支持长度外推。"""
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, q, k, start_pos=0, max_len=2048):
        cos, sin = _rope_cos_sin(self.inv_freq, start_pos, q.size(2), q.device, q.dtype, max_len=max_len)
        return self._rope_apply(q, cos, sin), self._rope_apply(k, cos, sin)

    @staticmethod
    def _rope_apply(x, cos, sin):
        d = x.size(-1) // 2
        x1, x2 = x[..., :d], x[..., d:]
        rot = torch.cat((-x2, x1), dim=-1)
        return x * cos.to(x.dtype) + rot * sin.to(x.dtype)


class SlidingWindowCausalSelfAttention(nn.Module):
    """因果自注意力，可选滑动窗口 + 可学习相对位置偏置。
     CUDA/CPU 用原生 fused SDPA；AMD DirectML 的 fused 内核会触发原生崩溃，
     故 DML(及其他后端)走手动 matmul+softmax+因果掩码 以规避该 bug。
    """
    def __init__(self, dim, num_heads, window=0, rel_bias=False, max_seq_length=64):
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
        self._mask = None
        self._rbias = None

    def _build_masks(self, T, device):
        if self._cached_T == T and (self._mask is not None or (not self.rel_bias)):
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

    def forward(self, x, past_kv=None, use_cache=False, start_pos=0):
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

    def _manual_attention(self, q, k, v, attn_mask=None):
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
      选择性扫描已向量化（cumprod + cumsum 解析展开），消除逐时间步 Python for 循环：
      既显著加快 CPU 训练，也避免低功耗 iGPU 上 DML 因单次步内 kernel 过多触发 TDR 设备重置。
    """
    def __init__(self, dim, d_state=16, d_inner_factor=1, dt_rank=None, conv_kernel=3):
        super().__init__()
        d_inner = dim * d_inner_factor
        dt_rank = dt_rank or max(1, math.ceil(dim / 16))
        self.dim = dim
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.norm = RMSNorm(dim)
        self.in_proj = nn.Linear(dim, 2 * d_inner, bias=False)
        self.conv = nn.Conv1d(d_inner, d_inner, kernel_size=conv_kernel,
                              padding=conv_kernel // 2, groups=d_inner, bias=False)
        self.act = nn.SiLU()
        # 从 conv 输出投影出 Δ 输入、B、C（选择性）
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        nn.init.constant_(self.dt_proj.bias, 0.1)
        # A 以对数形式存储，保证 A = -exp(A_log) 为负且稳定
        self.A_log = nn.Parameter(torch.empty(d_inner, d_state))
        nn.init.uniform_(self.A_log, -1, 1)
        self.D = nn.Parameter(torch.ones(d_inner))   # 跳跃连接
        self.out_proj = nn.Linear(d_inner, dim, bias=False)
        self.proper_init()

    def proper_init(self):
        """SSM 专用初始化（避免被 TransformerModel._init_weights 的通用初始化覆盖）：
         - in/out/x_proj/dt_proj 权重用 Xavier（B/C 投影更稳）
         - dt_proj 偏置置 0.1（遗忘偏置，缓解早期不稳定）
         - A_log 用 uniform(-1,1) -> A=-exp(A_log) 为负且稳定（Mamba 风格）
         - D 跳跃连接置 1
        """
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.xavier_uniform_(self.dt_proj.weight)
        nn.init.constant_(self.dt_proj.bias, 0.1)
        nn.init.uniform_(self.A_log, -1, 1)
        nn.init.ones_(self.D)

    def forward(self, x):
        B, L, _ = x.shape
        x = self.norm(x)
        xz = self.in_proj(x)                          # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv = self.conv(x_in.transpose(1, 2)).transpose(1, 2)
        x_conv = self.act(x_conv)                    # (B, L, d_inner)
        ssm = self.x_proj(x_conv)                    # (B, L, dt_rank + 2*d_state)
        dt_in, Bp, Cp = ssm.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = torch.nn.functional.softplus(self.dt_proj(dt_in))   # (B, L, d_inner)
        A = -torch.exp(self.A_log)                   # (d_inner, d_state)
        dA = torch.exp(dt.unsqueeze(-1) * A)          # (B, L, d_inner, d_state)
        dB = dt.unsqueeze(-1) * Bp.unsqueeze(2)       # (B, L, d_inner, d_state)
        xb = dB * x_conv.unsqueeze(-1)               # (B, L, d_inner, d_state)
        C = Cp                                        # (B, L, d_state)
        # 向量化选择性扫描（并行前缀扫描，log2(L) 步，数值稳定且无逐时间步 for 循环）：
        # 递推 h_t = dA_t*h_{t-1} + xb_t（h_0=0）用半群 (A,B)⊙(A',B') = (A·A', A'·B + B')
        # 做前缀扫描。相比逐时间步 for 循环，kernel 启动数从 L 降到 log2(L)，消除 DML 上
        # 单步 kernel 过多导致的 TDR 设备重置；相比闭式 cumprod/cumsum，全程不除 cumprod
        # （其长序列会下溢为 0，导致 0·inf=NaN），数值稳定。
        h = self._selective_scan(dA, xb)             # (B, L, d_inner, d_state)
        y = (h * C.unsqueeze(2)).sum(-1)             # (B, L, d_inner)
        y = y + self.D * x_conv                      # 跳跃连接
        y = y * self.act(z)                          # 门控
        return self.out_proj(y)

    @staticmethod
    def _selective_scan(a, b):
        """并行前缀扫描计算 h_t = a_t * h_{t-1} + b_t（h_0=0）。

        a, b: (B, L, d_inner, d_state)。返回 h: (B, L, d_inner, d_state)。
        半群 (A, B)⊙(A', B') = (A·A', A'·B + B') 满足结合律；
        Hillis-Steele 含扫描：每轮把左邻 2^k 步的变换合并进来，offset 从 1 翻倍到 <L。
        单位元为 (A=1, B=0)，越界位置用单位元填充。
        """
        L = a.shape[1]
        A = a
        B = b
        offset = 1
        while offset < L:
            # 左移 offset：位置 i 取 i-offset（越界填单位元 A=1, B=0）
            A_prev = torch.cat([torch.ones_like(A[:, :offset]), A[:, :-offset]], dim=1)
            B_prev = torch.cat([torch.zeros_like(B[:, :offset]), B[:, :-offset]], dim=1)
            A_new = A_prev * A
            B_new = A * B_prev + B
            A, B = A_new, B_new
            offset <<= 1
        return B


class SwiGLU(nn.Module):
    """SwiGLU 前馈（LLaMA 风格门控 FFN，比 GELU MLP 更有表达力）。"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """可配置混合块：attn / ssm / hybrid(attn+ssm 并行)。Pre-LN。"""
    def __init__(self, dim, num_heads, hidden_dim, block_type='attn',
                 dropout=0.0, max_seq_length=64,
                 ssm_kwargs=None, attn_kwargs=None):
        super().__init__()
        self.block_type = block_type
        self.drop = nn.Dropout(dropout)
        ssm_kwargs = ssm_kwargs or {}
        attn_kwargs = attn_kwargs or {}
        if block_type in ('attn', 'hybrid'):
            self.ln1 = RMSNorm(dim)
            self.attn = SlidingWindowCausalSelfAttention(
                dim, num_heads, max_seq_length=max_seq_length, **attn_kwargs)
        if block_type in ('ssm', 'hybrid'):
            self.ssm = MambaSSM(dim, **ssm_kwargs)
        self.ln2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden_dim)

    def forward(self, x, past_kv=None, use_cache=False, start_pos=0):
        present = None
        if self.block_type == 'attn':
            h, present = self.attn(self.ln1(x), past_kv, use_cache, start_pos)
            x = x + self.drop(h)
        elif self.block_type == 'ssm':
            x = x + self.drop(self.ssm(x))
        elif self.block_type == 'hybrid':
            h, present = self.attn(self.ln1(x), past_kv, use_cache, start_pos)
            x = x + self.drop(h) + self.drop(self.ssm(x))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x, present


def _parse_layer_plan(layer_plan, num_layers):
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

    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers,
                 hidden_dim, max_seq_length, dropout=0.0, tie_weights=True,
                 gradient_checkpointing=True,
                 layer_plan=None, ssm_d_state=16, ssm_d_inner_factor=1,
                 ssm_dt_rank=None, attn_window=0, attn_rel_bias=False):
        super(TransformerModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.gradient_checkpointing = gradient_checkpointing
        self.layer_plan = _parse_layer_plan(layer_plan, num_layers)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.drop = nn.Dropout(dropout)
        ssm_kwargs = dict(d_state=ssm_d_state, d_inner_factor=ssm_d_inner_factor, dt_rank=ssm_dt_rank)
        attn_kwargs = dict(window=attn_window, rel_bias=attn_rel_bias)
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, hidden_dim, block_type=bt,
                             dropout=dropout, max_seq_length=max_seq_length,
                             ssm_kwargs=ssm_kwargs, attn_kwargs=attn_kwargs)
            for bt in self.layer_plan
        ])
        self.ln_f = RMSNorm(embedding_dim)
        self.output_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        if tie_weights:
            self.output_head.weight = self.embedding.weight

        self._init_weights()

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

    def forward(self, src, src_mask=None, past_key_values=None, use_cache=False):
        # src: (batch, seq_len)；RoPE 在注意力内部按位置旋转，无需外部 PE
        x = self.embedding(src) * math.sqrt(self.embedding_dim)
        x = self.drop(x)
        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)
        presents = []
        start_pos = 0
        if use_cache:
            for pk in past_key_values:
                if pk is not None:
                    start_pos = pk[0].size(2)
                    break
        for i, block in enumerate(self.blocks):
            if self.training and self.gradient_checkpointing:
                x, present = checkpoint(block, x, past_key_values[i], use_cache, start_pos,
                                        use_reentrant=False)
            else:
                x, present = block(x, past_key_values[i], use_cache, start_pos)
            presents.append(present)
        x = self.ln_f(x)
        if use_cache:
            return self.output_head(x), presents
        return self.output_head(x)

    def generate(self, token_ids, max_length=50, temperature=1.0, top_k=50,
                  device='cpu', penalty_alpha=0.6, repetition_penalty=1.2,
                  ngram_fn=None, ngram_weight=0.0,
                  eos_id=3, pad_id=0, sep_id=4):
        self.eval()
        generated = list(token_ids)
        max_seq_length = self.max_seq_length
        eos_token_id = eos_id
        pad_token_id = pad_id
        sep_token_id = sep_id
        # 纯注意力模型可用 KV-cache 做增量解码（O(L)/步）；含 SSM 的模型回退全量重算
        use_cache = all(b == 'attn' for b in self.layer_plan)

        def sample_step(logits_t):
            next_token_logits = logits_t / temperature
            for prev_token in set(generated):
                if 0 <= prev_token < next_token_logits.shape[0]:
                    next_token_logits[prev_token] = next_token_logits[prev_token] / repetition_penalty
            if ngram_fn is not None and ngram_weight != 0.0:
                next_token_logits = next_token_logits + ngram_weight * ngram_fn(generated, device)
            next_token_logits[pad_token_id] = float('-inf')
            next_token_logits[sep_token_id] = float('-inf')
            min_length = max(3, len(token_ids) + 2)
            if len(generated) < min_length:
                next_token_logits[eos_token_id] = float('-inf')
            else:
                next_token_logits[eos_token_id] = next_token_logits[eos_token_id] - 5.0
            if top_k > 0 and top_k < next_token_logits.shape[0]:
                top_k_vals = torch.topk(next_token_logits, min(top_k, next_token_logits.shape[0]))[0]
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
                if next_token == eos_token_id and len(generated) >= max(3, len(token_ids) + 2):
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
