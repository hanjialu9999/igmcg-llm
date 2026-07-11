import torch
import torch.nn as nn
import math
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


class RotaryEmbedding(nn.Module):
    """旋转位置编码 RoPE：对 Q/K 按位置旋转，天然支持长度外推。"""
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, q, k):
        # q, k: (batch, heads, seq, head_dim)
        t = torch.arange(q.size(2), device=q.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)          # (seq, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)         # (seq, head_dim)
        cos = emb.cos()[None, None, :, :]              # (1, 1, seq, head_dim) 可广播到 (B, H, T, D)
        sin = emb.sin()[None, None, :, :]
        return self._rope_apply(q, cos, sin), self._rope_apply(k, cos, sin)

    @staticmethod
    def _rope_apply(x, cos, sin):
        d = x.size(-1) // 2
        x1, x2 = x[..., :d], x[..., d:]
        rot = torch.cat((-x2, x1), dim=-1)
        return x * cos.to(x.dtype) + rot * sin.to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.rope(q, k)
        out = scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


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
    """Pre-LN Transformer 块：RMSNorm -> 因果注意力 -> RMSNorm -> SwiGLU。"""
    def __init__(self, dim, num_heads, hidden_dim):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads)
        self.ln2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerModel(nn.Module):
    """现代 decoder-only 语言模型（Pre-LN + RMSNorm + RoPE + SwiGLU + 权重共享）。

    架构固定为 2026 主流小模型配方；如需经典变体可后续扩展 norm/ffn/pos 选项。
    """

    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers,
                 hidden_dim, max_seq_length, dropout=0.0, tie_weights=True,
                 gradient_checkpointing=True):
        super(TransformerModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.gradient_checkpointing = gradient_checkpointing

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])
        self.ln_f = RMSNorm(embedding_dim)
        # 权重共享：输出头复用 embedding 权重
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

    def forward(self, src, src_mask=None):
        # src: (batch, seq_len)；RoPE 在注意力内部按位置旋转，无需外部 PE
        x = self.embedding(src) * math.sqrt(self.embedding_dim)
        x = self.drop(x)
        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)
        return self.output_head(x)

    def generate(self, token_ids, max_length=50, temperature=1.0, top_k=50,
                 device='cpu', penalty_alpha=0.6, repetition_penalty=1.2):
        self.eval()
        generated = token_ids.copy()
        max_seq_length = self.max_seq_length
        eos_token_id = 3
        pad_token_id = 0
        sep_token_id = 4

        with torch.no_grad():
            for _ in range(max_length):
                if len(generated) > max_seq_length:
                    input_ids = torch.tensor([generated[-max_seq_length:]], dtype=torch.long).to(device)
                else:
                    input_ids = torch.tensor([generated], dtype=torch.long).to(device)

                logits = self.forward(input_ids)
                next_token_logits = logits[0, -1, :] / temperature

                for prev_token in set(generated):
                    if 0 <= prev_token < next_token_logits.shape[0]:
                        next_token_logits[prev_token] = next_token_logits[prev_token] / repetition_penalty

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
                    next_token_logits = logits[0, -1, :] / temperature
                    next_token_logits[pad_token_id] = float('-inf')

                probs = torch.softmax(next_token_logits, dim=-1)
                if probs.max() < 0.01:
                    break

                next_token = torch.multinomial(probs, num_samples=1).item()
                generated.append(next_token)
                if next_token == eos_token_id and len(generated) >= min_length:
                    break

        return generated
