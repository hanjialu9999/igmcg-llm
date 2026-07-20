from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import torch


class NGramModel:
    """统计语言模型（泛化 n-gram，1..max_order 阶）。

    解码期作为神经 LM 的先验，不改变模型、不需重训，专门改善字符级 LM 的局部连贯性。
    模型侧用可学 order_logits 做各阶加权混合（阶段8.7，替代固定的 l1/l2/l3 插值），
    本类只负责构建计数表与查表（logprob_orders_* 系列）。

    统一事实来源（`logprob_orders_*` 系列）：训练期可学融合走 `_compute_logprob_orders`
    + `_orders_cache` 单缓存，逐阶独立返回（各阶仅对自己与 unigram 用顶权混合），
    由可学 `ngram_order_logits` 加权。

    旧接口 `logprob_vector` / `logprob_matrix`（uni/bi/tri 固定插值）是**生产解码路径**
    （scripts/generate.py 的 `--ngram` 固定先验、transformer.py:303、_ngram_coherence 评分），
    经 `_vec_for_ctx` 实现**独立顺序嵌套插值**（uni→bi→tri 依次混合），与 orders 路径的
    逐阶独立插值数学上**不等价**（双阶命中时 max abs diff ≈ 1e-3，见 `_vec_for_ctx` 注释）。
    二者语义分离是有意的（固定 CLI 先验 vs 可学融合），故保留 `_vec_for_ctx` 独立实现，
    不合并以消除“双插值漂移”——反而明确标注其差异，避免误合并改变解码分布。
    """

    def __init__(self, vocab, corpus_file, max_order: int = 10, smoothing: float = 1.0,
                 l1: float = 0.1, l2: float = 0.3, l3: float = 0.6, vocab_size: Optional[int] = None,
                 min_count: int = 1):
        self.vocab = vocab
        self.max_order = max_order
        self.smoothing = smoothing
        self.l1, self.l2, self.l3 = l1, l2, l3
        # 计数剪枝阈值：单/低频次 n-gram（count < min_count）多为语料噪声，既浪费内存
        # 又拖累泛化；训练/融合默认剪掉（min_count=2），仅保留有统计意义的高频上下文。
        self.min_count = int(max(1, min_count))
        # vocab_size 决定统计缓冲维度：默认取 len(vocab)（语料实际覆盖的 token 数），
        # 但融合时需对齐模型词表（可能远大于语料覆盖），故允许显式覆盖。
        self.vocab_size = int(vocab_size) if vocab_size is not None else len(vocab)
        # 泛化 n-gram 存储：self.ngrams[order] = context_tuple → Counter(next_token)
        # order=1 → 无上下文（unigram），order=2 → (w1,)，order=3 → (w2,w1) ...
        self.ngrams = {o: defaultdict(Counter) for o in range(1, max_order + 1)}
        self.uni = Counter()  # 保留 unigram 作为快捷访问
        self._build(corpus_file)
        # 解码期同一上下文会被多个候选反复查询，缓存 logprob 向量避免重复建表计算
        self._logprob_cache: Dict = {}
        self._logprob_cache_max = 8192
        self._orders_cache_store: Dict = {}
        self._orders_cache_max = 8192

    def _build(self, corpus_file):
        # errors='replace' 避免脏语料含非法 UTF-8 序列时直接抛 UnicodeDecodeError
        with open(corpus_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                ids = self.vocab.encode(line, add_special_tokens=False)
                # 左填充 2 个 pad 保证 context 窗口足够
                ids = [self.vocab.pad_idx] * 2 + ids
                for i in range(2, len(ids)):
                    self.uni[ids[i]] += 1
                    # 构建 order 2..max_order 的 n-gram
                    for order in range(2, self.max_order + 1):
                        if i >= order - 1:
                            ctx = tuple(ids[i - order + 1: i])  # (order-1) 个上下文 token
                            self.ngrams[order][ctx][ids[i]] += 1
        # 预计算（加速解码期每次调用的 logprob_vector）
        V = self.vocab_size
        self.uni_total = sum(self.uni.values()) + self.smoothing * V
        # 各阶上下文总计数（先按 min_count 剪枝低频次计数，降低内存并按住噪声）
        for order in range(2, self.max_order + 1):
            for ctx in list(self.ngrams[order].keys()):
                c = self.ngrams[order][ctx]
                dead = [t for t, n in c.items() if n < self.min_count]
                for t in dead:
                    del c[t]
                if not c:
                    del self.ngrams[order][ctx]
        self.ngram_totals = {}
        for order in range(2, self.max_order + 1):
            self.ngram_totals[order] = {
                ctx: sum(c.values()) + self.smoothing * V
                for ctx, c in self.ngrams[order].items()
            }
        u = torch.full((V,), self.smoothing)
        for t, c in self.uni.items():
            if 0 <= t < V:
                u[t] += c
        self.uni_prob = (u / u.sum()).clone()

    # ------------------------------------------------------------------
    # 统一事实来源：逐阶 n-gram log 概率（含 unigram 兜底）
    # ------------------------------------------------------------------
    def _interp_weights(self, order):
        """返回 order 阶插值的子阶权重 (w1, w2, ..., w_{order})，归一化和为 1。
        默认方案：指数衰减，高阶权重更大（如 order=3 → 0.1/0.3/0.6）。"""
        if order <= 3:
            # 向后兼容：保留原 l1/l2/l3
            ws = [self.l1, self.l2, self.l3][:order]
        else:
            # 泛化：指数衰减 w_i = 0.5^(order-i)，归一化
            ws = [0.5 ** (order - i) for i in range(1, order + 1)]
        s = sum(ws)
        return [w / s for w in ws]

    def _compute_logprob_orders(self, ctx_tokens: List[int], V: int, device) -> torch.Tensor:
        """泛化版：返回各阶 n-gram 的 log 概率向量（未插值），shape (V, max_order)。
        ctx_tokens: 完整上下文 token 列表（最近 order-1 个 token），长度 >= max_order-1。
        order 0 = unigram，order k (k>=1) = 用 ctx_tokens[-k:] 作为上下文的 k+1 gram。
        各阶独立返回，由模型学 order 权重（自选 n 的数量/占比）做可微混合，
        替代固定 l1/l2/l3 插值。无对应上下文的阶用 unigram 兜底。"""
        K = self.max_order
        out = torch.empty(V, K, device=device)
        uni = self.uni_prob.to(device)           # (V,) 已归一化 unigram
        out[:, 0] = torch.log(uni + 1e-10)     # unigram 兜底
        # 泛化：order 1..K-1 对应 2-gram .. K-gram
        for k in range(1, K):
            order = k + 1  # n-gram 阶数
            vec = uni.clone()
            # 获取该阶所需的上下文 (order-1) 个 token
            if len(ctx_tokens) >= order - 1:
                ctx = tuple(ctx_tokens[-(order - 1):])  # 最近 order-1 个 token
            else:
                ctx = None
            if ctx is not None and ctx in self.ngrams.get(order, {}):
                counter = self.ngrams[order][ctx]
                total = self.ngram_totals.get(order, {}).get(ctx, self.smoothing * V)
                idx = torch.tensor(list(counter.keys()), dtype=torch.long, device=device)
                if idx.numel():
                    counts = torch.tensor(list(counter.values()), device=device)
                    p_order = (counts + self.smoothing) / total
                    # 低阶混合（简化：用 unigram 兜底）
                    ws = self._interp_weights(order)
                    vec[idx] = ws[-1] * p_order + (1 - ws[-1]) * uni[idx]
                    vec = vec / vec.sum()
            out[:, k] = torch.log(vec + 1e-10)
        return out

    @property
    def _orders_cache(self):
        if not hasattr(self, '_orders_cache_store'):
            self._orders_cache_store = {}
        return self._orders_cache_store

    def logprob_orders_matrix(self, ids: torch.Tensor, device) -> torch.Tensor:
        """泛化版：返回逐阶 n-gram log 概率，shape (B, T, V, max_order)。
        供 TransformerModel 用可学 order 权重做混合（模型自选各阶占比）。
        上下文窗口长度 = max_order-1（如 max_order=10 → 9 token 上下文）。"""
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        B, T = ids.shape
        K = self.max_order
        V = self.vocab_size
        out = torch.empty(B, T, V, K, device=device)
        pad = self.vocab.pad_idx if hasattr(self.vocab, 'pad_idx') else 0
        ctx_len = max(1, K - 1)  # 上下文窗口长度
        for b in range(B):
            seq = ids[b].tolist()
            # 左填充 ctx_len 个 pad，保证每个位置都有足够上下文
            padded = [pad] * ctx_len + seq
            for t in range(T):
                # 位置 t 的上下文窗口 = padded[t : t+ctx_len]
                ctx_tokens = padded[t: t + ctx_len]
                ck = tuple(ctx_tokens)
                if ck in self._orders_cache:
                    out[b, t] = self._orders_cache[ck].to(device)
                else:
                    v = self._compute_logprob_orders(ctx_tokens, V, device)
                    if len(self._orders_cache) > self._orders_cache_max:
                        self._orders_cache.clear()
                    self._orders_cache[ck] = v.cpu()
                    out[b, t] = v
        return out

    def logprob_orders_incremental(self, ctx2: torch.Tensor, new_ids: torch.Tensor, device):
        """增量解码：给定 (B,ctx_len) 滚动上下文 与 (B,T) 新 token，
        仅计算新 token 各位置的逐阶 log 概率 (B,T,V,K)，不重建整段 ctx（避免 O(T^2)）。
        ctx2: (B, ctx_len)，其中 ctx_len = max_order-1，含历史 token 用于构建上下文窗口。
        复用 _orders_cache：上下文相同的位置直接命中，与全量路径完全一致。"""
        if new_ids.dim() == 1:
            new_ids = new_ids.unsqueeze(0)
        B, T = new_ids.shape
        K = self.max_order
        V = self.vocab_size
        ctx_len = max(1, K - 1)
        # 拼接滚动上下文 + 新 token：[ctx0..ctx_{L-1}, new0..new_{T-1}]
        full = torch.cat([ctx2, new_ids], dim=1)                     # (B, ctx_len+T)
        out = torch.empty(B, T, V, K, device=device)
        for b in range(B):
            for t in range(T):
                # 位置 t 的上下文窗口 = full[t: t+ctx_len]
                ctx_tokens = full[b, t: t + ctx_len].tolist()
                ck = tuple(ctx_tokens)
                if ck in self._orders_cache:
                    out[b, t] = self._orders_cache[ck].to(device)
                else:
                    v = self._compute_logprob_orders(ctx_tokens, V, device)
                    if len(self._orders_cache) > self._orders_cache_max:
                        self._orders_cache.clear()
                    self._orders_cache[ck] = v.cpu()
                    out[b, t] = v
        return out

    # ------------------------------------------------------------------
    # 向后兼容薄封装（保持 train.py / generate.py 旧调用签名）
    # ------------------------------------------------------------------
    def _vec_for_ctx(self, w2, w1, device):
        """上下文 (w2,w1) 下的 log 概率向量 (V,)，按上下文缓存。

        **有意保留的独立插值路径（非 `_compute_logprob_orders` 的薄封装）。**

        §6.1 去重调查发现本函数与 `_compute_logprob_orders` 在数学上**不等价**，
        不可为去重而改写，否则会无声改变生产解码分布（scripts/generate.py 的
        `--ngram` 固定先验、transformer.py:303 的 `ngram_weight*ngram_fn`、
        `_ngram_coherence` 评分均走此路）。

        差异来源（数值证据，max_order=10, smoothing=1.0, l1/l2/l3=0.1/0.3/0.6）：
        - 双阶命中（bi+tri 都命中上下文）时，两路对 V 向量 max abs diff ≈ 1.4e-3
          （bi-only 情形 ≈ 1.0e-2；uni-only ≈ 7e-9，因退化到 unigram 才一致）。
        - 混合方式不同：
            * 本函数：从 unigram 起**顺序嵌套混合**——先 `vec=0.7*uni+0.3*p_bi`，
              再 `vec=0.4*vec+0.6*p_tri`，即 `0.6*p_tri+0.28*uni+0.12*p_bi`，
              bigram 信息会“漏进” trigram 结果。
            * `_compute_logprob_orders`：每个 order 阶**独立**用其顶权 `ws[-1]`
              仅与该阶对 unigram 混合（tri 列 = `0.6*p_tri+0.4*uni`），各阶并列
              返回，由可学 `ngram_order_logits` 做融合（服务训练期可学插值），
              而非固定 l1/l2/l3 顺序混合。
        - 语义分离是有意的：orders 路径服务**可学融合**；vector 路径服务**固定 CLI
          先验**。两者权重定义/混合方式/归一化位置均不同，强行统一会漂移解码分布。

        因此保留独立实现，仅在此明确标注，杜绝后续误判为“重复可合并代码”。
        """
        cache_key = (w2, w1)
        if cache_key in self._logprob_cache:
            return self._logprob_cache[cache_key].to(device)
        ctx_tokens = [c for c in (w2, w1) if c is not None]
        V = self.vocab_size
        # 取 tri/unigram 等价的逐位置 logp：仅用 unigram 兜底 + 高阶叠加，
        # 与旧 _compute_logprob 等价（高阶优先、低阶兜底）。
        vec = self._uni.clone()
        order_data = []
        for order in range(2, self.max_order + 1):
            if order == 2 and w1 is not None:
                ctx = (w1,)
            elif order == 3 and w2 is not None:
                ctx = (w2, w1)
            else:
                continue
            grams = self.ngrams.get(order, {})
            if ctx in grams:
                counter = grams[ctx]
                total = self.ngram_totals.get(order, {}).get(ctx, self.smoothing * V)
                order_data.append((order, counter, total))
        if order_data:
            max_hit = max(o for o, _, _ in order_data)
            ws = self._interp_weights(max_hit)
            for order, counter, total in order_data:
                idx = torch.tensor(list(counter.keys()), dtype=torch.long, device=device)
                if idx.numel():
                    counts = torch.tensor(list(counter.values()), device=device)
                    p = (counts + self.smoothing) / total
                    w = ws[order - 1] if order <= len(ws) else ws[-1]
                    vec[idx] = vec[idx] * (1 - w) + p * w
        vec = vec / vec.sum()
        vec = torch.log(vec + 1e-10)
        if len(self._logprob_cache) > self._logprob_cache_max:
            self._logprob_cache.clear()
        self._logprob_cache[cache_key] = vec.cpu()
        return vec

    @property
    def _uni(self):
        return self.uni_prob

    def logprob_vector(self, generated, device):
        """返回词表每个 token 的 log 概率向量（uni/bi/tri 插值，三元真正参与）。
        只在与当前上下文相关的少量 token 上计算，避免遍历全词表。结果按上下文
        (w2,w1) 缓存，因为解码期同一上下文会被多个候选反复查询。"""
        V = self.vocab_size
        w2 = generated[-2] if len(generated) >= 2 else None
        w1 = generated[-1] if len(generated) >= 1 else None
        return self._vec_for_ctx(w2, w1, device)

    def logprob_matrix(self, ids: torch.Tensor, device) -> torch.Tensor:
        """向量化版本：输入 (B,T) 的 token id，返回 (B,T,V) 的 n-gram log 概率矩阵，
        每个位置 t 用其前 2 个 token 上下文 (ids[t-2], ids[t-1]) 插值。"""
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        B, T = ids.shape
        V = self.vocab_size
        out = torch.empty(B, T, V, device=device)
        pad = self.vocab.pad_idx if hasattr(self.vocab, 'pad_idx') else 0
        for b in range(B):
            seq = ids[b].tolist()
            ctx_w2 = [pad, pad] + seq[:-2]
            ctx_w1 = [pad] + seq[:-1]
            for t in range(T):
                w2 = ctx_w2[t]
                w1 = ctx_w1[t]
                out[b, t] = self._vec_for_ctx(w2, w1, device)
        return out
