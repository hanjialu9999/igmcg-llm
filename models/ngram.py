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
        # 构建期预存张量化计数（idx + 已平滑概率 p=(count+smooth)/total），避免训练/解码期
        # 每次 _compute_logprob_orders 都把 Counter 现转张量 + 做除法；运行时仅 .to(device) +
        # 一次 scatter，省去 Python 层 torch.tensor(list(...)) 转换与浮点除法开销（DML 小 kernel 友好）。
        self._ngram_tensors: Dict = {}
        for order in range(2, self.max_order + 1):
            self._ngram_tensors[order] = {}
            for ctx, c in self.ngrams[order].items():
                total = self.ngram_totals[order][ctx]
                idx = torch.tensor(list(c.keys()), dtype=torch.long)
                counts = torch.tensor(list(c.values()), dtype=torch.float32)
                p = (counts + self.smoothing) / total
                self._ngram_tensors[order][ctx] = (idx, p)
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

    def _ensure_dev_caches(self, device):
        """惰性设备缓存：首次访问某设备时，一次性把 uni_prob + ngram_tensors 传到 device。
        后续 _compute_logprob_orders 直接用设备版本，免去逐次 .to(device) 的小传输开销
        （DML 上每个 .to(device) 有固定启动税，U 个唯一上下文 × K-1 阶 = 数千次/步）。"""
        dev_key = str(device)
        if hasattr(self, '_dev_cache_key') and self._dev_cache_key == dev_key:
            return
        self._uni_dev_tensor = self.uni_prob.to(device)
        self._ngt_dev: Dict = {}
        for order, ctxs in self._ngram_tensors.items():
            self._ngt_dev[order] = {ctx: (idx.to(device), p.to(device))
                                     for ctx, (idx, p) in ctxs.items()}
        self._dev_cache_key = dev_key

    def _compute_logprob_orders(self, ctx_tokens: List[int], V: int, device) -> torch.Tensor:
        """泛化版：返回各阶 n-gram 的 log 概率向量（未插值），shape (V, max_order)。
        ctx_tokens: 完整上下文 token 列表（最近 order-1 个 token），长度 >= max_order-1。
        order 0 = unigram，order k (k>=1) = 用 ctx_tokens[-k:] 作为上下文的 k+1 gram。
        各阶独立返回，由模型学 order 权重（自选 n 的数量/占比）做可微混合，
        替代固定 l1/l2/l3 插值。无对应上下文的阶用 unigram 兜底。

        性能：计数已在 _build 期预存为 (idx_tensor, p_tensor)，首次调用时一次性
        传到 device 并缓存（_ensure_dev_caches），后续仅查表 + 一次 scatter；
        各阶并行叠加到 (K-1, V) 背景矩阵后统一归一化，避免逐阶
        vec.clone()/vec.sum() 的重复全 V 设备 reduce。"""
        K = self.max_order
        # 设备缓存：uni_prob + ngram_tensors 一次性传到 device（惰性，仅首次）
        self._ensure_dev_caches(device)
        uni = self._uni_dev_tensor                     # (V,) 已归一化 unigram
        # (K-1, V) 背景：各高阶阶从 unigram 起步，逐阶独立叠加自身命中修正
        base = uni.unsqueeze(0).expand(K - 1, V).clone()   # (K-1, V)
        for k in range(1, K):
            order = k + 1  # n-gram 阶数
            if len(ctx_tokens) >= order - 1:
                ctx = tuple(ctx_tokens[-(order - 1):])  # 最近 order-1 个 token
            else:
                ctx = None
            if ctx is not None and ctx in self._ngt_dev.get(order, {}):
                idx, p = self._ngt_dev[order][ctx]     # 已在 device，无需 .to(device)
                ws = self._interp_weights(order)
                w = ws[-1]
                # vec[idx] = w*p + (1-w)*uni[idx]，其余位置保持 uni
                base[k - 1, idx] = w * p + (1.0 - w) * uni[idx]
        # 逐阶归一化后取 log（每阶独立归一，与逐阶 vec/vec.sum() 完全等价）
        base = torch.log(base / base.sum(dim=-1, keepdim=True) + 1e-10)  # (K-1, V)
        out = torch.empty(V, K, device=device)
        out[:, 0] = torch.log(uni + 1e-10)       # unigram 兜底（order 0）
        out[:, 1:] = base.T                      # (V, K-1) -> (V, K) 高阶
        return out

    @property
    def _orders_cache(self):
        if not hasattr(self, '_orders_cache_store'):
            self._orders_cache_store = {}
        return self._orders_cache_store

    def logprob_orders_matrix(self, ids: torch.Tensor, device) -> torch.Tensor:
        """泛化版：返回逐阶 n-gram log 概率，shape (B, T, V, max_order)。
        供 TransformerModel 用可学 order 权重做混合（模型自选各阶占比）。
        上下文窗口长度 = max_order-1（如 max_order=10 → 9 token 上下文）。

        性能：训练期 batch 内各位置的上下文大量重复，逐 (b,t) 调
        `_compute_logprob_orders` + 设备小 kernel 在 DML 上极慢。故改为
        先收集全部上下文并去重，仅对【唯一上下文】各算一次（命中 `_orders_cache`
        则免算），再用 `index_select` 一次性批量搬回 (B,T,V,K)。数值与逐位置
        循环完全一致（同一 `_compute_logprob_orders` + 同一缓存键）。"""
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        B, T = ids.shape
        K = self.max_order
        V = self.vocab_size
        pad = self.vocab.pad_idx if hasattr(self.vocab, 'pad_idx') else 0
        ctx_len = max(1, K - 1)  # 上下文窗口长度
        # 1) 收集每个位置的上下文键，并去重
        # 一次性把 ids 搬到 CPU 再 .tolist()，用1次传输替代 B 次 GPU-CPU 同步
        ids_cpu = ids.cpu() if ids.is_cuda or (hasattr(ids, 'device') and ids.device.type in ('cuda', 'privateuseone')) else ids
        ctx_keys: List[Tuple[int, ...]] = []
        pos_to_key: List[int] = []
        uniq: Dict[Tuple[int, ...], int] = {}
        for b in range(B):
            seq = ids_cpu[b].tolist()
            padded = [pad] * ctx_len + seq
            for t in range(T):
                ck = tuple(padded[t: t + ctx_len])
                pos_to_key.append(ck)
                if ck not in uniq:
                    uniq[ck] = len(ctx_keys)
                    ctx_keys.append(ck)
        # 2) 仅对唯一上下文计算（命中缓存则免算），结果堆叠为 (U, V, K) 后搬设备
        #    优化：缓存命中项的 .to(device) 批量传输——DML 上逐张小传输有固定开销，
        #    warmup 后绝大多数上下文命中缓存，逐张 .to(device) 成为瓶颈。
        #    改为 stack 一次 → .to(device) 一次 → unbind 回各位置。
        cached_idxs: List[int] = []
        cached_cpu: List[torch.Tensor] = []
        uncached_idxs: List[int] = []
        for i, ck in enumerate(ctx_keys):
            if ck in self._orders_cache:
                cached_idxs.append(i)
                cached_cpu.append(self._orders_cache[ck])
            else:
                uncached_idxs.append(i)

        uniq_vecs: List[Optional[torch.Tensor]] = [None] * len(ctx_keys)
        # 缓存命中：批量传输（单次 .to(device) 替代 U_cached 次）
        if cached_cpu:
            cached_stack = torch.stack(cached_cpu, dim=0).to(device)  # (U_cached, V, K)
            for j, idx in enumerate(cached_idxs):
                uniq_vecs[idx] = cached_stack[j]
        # 未命中：逐个计算（结果已在 device）
        for idx in uncached_idxs:
            ck = ctx_keys[idx]
            v = self._compute_logprob_orders(ck, V, device)  # (V, K) 已在 device
            if len(self._orders_cache) > self._orders_cache_max:
                self._orders_cache.clear()
            self._orders_cache[ck] = v.cpu()
            uniq_vecs[idx] = v
        # 3) 向量化填充：stack 唯一上下文结果为 (U,V,K)，用 index_select 一次性搬回 (B*T,V,K)。
        #    旧路径逐 i 做 mask + fancy index，在 DML 上每次都是 GPU 同步（U 大时极慢）。
        #    stack(U,V,K) 峰值内存 = U*V*K = 8000*3*U bytes，U≤B*T=2048 时 ≤ 48MB，可接受。
        stacked = torch.stack(uniq_vecs, dim=0)  # (U, V, K)
        key_idx_tensor = torch.tensor([uniq[ck] for ck in pos_to_key],
                                      dtype=torch.long, device=device)
        out = stacked.index_select(0, key_idx_tensor.view(-1)).view(B, T, V, K)
        return out

    def logprob_orders_incremental(self, ctx2: torch.Tensor, new_ids: torch.Tensor, device):
        """增量解码：给定 (B,ctx_len) 滚动上下文 与 (B,T) 新 token，
        仅计算新 token 各位置的逐阶 log 概率 (B,T,V,K)，不重建整段 ctx（避免 O(T^2)）。
        ctx2: (B, ctx_len)，其中 ctx_len = max_order-1，含历史 token 用于构建上下文窗口。
        复用 _orders_cache：上下文相同的位置直接命中，与全量路径完全一致。

        性能优化（仿 logprob_orders_matrix）：收集唯一上下文 → 批量 stack .to(device) →
        index_select 一次性填充，消除逐 (b,t) .to(device) 的 DML 启动税。数值与逐位置循环等价。"""
        if new_ids.dim() == 1:
            new_ids = new_ids.unsqueeze(0)
        B, T = new_ids.shape
        K = self.max_order
        V = self.vocab_size
        ctx_len = max(1, K - 1)
        # 拼接滚动上下文 + 新 token：[ctx0..ctx_{L-1}, new0..new_{T-1}]
        full = torch.cat([ctx2, new_ids], dim=1)                     # (B, ctx_len+T)
        full_cpu = full.cpu() if full.is_cuda or (hasattr(full, 'device') and full.device.type in ('cuda', 'privateuseone')) else full
        # 1) 收集每个位置的上下文键并去重（与 logprob_orders_matrix 同模式）
        ctx_keys: List[Tuple[int, ...]] = []
        pos_to_key: List[int] = []
        uniq: Dict[Tuple[int, ...], int] = {}
        for b in range(B):
            seq = full_cpu[b].tolist()
            for t in range(T):
                ck = tuple(seq[t: t + ctx_len])
                if ck not in uniq:
                    uniq[ck] = len(ctx_keys)
                    ctx_keys.append(ck)
                pos_to_key.append(uniq[ck])
        # 2) 仅对唯一上下文计算（命中缓存免算），缓存命中项批量传输
        cached_idxs: List[int] = []
        cached_cpu: List[torch.Tensor] = []
        uncached_idxs: List[int] = []
        for i, ck in enumerate(ctx_keys):
            if ck in self._orders_cache:
                cached_idxs.append(i)
                cached_cpu.append(self._orders_cache[ck])
            else:
                uncached_idxs.append(i)
        uniq_vecs: List[Optional[torch.Tensor]] = [None] * len(ctx_keys)
        if cached_cpu:
            cached_stack = torch.stack(cached_cpu, dim=0).to(device)  # (U_cached, V, K)
            for j, idx in enumerate(cached_idxs):
                uniq_vecs[idx] = cached_stack[j]
        for idx in uncached_idxs:
            ck = ctx_keys[idx]
            v = self._compute_logprob_orders(list(ck), V, device)
            if len(self._orders_cache) > self._orders_cache_max:
                self._orders_cache.clear()
            self._orders_cache[ck] = v.cpu()
            uniq_vecs[idx] = v
        # 3) 向量化填充：stack + index_select 一次性搬回 (B,T,V,K)
        stacked = torch.stack(uniq_vecs, dim=0)  # (U, V, K)
        key_idx_tensor = torch.tensor(pos_to_key, dtype=torch.long, device=device)
        out = stacked.index_select(0, key_idx_tensor.view(-1)).view(B, T, V, K)
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
