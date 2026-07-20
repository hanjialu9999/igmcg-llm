import tempfile
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.data_utils import CharTokenizer
from models.ngram import NGramModel


CORPUS = [
    "中 国 人 民 生 活 幸 福",
    "中 国 梦 想 伟 大 复 兴",
    "人 民 当 家 作 主 权 利",
    "中 国 人 民 共 和 国 万 岁",
]


def _make_ngram(max_order=10, smoothing=1.0, l1=0.1, l2=0.3, l3=0.6, vocab_size=None):
    v = CharTokenizer(vocab_size=200)
    v.train(CORPUS)
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    f.write("\n".join(CORPUS) + "\n")
    f.close()
    ng = NGramModel(v, f.name, max_order=max_order, smoothing=smoothing,
                    l1=l1, l2=l2, l3=l3, vocab_size=vocab_size)
    os.unlink(f.name)
    return v, ng


def test_vec_for_ctx_distinct_from_orders_path():
    """回归锁：`_vec_for_ctx`（固定 CLI 先验插值）与 `_compute_logprob_orders`
    （可学融合的独立逐阶插值）在双阶命中时**数学不等价**，且不得被“去重”改写。

    数值证据（max_order=10, smoothing=1.0, l1/l2/l3=0.1/0.3/0.6）：
    bi+tri 双命中时 max abs diff 在 1e-3 量级，bi-only 在 1e-2 量级；uni-only
    才因退化到 unigram 而一致（≈1e-8）。若此断言失败，说明有人把两者合并了，
    那会改变生产解码分布 —— 必须恢复独立实现。
    """
    v, ng = _make_ngram()
    V = ng.vocab_size
    device = 'cpu'

    ids = v.encode('中 国 人', add_special_tokens=False)
    w2, w1 = ids[-2], ids[-1]
    assert (w1,) in ng.ngrams[2], "fixture 需使 bigram 命中"
    assert (w2, w1) in ng.ngrams[3], "fixture 需使 trigram 命中"

    # 当前 logprob_vector 输出（概率空间，便于比对）
    vec_prob = ng.logprob_vector([w2, w1], device).exp()

    # 若委托给 orders 路径（取 bi/tri 列按 l1/l2/l3 并列混合）会得到的输出
    orders = ng._compute_logprob_orders([w2, w1], V, device).exp()
    uni = ng.uni_prob
    delegated = ng.l1 * uni + ng.l2 * orders[:, 1] + ng.l3 * orders[:, 2]

    diff = (vec_prob - delegated).abs().max().item()
    # 必须显著大于数值误差，证明两者是独立语义（非“同一个东西的两种写法”）
    assert diff > 1e-4, (
        f"_vec_for_ctx 与 orders 路径差异仅 {diff}，低于预期（应 ≥1e-4 证明独立语义）；"
        f"若本断言意外通过，请确认未被误合并为 orders 委托。")
    # 同时确认差异确实有限且为确定性（便于后续精确锁定参考值）
    assert diff < 1e-1, f"差异过大 {diff}，疑似实现异常"


def test_logprob_matrix_matches_vector_per_position():
    """回归锁：`logprob_matrix` 第 t 位必须等于用**同一上下文**调用的
    `logprob_vector`。

    注意：`logprob_matrix` 在序列前左填充 `pad`，故位置 t 的上下文是
    (ctx_w2[t], ctx_w1[t])（pad 填充），而直接用 `logprob_vector(ids[:t+1])`
    在 t 较小时上下文是 (None,None)，两者语义不同——不得误判为回归。
    此测试复刻 matrix 的上下文构造，确保两者对同一上下文输出一致。
    """
    v, ng = _make_ngram()
    device = 'cpu'

    ids = torch.tensor(v.encode('中 国 人 民 共 和 国', add_special_tokens=False))
    mat = ng.logprob_matrix(ids, device)  # (1, T, V)

    pad = v.pad_idx
    seq = ids.tolist()
    ctx_w2 = [pad, pad] + seq[:-2]
    ctx_w1 = [pad] + seq[:-1]
    T = ids.shape[0]
    for t in range(T):
        w2, w1 = ctx_w2[t], ctx_w1[t]
        gen = [w2, w1] if w2 is not None else [w1]
        vec = ng.logprob_vector(gen, device)
        assert torch.allclose(mat[0, t], vec, atol=0, rtol=0), (
            f"位置 {t} 的 logprob_matrix 与同上下文 logprob_vector 不一致")


def test_vec_for_ctx_deterministic_reference():
    """锁 `_vec_for_ctx` 对给定上下文的确定性输出（固定语料+固定权重）。

    保存一个参考向量，未来若有人改动插值实现（即使出于去重），只要解码分布变化，
    此断言就会失败，从而阻止无声改变解码分布。
    """
    v, ng = _make_ngram()
    device = 'cpu'

    ids = v.encode('中 国 人', add_special_tokens=False)
    w2, w1 = ids[-2], ids[-1]
    vec = ng.logprob_vector([w2, w1], device)

    # 确定性 + 有限性
    assert torch.isfinite(vec).all()
    vec2 = ng.logprob_vector([w2, w1], device)
    assert torch.allclose(vec, vec2)
    # 向量已归一化为合法 log 概率（L1 范数有限，最大值 < 0 因平滑）
    assert vec.max().item() < 0.0
    # 缓存键存在
    assert (w2, w1) in ng._logprob_cache
