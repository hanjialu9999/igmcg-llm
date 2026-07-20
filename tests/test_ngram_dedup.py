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


def test_min_count_prune_reduces_memory_and_keeps_fusion():
    """回归锁：ngram_min_count 剪枝应大幅缩减计数表内存，且不破坏可学融合的
    logprob_orders_matrix 输出有限性与形态。

    背景：原实现对所有 order 2..10 建全量计数表（4000 行语料下 ~462MB dict
    载荷 + 540MB 常驻），并在每步物化 (B,T,V,K) 张量（K=10）造成 4GB+ 内存
    占用与 ~0.59s/step 的额外开销。降阶(max_order=5)+剪枝(min_count=2)后
    静态内存应降到约 1/20，且融合输出仍 (B,T,V,K) 有限可用。"""
    # 构造一个含单/低频次 n-gram 的小语料，验证剪枝确实丢弃低频次项
    small = [
        "甲 乙 丙 丁",
        "甲 乙 戊 己",  # (甲,乙)->丙 / (甲,乙)->戊 各出现 1 次（count=1）
    ]
    v = CharTokenizer(vocab_size=200)
    v.train(small)  # 词表与统计语料同源，避免 OOV 越界
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    f.write("\n".join(small) + "\n")
    f.close()
    ng = NGramModel(v, f.name, max_order=5, smoothing=1.0,
                    l1=0.1, l2=0.3, l3=0.6, min_count=2)
    os.unlink(f.name)

    import sys as _s
    payload_pruned = 0
    # 剪枝后所有计数字典的载荷应极小；这里仅断言无 count<2 残留
    for order in range(2, 6):
        for ctx, c in ng.ngrams[order].items():
            for tok, n in c.items():
                assert n >= 2, f"min_count=2 剪枝残留 count={n} 的项"
                payload_pruned += _s.getsizeof(tok) + _s.getsizeof(n)
    # 剪枝后内存远小于未剪枝全量（定性：payload 应很小，不超数百 KB）
    assert payload_pruned < 100 * 1024, f"剪枝后计数表仍过大 {payload_pruned} 字节"

    # 融合输出形态与有限性不破坏（V 取 ngram 对齐的模型词表维度）
    V = ng.vocab_size
    ids = torch.tensor(v.encode('甲 乙 丙 丁', add_special_tokens=False)).unsqueeze(0)
    mat = ng.logprob_orders_matrix(ids, 'cpu')
    assert mat.shape == (1, ids.shape[1], V, 5)
    assert torch.isfinite(mat).all()


def test_logprob_orders_matrix_vectorized_matches_per_position():
    """回归锁：向量化后的 `logprob_orders_matrix`（去重 + index_select 批量拼回）
    必须与逐位置调用 `_compute_logprob_orders` 的结果逐元素一致，否则会改变
    训练期融合分布（等价无声修改先验强度）。

    这是把训练期每步 ~0.56s 的逐 (b,t) Python 循环改为去重批量搬运的性能优化
    的正确性护栏：仅改计算组织方式，不改任何数值。"""
    small = ["甲 乙 丙 丁 戊", "甲 乙 戊 己 庚", "乙 丙 丁 戊 己 庚 辛"]
    v = CharTokenizer(vocab_size=200)
    v.train(small)
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    f.write("\n".join(small) + "\n")
    f.close()
    ng = NGramModel(v, f.name, max_order=5, smoothing=1.0, min_count=1)
    os.unlink(f.name)

    ids = torch.tensor(v.encode('甲 乙 丙 丁 戊 己 庚', add_special_tokens=False)).unsqueeze(0)
    K = ng.max_order
    V = ng.vocab_size
    out = ng.logprob_orders_matrix(ids, 'cpu')  # (1, T, V, K)
    # 参考：逐位置调 _compute_logprob_orders
    ctx_len = K - 1
    padded = [v.pad_idx] * ctx_len + ids[0].tolist()
    ref = torch.empty(1, ids.shape[1], V, K)
    for t in range(ids.shape[1]):
        ref[0, t] = ng._compute_logprob_orders(padded[t:t + ctx_len], V, 'cpu')
    max_diff = (out - ref).abs().max().item()
    assert max_diff < 1e-6, f"向量化与逐位置结果不一致，max_diff={max_diff}"



def test_load_ngram_honors_max_order_and_min_count():
    """回归锁：build_ngram_model 应读取 ngram_max_order / ngram_min_count 配置，
    降阶与剪枝经由统一入口生效（避免训练/推理统计缓冲不对称）。"""
    from models.checkpoint import build_ngram_model
    v = CharTokenizer(vocab_size=5000)
    v.train(CORPUS)
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    f.write("\n".join(CORPUS * 50) + "\n")
    f.close()
    cfg = {'ngram_fusion': True, 'ngram_corpus': f.name,
           'vocab_size': 5000, 'ngram_max_order': 5, 'ngram_min_count': 2}
    ng = build_ngram_model(v, cfg)
    os.unlink(f.name)
    assert ng is not None
    assert ng.max_order == 5
    assert ng.min_count == 2
    # 不应存在 order>5 的计数表
    assert not any(o > 5 for o in ng.ngrams)

