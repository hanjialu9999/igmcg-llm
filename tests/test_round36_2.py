"""第三十六轮-2 回归测试：ngram _compute_logprob_orders 循环优化 + _interp_weights 缓存。

验证 R36-2 优化：
1. _interp_weights 缓存正确性（首次计算 vs 缓存命中数值一致）
2. _interp_weights 缓存命中后返回等值列表
3. _compute_logprob_orders 优化后与参考实现数值等价
4. 边界条件：ctx 长度不足 / order 不存在 / ctx 未命中
5. 多阶命中场景
6. _vec_for_ctx 路径也受益于缓存
7. logprob_orders_matrix 集成（去重 + index_select 路径不回归）
8. logprob_orders_incremental 集成（增量解码不回归）
"""
import tempfile
import os
import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.data_utils import CharTokenizer
from models.ngram import NGramModel


CORPUS = [
    "中 国 人 民 生 活 幸 福",
    "中 国 梦 想 伟 大 复 兴",
    "人 民 当 家 作 主 权 利",
    "中 国 人 民 共 和 国 万 岁",
    "中 国 人 民 大 团 结 万 岁",
    "中 国 人 民 幸 福 生 活 好",
]


def _make_ngram(max_order=10, smoothing=1.0, l1=0.1, l2=0.3, l3=0.6, vocab_size=None,
                min_count=1):
    v = CharTokenizer(vocab_size=200)
    v.train(CORPUS)
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    f.write("\n".join(CORPUS) + "\n")
    f.close()
    ng = NGramModel(v, f.name, max_order=max_order, smoothing=smoothing,
                    l1=l1, l2=l2, l3=l3, vocab_size=vocab_size, min_count=min_count)
    os.unlink(f.name)
    return v, ng


def _reference_interp_weights(ng, order):
    """参考实现（无缓存）——用于验证缓存值正确。"""
    if order <= 3:
        ws = [ng.l1, ng.l2, ng.l3][:order]
    else:
        ws = [0.5 ** (order - i) for i in range(1, order + 1)]
    s = sum(ws)
    return [w / s for w in ws]


def _reference_compute_logprob_orders(ng, ctx_tokens, V, device):
    """参考实现（无优化）——用于验证优化后数值等价。

    复刻 R35 的逻辑（lerp + 无缓存 _interp_weights），不含 R36-2 的预检查/缓存。
    """
    K = ng.max_order
    ng._ensure_dev_caches(device)
    uni = ng._uni_dev_tensor
    base = uni.unsqueeze(0).expand(K - 1, V).clone()
    for k in range(1, K):
        order = k + 1
        if len(ctx_tokens) >= order - 1:
            ctx = tuple(ctx_tokens[-(order - 1):])
        else:
            ctx = None
        if ctx is not None and ctx in ng._ngt_dev.get(order, {}):
            idx, p = ng._ngt_dev[order][ctx]
            ws = _reference_interp_weights(ng, order)
            w = ws[-1]
            u_idx = uni[idx]
            base[k - 1, idx] = torch.lerp(u_idx, p, w)
    base = torch.log(base / base.sum(dim=-1, keepdim=True) + 1e-10)
    out = torch.empty(V, K, device=device)
    out[:, 0] = torch.log(uni + 1e-10)
    out[:, 1:] = base.T
    return out


# ===== _interp_weights 缓存测试 =====

def test_interp_weights_cache_correctness():
    """R36-2-1: _interp_weights 缓存值与参考实现数值一致。"""
    _, ng = _make_ngram()
    for order in range(1, ng.max_order + 2):  # 含越界 order
        cached = ng._interp_weights(order)
        ref = _reference_interp_weights(ng, order)
        assert len(cached) == len(ref), f"order={order} 长度不一致"
        for a, b in zip(cached, ref):
            assert abs(a - b) < 1e-12, f"order={order} 权重不一致: {a} vs {b}"


def test_interp_weights_cache_hit():
    """R36-2-2: 缓存命中后返回等值列表（第二次调用结果与第一次一致）。"""
    _, ng = _make_ngram()
    for order in range(2, ng.max_order + 1):
        first = ng._interp_weights(order)
        second = ng._interp_weights(order)
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert abs(a - b) < 1e-15, f"order={order} 缓存不一致"


def test_interp_weights_cache_attribute_exists():
    """R36-2-3: 首次调用后 _interp_w_cache 属性存在。"""
    _, ng = _make_ngram()
    assert not hasattr(ng, '_interp_w_cache'), "缓存不应在调用前存在"
    ng._interp_weights(2)
    assert hasattr(ng, '_interp_w_cache'), "缓存应在首次调用后存在"
    assert 2 in ng._interp_w_cache


def test_interp_weights_cache_normalization():
    """R36-2-4: 缓存的权重列表归一化（和为 1）。"""
    _, ng = _make_ngram()
    for order in range(1, ng.max_order + 2):
        ws = ng._interp_weights(order)
        assert abs(sum(ws) - 1.0) < 1e-12, f"order={order} 权重和={sum(ws)} != 1"


def test_interp_weights_cache_order_out_of_range():
    """R36-2-5: 超出 max_order 的 order 也能正确计算（泛化公式）。"""
    _, ng = _make_ngram(max_order=5)
    # order=10 超出 max_order=5，但 _interp_weights 仍能计算（泛化公式）
    ws = ng._interp_weights(10)
    assert len(ws) == 10
    assert abs(sum(ws) - 1.0) < 1e-12


# ===== _compute_logprob_orders 数值等价性测试 =====

def test_compute_logprob_orders_equivalence():
    """R36-2-6: 优化后 _compute_logprob_orders 与参考实现数值等价。"""
    v, ng = _make_ngram()
    V = ng.vocab_size
    device = 'cpu'
    ids = v.encode('中 国 人', add_special_tokens=False)
    out_opt = ng._compute_logprob_orders(ids, V, device)
    out_ref = _reference_compute_logprob_orders(ng, ids, V, device)
    assert torch.allclose(out_opt, out_ref, atol=1e-7), \
        f"优化后不等价: max_diff={(out_opt - out_ref).abs().max().item():.2e}"


def test_compute_logprob_orders_multi_hit_equivalence():
    """R36-2-7: 多阶命中场景下数值等价。"""
    v, ng = _make_ngram(max_order=5)
    V = ng.vocab_size
    device = 'cpu'
    # 用足够长的上下文触发多阶命中
    ids = v.encode('中 国 人 民', add_special_tokens=False)
    if len(ids) >= 4:
        out_opt = ng._compute_logprob_orders(ids, V, device)
        out_ref = _reference_compute_logprob_orders(ng, ids, V, device)
        assert torch.allclose(out_opt, out_ref, atol=1e-7), \
            f"多阶命中不等价: max_diff={(out_opt - out_ref).abs().max().item():.2e}"


def test_compute_logprob_orders_short_ctx():
    """R36-2-8: ctx 长度不足时正确跳过高阶（只算能命中的阶）。"""
    v, ng = _make_ngram(max_order=10)
    V = ng.vocab_size
    device = 'cpu'
    # 只有 1 个 token，只能算 order=2（需 1 个上下文）
    ids = v.encode('中', add_special_tokens=False)
    out = ng._compute_logprob_orders(ids, V, device)
    assert out.shape == (V, ng.max_order)
    # unigram 列应与 uni_prob 的 log 一致
    expected_uni = torch.log(ng.uni_prob + 1e-10)
    assert torch.allclose(out[:, 0], expected_uni, atol=1e-7), "unigram 列不一致"


def test_compute_logprob_orders_empty_ctx():
    """R36-2-9: 空上下文时所有高阶退化为 unigram。"""
    v, ng = _make_ngram(max_order=5)
    V = ng.vocab_size
    device = 'cpu'
    out = ng._compute_logprob_orders([], V, device)
    assert out.shape == (V, ng.max_order)
    # 所有阶都应退化为 unigram（无上下文命中）
    expected_uni = torch.log(ng.uni_prob + 1e-10)
    # unigram 列
    assert torch.allclose(out[:, 0], expected_uni, atol=1e-7)
    # 高阶列经归一化后也应接近 unigram（无命中修正）
    for k in range(1, ng.max_order):
        col = out[:, k].exp()  # 转概率
        # 归一化后应与 unigram 一致（无命中时 base = uni）
        assert torch.allclose(col, ng.uni_prob, atol=1e-5), \
            f"空 ctx 阶 {k} 未退化为 unigram"


def test_compute_logprob_orders_no_hit_ctx():
    """R36-2-10: ctx 未命中时该阶退化为 unigram。"""
    v, ng = _make_ngram(max_order=5)
    V = ng.vocab_size
    device = 'cpu'
    # 构造一个不存在的上下文（用 vocab 外的 token id）
    ids = [999, 998]  # 假设不在语料中
    out = ng._compute_logprob_orders(ids, V, device)
    assert out.shape == (V, ng.max_order)
    # 所有高阶都应退化为 unigram（无命中）
    for k in range(1, ng.max_order):
        col = out[:, k].exp()
        assert torch.allclose(col, ng.uni_prob, atol=1e-5), \
            f"未命中 ctx 阶 {k} 未退化为 unigram"


def test_compute_logprob_orders_shape():
    """R36-2-11: 输出形状 (V, max_order)。"""
    v, ng = _make_ngram(max_order=8)
    V = ng.vocab_size
    device = 'cpu'
    ids = v.encode('中 国 人 民', add_special_tokens=False)
    out = ng._compute_logprob_orders(ids, V, device)
    assert out.shape == (V, 8), f"形状错误: {out.shape}"


def test_compute_logprob_orders_finite():
    """R36-2-12: 输出全部有限（无 NaN/Inf）。"""
    v, ng = _make_ngram(max_order=10)
    V = ng.vocab_size
    device = 'cpu'
    ids = v.encode('中 国 人 民 共 和 国', add_special_tokens=False)
    out = ng._compute_logprob_orders(ids, V, device)
    assert torch.isfinite(out).all(), "输出含 NaN/Inf"


# ===== _vec_for_ctx 路径缓存受益测试 =====

def test_vec_for_ctx_benefits_from_cache():
    """R36-2-13: _vec_for_ctx 调用后 _interp_w_cache 被填充。"""
    v, ng = _make_ngram(max_order=5)
    device = 'cpu'
    ids = v.encode('中 国 人', add_special_tokens=False)
    w2, w1 = ids[-2], ids[-1]
    # 调用 logprob_vector（走 _vec_for_ctx 路径）
    ng.logprob_vector([w2, w1], device)
    # 验证缓存被填充
    assert hasattr(ng, '_interp_w_cache'), "_interp_w_cache 未创建"
    # _vec_for_ctx 用 max_hit 阶，至少应缓存了某阶
    assert len(ng._interp_w_cache) > 0, "缓存未填充"


def test_vec_for_ctx_equivalence_after_cache():
    """R36-2-14: 缓存启用后 _vec_for_ctx 输出与首次一致。"""
    v, ng = _make_ngram(max_order=5)
    device = 'cpu'
    ids = v.encode('中 国 人', add_special_tokens=False)
    w2, w1 = ids[-2], ids[-1]
    # 首次调用（触发缓存）
    first = ng.logprob_vector([w2, w1], device).clone()
    # 第二次调用（用缓存）
    second = ng.logprob_vector([w2, w1], device)
    assert torch.allclose(first, second, atol=1e-12), \
        f"缓存前后不一致: max_diff={(first - second).abs().max().item():.2e}"


# ===== 集成测试：logprob_orders_matrix / incremental =====

def test_logprob_orders_matrix_integration():
    """R36-2-15: logprob_orders_matrix 集成测试（去重 + index_select 路径不回归）。"""
    v, ng = _make_ngram(max_order=5, smoothing=1.0, min_count=1)
    V = ng.vocab_size
    device = 'cpu'
    ids = torch.tensor([v.encode('中 国 人 民 共 和 国', add_special_tokens=False)], dtype=torch.long)
    out = ng.logprob_orders_matrix(ids, device)
    assert out.shape == (1, ids.size(1), V, ng.max_order), f"形状错误: {out.shape}"
    assert torch.isfinite(out).all(), "输出含 NaN/Inf"


def test_logprob_orders_matrix_matches_per_position():
    """R36-2-16: logprob_orders_matrix 与逐位置 _compute_logprob_orders 等价。"""
    v, ng = _make_ngram(max_order=5, smoothing=1.0, min_count=1)
    V = ng.vocab_size
    device = 'cpu'
    ids = torch.tensor([v.encode('中 国 人 民', add_special_tokens=False)], dtype=torch.long)
    B, T = ids.shape
    K = ng.max_order
    ctx_len = max(1, K - 1)
    pad = 0
    out = ng.logprob_orders_matrix(ids, device)  # (1, T, V, K)
    # 参考：逐位置调 _compute_logprob_orders
    ref = torch.empty(1, T, V, K)
    seq = ids[0].tolist()
    padded = [pad] * ctx_len + seq
    for t in range(T):
        ctx = padded[t: t + ctx_len]
        ref[0, t] = ng._compute_logprob_orders(ctx, V, device)
    assert torch.allclose(out, ref, atol=1e-6), \
        f"matrix 与逐位置不等价: max_diff={(out - ref).abs().max().item():.2e}"


def test_logprob_orders_incremental_integration():
    """R36-2-17: logprob_orders_incremental 集成测试（增量解码不回归）。"""
    v, ng = _make_ngram(max_order=5, smoothing=1.0, min_count=1)
    V = ng.vocab_size
    device = 'cpu'
    K = ng.max_order
    ctx_len = max(1, K - 1)
    ids = torch.tensor([v.encode('中 国 人 民', add_special_tokens=False)], dtype=torch.long)
    # 全量
    full = ng.logprob_orders_matrix(ids, device)
    # 增量：先取前 ctx_len 个作上下文，再逐段增量
    ctx2 = ids[:, :ctx_len]
    new_ids = ids[:, ctx_len:]
    inc = ng.logprob_orders_incremental(ctx2, new_ids, device)
    # 拼接全量结果对比
    full_seg = full[:, ctx_len:]
    assert inc.shape == full_seg.shape, f"形状不一致: {inc.shape} vs {full_seg.shape}"
    assert torch.allclose(inc, full_seg, atol=1e-6), \
        f"增量与全量不等价: max_diff={(inc - full_seg).abs().max().item():.2e}"


def test_logprob_orders_cache_invalidation():
    """R36-2-18: _orders_cache 缓存清理后重新计算结果一致。"""
    v, ng = _make_ngram(max_order=5)
    V = ng.vocab_size
    device = 'cpu'
    ids = v.encode('中 国 人', add_special_tokens=False)
    # 首次计算
    first = ng._compute_logprob_orders(ids, V, device).clone()
    # 清理 orders_cache（不影响 _interp_w_cache）
    if hasattr(ng, '_orders_cache_store'):
        ng._orders_cache_store.clear()
    # 再次计算
    second = ng._compute_logprob_orders(ids, V, device)
    assert torch.allclose(first, second, atol=1e-12), \
        f"缓存清理后结果不一致: max_diff={(first - second).abs().max().item():.2e}"


def test_interp_weights_cache_different_l1l2l3():
    """R36-2-19: 不同 l1/l2/l3 参数下缓存值正确。"""
    for l1, l2, l3 in [(0.1, 0.3, 0.6), (0.2, 0.3, 0.5), (0.5, 0.3, 0.2)]:
        _, ng = _make_ngram(l1=l1, l2=l2, l3=l3)
        for order in [2, 3]:
            ws = ng._interp_weights(order)
            ref = _reference_interp_weights(ng, order)
            for a, b in zip(ws, ref):
                assert abs(a - b) < 1e-12, f"l1/l2/l3={l1}/{l2}/{l3} order={order} 不一致"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
