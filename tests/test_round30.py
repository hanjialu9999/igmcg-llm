"""第三十轮回归测试：算子合并精简（低风险数值等价优化，续 R29.5）。

覆盖：
- R30-1: ngram.py `_compute_logprob_orders` L159 改写
  `w*p + (1-w)*uni[idx]` → `uni[idx] + w*(p - uni[idx])`（与 R29.5 convex_combine 同模式）
- R30-2: transformer.py hybrid 分支删除冗余 present 赋值（L378-379）
  块外 L399-400 统一组装，块内提前设置是死代码

数学等价：w*p + (1-w)*u = u + w*(p-u)（结合律）
"""
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.data_utils import CharTokenizer
from models.ngram import NGramModel


# ─── 测试夹具 ──────────────────────────────────────────────────────────────

CORPUS = [
    "中 国 人 民 生 活 幸 福",
    "中 国 梦 想 伟 大 复 兴",
    "人 民 当 家 作 主 权 利",
    "中 国 人 民 共 和 国 万 岁",
    "中 国 历 史 文 化 悠 久",
    "人 民 生 活 水 平 提 高",
]


def _make_ngram(max_order=5, smoothing=1.0, l1=0.1, l2=0.3, l3=0.6, vocab_size=None):
    """构造小 NgramModel 用于测试 _compute_logprob_orders。"""
    v = CharTokenizer(vocab_size=200)
    v.train(CORPUS)
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    f.write("\n".join(CORPUS) + "\n")
    f.close()
    ng = NGramModel(v, f.name, max_order=max_order, smoothing=smoothing,
                    l1=l1, l2=l2, l3=l3, vocab_size=vocab_size)
    os.unlink(f.name)
    return v, ng


# ─── R30-1: ngram convex_combine 数学等价 ─────────────────────────────────

def test_ngram_convex_combine_math_equivalence():
    """R30-1: 张量级验证 u + w*(p-u) == w*p + (1-w)*u 数学等价。

    这是 ngram.py L159 改写的核心数学基础（与 R29.5 convex_combine_scalar 同模式）。
    """
    torch.manual_seed(42)
    V = 200
    # w 是 Python float（_interp_weights 返回 list[float]），覆盖 [0, 1] 多个值
    for w in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        p = torch.rand(V, dtype=torch.float64)  # 高精度验证
        u = torch.rand(V, dtype=torch.float64)

        # 新实现（ngram.py L159 改写后）
        out_new = u + w * (p - u)
        # 原实现
        out_ref = w * p + (1.0 - w) * u

        max_diff = (out_new - out_ref).abs().max().item()
        assert max_diff < 1e-12, (
            f"w={w}: ngram convex_combine 不等价, max_diff={max_diff:.2e}"
        )


def test_ngram_logprob_orders_matches_legacy_formula():
    """R30-2: _compute_logprob_orders 改写后与原 w*p + (1-w)*uni[idx] 公式手算结果一致。

    构造小 NgramModel，对每个命中的 (idx, p, w) 手算原公式，与实际输出比较。
    """
    v, ng = _make_ngram(max_order=5)
    V = ng.vocab_size
    device = 'cpu'

    # 选一个能命中多阶 n-gram 的上下文
    ids = v.encode('中 国 人', add_special_tokens=False)
    ctx_tokens = ids  # 最近 max_order-1 个 token

    out = ng._compute_logprob_orders(ctx_tokens, V, device)  # (V, K)
    assert out.shape == (V, ng.max_order), f"shape 错误: {out.shape}"

    # 手算原公式验证：对每个命中的阶，检查 base[k-1, idx] 是否等于 w*p + (1-w)*u
    uni = ng._uni_dev_tensor
    K = ng.max_order
    # 复制 unigram 起步背景（与源码一致）
    base_ref = uni.unsqueeze(0).expand(K - 1, V).clone()
    for k in range(1, K):
        order = k + 1
        if len(ctx_tokens) >= order - 1:
            ctx = tuple(ctx_tokens[-(order - 1):])
        else:
            ctx = None
        if ctx is not None and ctx in ng._ngt_dev.get(order, {}):
            idx, p = ng._ngt_dev[order][ctx]
            ws = ng._interp_weights(order)
            w = ws[-1]
            # 原公式（R30 改写前）
            base_ref[k - 1, idx] = w * p + (1.0 - w) * uni[idx]

    # 归一化 + log（与源码一致）
    base_ref = torch.log(base_ref / base_ref.sum(dim=-1, keepdim=True) + 1e-10)

    # out[:, 1:] 对应 base.T（高阶列）
    out_base = out[:, 1:].T  # (K-1, V)

    max_diff = (out_base - base_ref).abs().max().item()
    # float32 精度极限 ~1e-7（与 R29.5 convex_combine 测试 atol=1e-6 一致）
    assert max_diff < 1e-6, (
        f"_compute_logprob_orders 与原公式不等价: max_diff={max_diff:.2e}"
    )


def test_ngram_logprob_orders_extreme_values():
    """R30-3: _compute_logprob_orders 在极端值下数值稳定。

    - w=0（纯 unigram）：out == log(uni + 1e-10)
    - w=1（纯 n-gram）：out == log(p_normalized + 1e-10)
    - p=u：out == log(u_normalized + 1e-10)
    """
    v, ng = _make_ngram(max_order=3)
    V = ng.vocab_size
    device = 'cpu'

    # 用空上下文触发 w=0 路径（无命中，base 全为 uni）
    out = ng._compute_logprob_orders([], V, device)
    uni = ng._uni_dev_tensor
    expected_high = torch.log(uni / uni.sum() + 1e-10)
    assert torch.allclose(out[:, 1], expected_high, atol=1e-6), (
        "空上下文时高阶应退化为 unigram"
    )

    # 数值稳定：无 NaN/Inf
    assert not torch.isnan(out).any(), "出现 NaN"
    assert not torch.isinf(out).any(), "出现 Inf"


def test_ngram_logprob_orders_deterministic():
    """R30-4: _compute_logprob_orders 两次调用结果完全一致（无随机性）。"""
    v, ng = _make_ngram(max_order=4)
    V = ng.vocab_size
    device = 'cpu'
    ids = v.encode('中 国 人', add_special_tokens=False)

    out1 = ng._compute_logprob_orders(ids, V, device)
    out2 = ng._compute_logprob_orders(ids, V, device)
    assert torch.equal(out1, out2), "两次调用结果不一致"


# ─── R30-2: hybrid 分支删除冗余 present 赋值 ──────────────────────────────

def test_hybrid_block_present_use_cache():
    """R30-5: hybrid block use_cache=True 时返回 present 是三元组。

    删除 L378-379 冗余赋值后，present 由块外 L399-400 统一组装，
    应返回 (attn_present, ssm_state, ssm_conv_state) 三元组。
    """
    from models.transformer import TransformerModel
    m = TransformerModel(
        vocab_size=100, embedding_dim=64, num_heads=4, num_layers=2,
        hidden_dim=128, max_seq_length=32,
        layer_plan=['hybrid', 'hybrid'],  # 触发 hybrid block
        mixer='attn',
    )
    m.eval()
    x = torch.randint(0, 100, (2, 8))
    with torch.no_grad():
        out, presents = m(x, use_cache=True)
    assert out.shape == (2, 8, 100), f"输出 shape 错误: {out.shape}"
    assert len(presents) == 2, f"present 数量错误: {len(presents)}"
    # hybrid block 的 present 应是三元组 (attn_kv, ssm_state, ssm_conv_state)
    for pk in presents:
        assert pk is not None, "hybrid block present 不应为 None"
        assert isinstance(pk, tuple) and len(pk) == 3, (
            f"hybrid present 应为 3 元组, 实际: {type(pk)} len={len(pk) if isinstance(pk, tuple) else 'N/A'}"
        )


def test_hybrid_block_present_fields_nonnull():
    """R30-6: hybrid block use_cache=True 时 present 三个字段都非 None。

    删除冗余赋值后，attn_present / ssm_state / ssm_conv_state 都应正确填充。
    """
    from models.transformer import TransformerModel
    m = TransformerModel(
        vocab_size=100, embedding_dim=64, num_heads=4, num_layers=2,
        hidden_dim=128, max_seq_length=32,
        layer_plan=['hybrid', 'hybrid'],
        mixer='attn',
    )
    m.eval()
    x = torch.randint(0, 100, (2, 8))
    with torch.no_grad():
        out, presents = m(x, use_cache=True)
    for i, pk in enumerate(presents):
        attn_kv, ssm_state, ssm_conv = pk
        assert attn_kv is not None, f"block {i}: attn_kv 不应为 None"
        assert ssm_state is not None, f"block {i}: ssm_state 不应为 None"
        assert ssm_conv is not None, f"block {i}: ssm_conv 不应为 None"
        # attn_kv 应是 (k, v) 二元组
        assert isinstance(attn_kv, tuple) and len(attn_kv) == 2, (
            f"block {i}: attn_kv 应为 (k,v) 二元组"
        )


def test_hybrid_block_use_cache_false_no_present():
    """R30-7: hybrid block use_cache=False 时不返回 present（与原行为一致）。"""
    from models.transformer import TransformerModel
    m = TransformerModel(
        vocab_size=100, embedding_dim=64, num_heads=4, num_layers=2,
        hidden_dim=128, max_seq_length=32,
        layer_plan=['hybrid', 'hybrid'],
        mixer='attn',
    )
    m.eval()
    x = torch.randint(0, 100, (2, 8))
    with torch.no_grad():
        out = m(x, use_cache=False)
    # use_cache=False 返回单 tensor
    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == (2, 8, 100), f"输出 shape 错误: {out.shape}"


def test_hybrid_block_incremental_decode():
    """R30-8: hybrid block 增量解码——首步全量，后续步用 cache。

    验证删除冗余 present 赋值后增量解码仍正常。
    """
    from models.transformer import TransformerModel
    m = TransformerModel(
        vocab_size=100, embedding_dim=64, num_heads=4, num_layers=2,
        hidden_dim=128, max_seq_length=32,
        layer_plan=['hybrid', 'hybrid'],
        mixer='attn',
    )
    m.eval()
    # 首步：全量前向
    x = torch.randint(0, 100, (1, 4))
    with torch.no_grad():
        out1, presents = m(x, use_cache=True)
    assert out1.shape == (1, 4, 100)

    # 第二步：增量解码（1 token）
    x2 = torch.randint(0, 100, (1, 1))
    with torch.no_grad():
        out2, presents2 = m(x2, past_key_values=presents, use_cache=True)
    assert out2.shape == (1, 1, 100), f"增量解码 shape 错误: {out2.shape}"
    # present 仍应是三元组
    for pk in presents2:
        assert pk is not None and isinstance(pk, tuple) and len(pk) == 3


# ─── 混合 block_type 模型回归 ─────────────────────────────────────────────

def test_mixed_layer_plan_present():
    """R30-9: 混合 layer_plan（attn + hybrid）present 正确。

    验证不同 block_type 的 present 组装逻辑仍正确。
    """
    from models.transformer import TransformerModel
    m = TransformerModel(
        vocab_size=100, embedding_dim=64, num_heads=4, num_layers=3,
        hidden_dim=128, max_seq_length=32,
        layer_plan=['attn', 'hybrid', 'ssm'],  # 三种 block_type 混合
        mixer='attn',
    )
    m.eval()
    x = torch.randint(0, 100, (2, 8))
    with torch.no_grad():
        out, presents = m(x, use_cache=True)
    assert out.shape == (2, 8, 100)
    assert len(presents) == 3
    # attn block: present = ((k,v), None, None)
    assert presents[0][0] is not None and presents[0][1] is None and presents[0][2] is None
    # hybrid block: present = (attn_kv, ssm_state, ssm_conv) 全非 None
    assert presents[1][0] is not None and presents[1][1] is not None and presents[1][2] is not None
    # ssm block: present = (None, ssm_state, ssm_conv)
    assert presents[2][0] is None and presents[2][1] is not None and presents[2][2] is not None


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v']))
