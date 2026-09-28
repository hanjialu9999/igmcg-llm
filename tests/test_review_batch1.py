# -*- coding: utf-8 -*-
"""2026-09-28 架构审查·零风险批回归（M2 / M3 / M9 / M10）。

对应 AGENT_MEMORY.md §11.2 / §11.3 / §11.4 的"已实施"项。本批不改任何数值路径，
因此断言集中在两件事：
  1. 状态正确——跨序列状态真的复位、剪枝只标记不删参；
  2. 告警真的会响——静默死配置/静默降级能被测试台看见，且合法配置保持安静。

M9  _ngram_last_ids 缺 is_fresh 复位 → 新序列首步吃到上一序列 token 尾巴
M10 ngram_fusion 配置开了但统计表缺失 → 静默降级为纯神经（igmcg 连带失效）
M2  prune_layers 返回值/docstring 与实现（只标记跳过、不删参数）不一致
M3  记忆/ALiBi 相关的静默死配置（只 warn 不 raise）
"""
import os
import sys
import tempfile
import warnings
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.data_utils import CharTokenizer
from models.model_config import AttnConfig, MemoryConfig, ModelConfig
from models.ngram import NGramModel
from models.transformer import TransformerModel


_CORPUS = [
    "中 国 人 民 生 活 幸 福",
    "中 国 梦 想 伟 大 复 兴",
    "人 民 当 家 作 主 权 利",
    "中 国 人 民 共 和 国 万 岁",
]


def _make_ngram(max_order=5):
    """构建 (tokenizer, NGramModel)；语料写临时文件后即可删（统计已进内存）。

    不传 vocab_size → NGramModel 取 len(vocab)（语料实际覆盖 262 个 token），
    与模型词表对齐；硬编码 200 会让 uni 张量 (200,) 撞上 id 233 越界。
    """
    v = CharTokenizer(vocab_size=200)
    v.train(_CORPUS)
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    f.write("\n".join(_CORPUS * 5) + "\n")
    f.close()
    ng = NGramModel(v, f.name, max_order=max_order, smoothing=1.0, min_count=1)
    os.unlink(f.name)
    return v, ng


def _build_ngram_model(v, ng):
    """带 n-gram 融合的小模型（M9 需要融合路径才会维护 _ngram_last_ids）。"""
    m = TransformerModel(vocab_size=len(v), embedding_dim=64, num_heads=4, num_layers=2,
                         hidden_dim=128, max_seq_length=32,
                         ngram_fusion=True, ngram_model=ng,
                         gradient_checkpointing=False)
    m.eval()
    return m


def _get_logits(out):
    if isinstance(out, dict):
        return out["logits"]
    if isinstance(out, tuple):
        return out[0]
    return out


def _decode(model, seq):
    """按 generate() 的方式做增量解码，返回 (每步 logits, 最终 past)。"""
    outs = []
    with torch.no_grad():
        out, past = model(seq[:, :1], past_key_values=None, use_cache=True)
        outs.append(_get_logits(out).clone())
        for t in range(1, seq.shape[1]):
            out, past = model(seq[:, t:t + 1], past_key_values=past, use_cache=True)
            outs.append(_get_logits(out).clone())
    return outs, past


# ============================================================
# M9：_ngram_last_ids 的 is_fresh 复位
# ============================================================

def test_m9_ngram_buffer_reset_on_new_sequence():
    """新序列首步（is_fresh）必须把增量 n-gram 缓冲清空，第二步以 pad 上下文重建。

    回归点：不复位时，序列 B 第二步会拿序列 A 的 token 尾巴去查 n-gram 计数表，
    跨序列串扰直接进 logits（generate() 首步走非增量分支，掩盖了这个缺陷）。
    """
    v, ng = _make_ngram()
    m = _build_ngram_model(v, ng)
    A = torch.tensor([[10, 11, 12, 13]])
    B = torch.tensor([[20, 21]])

    with torch.no_grad():
        # 序列 A：完整增量解码 → 缓冲被 A 的 token 填满
        _, past_a = m(A[:, :1], past_key_values=None, use_cache=True)
        for t in range(1, A.shape[1]):
            _, past_a = m(A[:, t:t + 1], past_key_values=past_a, use_cache=True)
        buf_a = m._ngram_last_ids
        assert buf_a is not None, "序列 A 解码后缓冲应已填充"
        assert torch.equal(buf_a, A), f"序列 A 后缓冲应恰为 A，实际 {buf_a.tolist()}"

        # 序列 B 首步：is_fresh 必须先复位，缓冲只能由 pad + B[0] 组成。
        # 不复位时这里会变成 [11,12,13,20]（A 的尾巴 + B 首 token）。
        _, past_b = m(B[:, :1], past_key_values=None, use_cache=True)
        buf_b1 = m._ngram_last_ids
        pad = getattr(ng.vocab, 'pad_idx', 0)
        ctx_len = max(1, getattr(ng, 'max_order', 10) - 1)
        want_b1 = torch.full((1, ctx_len), pad, dtype=buf_b1.dtype)
        want_b1[0, -1] = B[0, 0]
        assert torch.equal(buf_b1, want_b1), \
            (f"M9 回归：新序列首步未复位 _ngram_last_ids（串到上一序列），"
             f"实际 {buf_b1.tolist()} 期望 {want_b1.tolist()}")

        # 序列 B 第二步：在 pad 基础上继续累积，绝不能混入 A 的 token
        m(B[:, 1:2], past_key_values=past_b, use_cache=True)
        buf_b2 = m._ngram_last_ids
        want_b2 = torch.full((1, ctx_len), pad, dtype=buf_b2.dtype)
        want_b2[0, -2:] = B
        assert torch.equal(buf_b2, want_b2), \
            f"M9 回归：缓冲混入上一序列，实际 {buf_b2.tolist()} 期望 {want_b2.tolist()}"


def test_m9_no_cross_sequence_logit_leak():
    """行为面锁：先解码 A 再解码 B，与只解码 B 的同权重模型逐位一致。"""
    v, ng = _make_ngram()
    # 两次相同 seed 的构造 → 同权重（NGramModel/Tokenizer 不消耗 torch RNG）
    torch.manual_seed(1234)
    m_dirty = _build_ngram_model(v, ng)
    torch.manual_seed(1234)
    m_clean = _build_ngram_model(v, ng)
    for (n1, p1), (n2, p2) in zip(m_dirty.named_parameters(),
                                  m_clean.named_parameters()):
        assert torch.equal(p1, p2), f"两次构造权重不一致：{n1}"

    A = torch.tensor([[10, 11, 12, 13, 14, 15]])
    B = torch.tensor([[20, 21, 22, 23]])

    dirty_outs, _ = _decode(m_dirty, A)
    dirty_b, _ = _decode(m_dirty, B)
    clean_b, _ = _decode(m_clean, B)

    assert len(dirty_b) == len(clean_b)
    for i, (d, c) in enumerate(zip(dirty_b, clean_b)):
        diff = (d - c).abs().max().item()
        assert diff < 1e-6, f"M9 跨序列泄漏：B 第 {i} 步 logits diff={diff:.3e}"


# ============================================================
# M10：ngram_fusion 配置与运行时不一致时必须告警
# ============================================================

def test_m10_fusion_requested_without_model_warns():
    """ngram_fusion=True 但没给统计表 → 实际生效状态是 False，必须可见。"""
    with pytest.warns(RuntimeWarning, match="ngram_model"):
        m = TransformerModel(vocab_size=200, embedding_dim=64, num_heads=4, num_layers=1,
                             hidden_dim=128, max_seq_length=32,
                             ngram_fusion=True, ngram_model=None,
                             gradient_checkpointing=False)
    assert not m.ngram_fusion_enabled
    assert not getattr(m, 'igmcg_enabled', False)


def test_m10_fusion_off_stays_silent():
    """默认关（旧配置）不应产生任何告警——保证向后兼容不刷屏。"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = TransformerModel(vocab_size=200, embedding_dim=64, num_heads=4, num_layers=1,
                             hidden_dim=128, max_seq_length=32,
                             gradient_checkpointing=False)
    assert not m.ngram_fusion_enabled


def test_m10_build_ngram_model_failure_warns(tmp_path):
    """语料缺失时 build_ngram_model 必须 RuntimeWarning，而非一行 print。"""
    from models.checkpoint import build_ngram_model
    v = CharTokenizer(vocab_size=200)
    v.train(_CORPUS)
    cfg = {'ngram_fusion': True,
           'ngram_corpus': str(tmp_path / 'definitely_missing.txt'),
           'vocab_size': 200}
    with pytest.warns(RuntimeWarning, match="降级为纯神经"):
        assert build_ngram_model(v, cfg) is None


def test_m10_fusion_disabled_build_is_silent():
    """配置关着 → build_ngram_model 直接返回 None，不得告警。"""
    from models.checkpoint import build_ngram_model
    v = CharTokenizer(vocab_size=200)
    v.train(_CORPUS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert build_ngram_model(v, {'ngram_fusion': False}) is None


# ============================================================
# M2：prune_layers 只标记不删参；返回值与 _pruned_layers 一致
# ============================================================

def _build_layer_skip():
    m = TransformerModel(vocab_size=200, embedding_dim=64, num_heads=4, num_layers=3,
                         hidden_dim=128, max_seq_length=32, layer_skip=True,
                         gradient_checkpointing=False)
    m.eval()
    # skip_gate init=1.0 → sigmoid≈0.73，全在 0.5 阈值之上；显式设值让剪枝结果确定
    with torch.no_grad():
        for i, blk in enumerate(m.blocks):
            if hasattr(blk, 'skip_gate'):
                blk.skip_gate.fill_(10.0 if i == 0 else -10.0)
    return m


def test_m2_prune_marks_without_deleting_params():
    """只标记跳过，参数/权重一条都不能少（docstring 契约）。"""
    m = _build_layer_skip()
    n_params = sum(p.numel() for p in m.parameters())
    sd_keys = set(m.state_dict().keys())

    pruned = m.prune_layers(0.5)
    assert pruned == [0]
    assert m._pruned_layers == {0}

    assert sum(p.numel() for p in m.parameters()) == n_params, "剪枝不得删除参数"
    assert set(m.state_dict().keys()) == sd_keys, "剪枝不得改 state_dict 结构"


def test_m2_pruned_layer_is_skipped_in_eval_forward():
    """被标记层在 eval 前向被短路，输出仍形状正确且前向可跑。"""
    m = _build_layer_skip()
    m.prune_layers(0.5)
    ids = torch.randint(0, 200, (2, 8))
    with torch.no_grad():
        out = _get_logits(m(ids, use_cache=False))
    assert out.shape == (2, 8, 200)
    assert torch.isfinite(out).all()


def test_m2_threshold_zero_cancels_and_reports_empty():
    """threshold<=0 = 取消剪枝：_pruned_layers 与返回值必须同时为空（原实现矛盾）。"""
    m = _build_layer_skip()
    assert m.prune_layers(0.0) == [], "threshold<=0 应返回空列表（取消剪枝）"
    assert m._pruned_layers == set()
    # 再次开启应恢复正常
    assert m.prune_layers(0.5) == [0]
    assert m._pruned_layers == {0}


def test_m2_docstring_documents_no_param_deletion():
    """文档契约锁：docstring 必须明说不删参数，防止后人按"移除"语义去实现。"""
    doc = TransformerModel.prune_layers.__doc__ or ""
    assert "不删除任何参数" in doc
    assert "_pruned_layers" in doc


# ============================================================
# M3：静默死配置必须告警（只 warn 不 raise）
# ============================================================

def test_m3_memory_options_with_size_zero_warn():
    """memory_size=0 时记忆模块根本不构建，下列开关全是死配置。"""
    with pytest.warns(RuntimeWarning, match="死配置"):
        MemoryConfig(size=0, retrieval=True)
    with pytest.warns(RuntimeWarning, match="死配置"):
        MemoryConfig(size=0, forget=True)
    with pytest.warns(RuntimeWarning, match="死配置"):
        MemoryConfig(size=0, product_key=True)
    with pytest.warns(RuntimeWarning, match="死配置"):
        MemoryConfig(size=0, sparse_topk=8)
    with pytest.warns(RuntimeWarning, match="死配置"):
        MemoryConfig(size=0, retrieval_full=True)


def test_m3_sparse_topk_boundary_warns():
    """sparse_topk >= size 时不满足 0 < topk < mem_cols → 稀疏召回静默退化为稠密。"""
    with pytest.warns(RuntimeWarning, match="稀疏召回未启用"):
        MemoryConfig(size=4, sparse_topk=4)
    with pytest.warns(RuntimeWarning, match="稀疏召回未启用"):
        MemoryConfig(size=4, sparse_topk=99)


def test_m3_valid_memory_config_stays_silent():
    """合法/默认配置必须安静——否则告警会淹没真正的问题。"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MemoryConfig()                                   # 全默认
        MemoryConfig(size=16)                            # 开记忆，其余关
        MemoryConfig(size=16, sparse_topk=4)             # topk 合法边界内
        MemoryConfig(size=16, retrieval=True)            # 开了记忆才能开检索


def test_m3_single_layer_memory_warns():
    """单层 + 记忆：读-写流水线无下一层消费 → 记忆参数不参与梯度（死参数）。"""
    with pytest.warns(RuntimeWarning, match="死参数"):
        ModelConfig(vocab_size=50, embedding_dim=64, num_heads=4, num_layers=1,
                    hidden_dim=128, max_seq_length=32,
                    memory=MemoryConfig(size=16, comp_dim=16))


def test_m3_alibi_with_memory_mem_cols_warns():
    """alibi + mem_cols>0 → 已知缺陷 H3（ALiBi 距离未做 mem_cols 还原）。"""
    with pytest.warns(RuntimeWarning, match="H3"):
        ModelConfig(vocab_size=50, embedding_dim=64, num_heads=4, num_layers=2,
                    hidden_dim=128, max_seq_length=32,
                    attn=AttnConfig(alibi=True),
                    memory=MemoryConfig(size=16, comp_dim=16))


def test_m3_alibi_with_controller_mem_cols_warns():
    """Controller 压缩记忆同样占 mem_cols，与 MemoryBank 叠加计算。"""
    with pytest.warns(RuntimeWarning, match="mem_cols=4"):
        ModelConfig(vocab_size=50, embedding_dim=64, num_heads=4, num_layers=2,
                    hidden_dim=128, max_seq_length=32,
                    attn=AttnConfig(alibi=True), controller=True,
                    controller_mem_slots=4)


def test_m3_alibi_without_mem_cols_stays_silent():
    """alibi 但 mem_cols=0（无记忆、无 Controller 记忆）→ 不该告警。"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ModelConfig(vocab_size=50, embedding_dim=64, num_heads=4, num_layers=2,
                    hidden_dim=128, max_seq_length=32, attn=AttnConfig(alibi=True))


def test_m3_baseline_model_config_stays_silent():
    """默认/典型配置整体静默，锁住"只在真死配置时开口"。"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ModelConfig(vocab_size=50, embedding_dim=64, num_heads=4, num_layers=2,
                    hidden_dim=128, max_seq_length=32)
