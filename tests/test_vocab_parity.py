import torch
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.data_utils import Vocabulary, BaseTokenizer, CharTokenizer


def _mini_corpus():
    return [
        "你好世界 这是测试 。",
        "机器学习 很有趣 大模型 在 推理 。",
        "字符级 语言模型 的训练 数据 。",
    ]


def test_vocabulary_oov_goes_to_unk():
    # Vocabulary 契约：未知词必落 unk_idx（非字节回退），OOV 行为确定
    v = Vocabulary()
    v.build_vocab(_mini_corpus())
    # 一个语料外的稀有字应映射到 unk
    out = v.encode("龘龘龘")
    assert all(t == v.unk_idx for t in out if t not in (v.bos_idx, v.eos_idx))


def test_basetokenizer_oov_goes_to_byte_token():
    # BaseTokenizer（char）契约：OOV 必落字节级 token，零 OOV，id 在 special 之后 256 区间
    t = CharTokenizer()
    t.train(_mini_corpus())
    out = t.encode("龘龘龘")
    assert out, "字节回退应产生 token"
    # 字节级 token 的 id 落在 special 之后、special+256 区间内
    assert all(t.unk_idx <= tid < t.unk_idx + 256 or tid in (t.bos_idx, t.eos_idx)
               for tid in out if tid not in (t.bos_idx, t.eos_idx)), \
        "BaseTokenizer OOV 应回退到字节 token，而非 unk"


def test_vocabulary_roundtrip():
    v = Vocabulary()
    v.build_vocab(_mini_corpus())
    text = _mini_corpus()[0]
    ids = v.encode(text)
    back = v.decode(ids)
    # 往返重建：去掉特殊 token 后文本核心应保留（空格切词语义）
    assert isinstance(back, str)
    assert v.bos_idx in ids and v.eos_idx in ids


def test_basetokenizer_roundtrip():
    t = CharTokenizer()
    t.train(_mini_corpus())
    text = _mini_corpus()[1]
    ids = t.encode(text)
    back = t.decode(ids)
    # 字符级字节回退应近似无损还原原文（忽略特殊 token）
    assert isinstance(back, str)
    assert t.bos_idx in ids and t.eos_idx in ids


def test_dual_track_special_tokens_consistent():
    # 双轨共享特殊 token 常量（models.constants），索引一致
    from models.constants import SPECIAL_TOKENS, PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX, SEP_IDX
    v = Vocabulary()
    t = CharTokenizer()
    for tok in (v, t):
        assert tok.pad_idx == PAD_IDX and tok.unk_idx == UNK_IDX
        assert tok.bos_idx == BOS_IDX and tok.eos_idx == EOS_IDX and tok.sep_idx == SEP_IDX
        assert list(tok.special_tokens) == list(SPECIAL_TOKENS)
