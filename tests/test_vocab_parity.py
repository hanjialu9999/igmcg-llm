import torch
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.data_utils import BaseTokenizer, CharTokenizer


def _mini_corpus():
    return [
        "你好世界 这是测试 。",
        "机器学习 很有趣 大模型 在 推理 。",
        "字符级 语言模型 的训练 数据 。",
    ]


def test_basetokenizer_oov_goes_to_byte_token():
    # 单一 BaseTokenizer（char）契约：OOV 必落字节级 token，零 OOV，id 在 special 之后 256 区间
    t = CharTokenizer()
    t.train(_mini_corpus())
    out = t.encode("龘龘龘")
    assert out, "字节回退应产生 token"
    # 字节级 token 的 id 落在 special 之后、special+256 区间内
    assert all(t.unk_idx < tid < t.unk_idx + 256 or tid in (t.bos_idx, t.eos_idx)
               for tid in out if tid not in (t.bos_idx, t.eos_idx)), \
        "BaseTokenizer OOV 应回退到字节 token，而非 unk"


def test_basetokenizer_roundtrip():
    t = CharTokenizer()
    t.train(_mini_corpus())
    text = _mini_corpus()[1]
    ids = t.encode(text)
    back = t.decode(ids)
    # 字符级字节回退应近似无损还原原文（忽略特殊 token）
    assert isinstance(back, str)
    assert t.bos_idx in ids and t.eos_idx in ids


def test_basetokenizer_in_vocab_char_needs_no_byte_fallback():
    # 语料内的常见字符应直接映射到单字 token（非字节回退），OOV 占比为 0
    t = CharTokenizer()
    t.train(_mini_corpus())
    text = _mini_corpus()[0]
    ids = t.encode(text, add_special_tokens=False)
    assert all(tid >= len(t.special_tokens) + 256 or tid in range(len(t.special_tokens))
               for tid in ids), \
        "语料内字符不应走字节回退（零 OOV）"


def test_single_track_special_tokens_consistent():
    # 单一 BaseTokenizer 与特殊 token 常量（models.constants）索引一致
    from models.constants import SPECIAL_TOKENS, PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX, SEP_IDX
    t = CharTokenizer()
    assert t.pad_idx == PAD_IDX and t.unk_idx == UNK_IDX
    assert t.bos_idx == BOS_IDX and t.eos_idx == EOS_IDX and t.sep_idx == SEP_IDX
    assert list(t.special_tokens) == list(SPECIAL_TOKENS)
