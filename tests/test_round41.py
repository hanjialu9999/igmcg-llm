# -*- coding: utf-8 -*-
"""R41 回归测试：n-gram 构建内存/时间优化 + 词表对齐。

问题（自主体验发现）：
  1. NGramModel 全量构建：8k 语料 max_order=10 + min_count=1 → 数百万纯 Python
     dict/Counter 条目（每 entry ~1KB）→ 数 GB 内存吃满 + 构建卡死无输出；
  2. 解码期 n-gram 只需局部统计先验，全语料统计纯浪费；
  3. generate.py 未传 vocab_size → 模型词表 12000 vs ngram 9186 维度不匹配崩溃。

修复：
  - ngram.py: 新增 max_lines（只统计前 N 行）、构建进度打印（每 5000 行 + 每阶剪枝统计）；
  - generate.py: NGramModel 调用 max_order=3 + max_lines=2000 + min_count=2
    + vocab_size=model.vocab_size（对齐模型词表）。
"""

import json
import os
import tempfile

import torch

from models.ngram import NGramModel


def _make_vocab(tmp):
    vocab_path = os.path.join(tmp, 'vocab.json')
    chars = ['的', '一', '是', '不', '了', '人', '我', '在', '有', '他',
             '中', '国', '北', '京', '上', '海', '天', '气', '真', '好',
             '测', '试', '文', '本', '数', '据']
    idx2word = {i: c for i, c in enumerate(chars)}
    word2idx = {c: i for i, c in enumerate(chars)}
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump({'char': True, 'word2idx': word2idx, 'idx2word': idx2word,
                   'merges': [], 'special_tokens': {}}, f, ensure_ascii=False)
    from models.data_utils import CharTokenizer
    v = CharTokenizer()
    v.word2idx = word2idx
    v.idx2word = idx2word
    v.merges = []
    v.vocab_size = len(word2idx)
    return v


def _make_corpus(tmp, n_lines=100):
    corpus = os.path.join(tmp, 'corpus.txt')
    words = ['北京上海', '中国天气', '今天是好', '文本数据', '测试汉字', '真不错呀',
             '人工智能', '机器学习', '深度学习', '神经网络']
    with open(corpus, 'w', encoding='utf-8') as f:
        for i in range(n_lines):
            # 三词块组合：(i%10, i*7%10, i//10%10) 周期 100，前 25 行与全量覆盖不同组合
            a = words[i % 10]
            b = words[(i * 7) % 10]
            c = words[i // 10 % 10]
            f.write(f"{a}{b}{c}\n")
    return corpus


def test_r41_max_lines_limits_corpus():
    """max_lines 只统计前 N 行：截断构建 == 全量构建的前 N 行结果。"""
    with tempfile.TemporaryDirectory() as tmp:
        vocab = _make_vocab(tmp)
        corpus = _make_corpus(tmp, n_lines=50)
        full = NGramModel(vocab, corpus, max_order=3, min_count=1)
        cut = NGramModel(vocab, corpus, max_order=3, min_count=1, max_lines=25)
        assert len(full.ngrams[3]) > len(cut.ngrams[3]), "截断后 order3 表应更小"
        # 截断表是全量的前缀统计：cut 中每个 (ctx, token) 计数必须 ≤ full 对应计数
        for ctx, counter in cut.ngrams[3].items():
            for tok, n in counter.items():
                assert full.ngrams[3][ctx][tok] >= n, f"截断表 {ctx} 计数 {tok}:{n} 超过全量"


def test_r41_min_count_prunes():
    """min_count=2 剪掉单次 n-gram。"""
    with tempfile.TemporaryDirectory() as tmp:
        vocab = _make_vocab(tmp)
        corpus = _make_corpus(tmp, n_lines=10)
        keep = NGramModel(vocab, corpus, max_order=3, min_count=1)
        prune = NGramModel(vocab, corpus, max_order=3, min_count=2)
        assert len(prune.ngrams[3]) <= len(keep.ngrams[3])
        # 剪枝后所有保留的 next-token 计数 >= min_count
        for ctx, counter in prune.ngrams[3].items():
            assert all(n >= 2 for n in counter.values()), f"order3 {ctx} 存在 <2 计数"


def test_r41_vocab_size_alignment():
    """vocab_size 对齐模型词表：logprob 向量形状 == vocab_size。"""
    with tempfile.TemporaryDirectory() as tmp:
        vocab = _make_vocab(tmp)
        corpus = _make_corpus(tmp, n_lines=10)
        # 模型词表（12000）远大于语料覆盖（26 字符）
        n = NGramModel(vocab, corpus, max_order=3, min_count=2,
                       vocab_size=12000)
        ids = [vocab.pad_idx, vocab.pad_idx] + [vocab.word2idx['天'], vocab.word2idx['气']]
        lp = n.logprob_vector(ids, torch.device('cpu'))
        assert lp.shape[0] == 12000, f"logprob 向量 {lp.shape[0]} != 12000"
        # 未对齐时（默认 len(vocab)）形状不同
        n2 = NGramModel(vocab, corpus, max_order=3, min_count=2)
        lp2 = n2.logprob_vector(ids, torch.device('cpu'))
        assert lp2.shape[0] == len(vocab)


def test_r41_generate_call_site():
    """generate.py 的 NGramModel 调用必须带 max_order=3/max_lines/min_count/vocab_size。"""
    src = open('scripts/generate.py', encoding='utf-8').read()
    assert 'max_order=3' in src, "generate.py 必须用 max_order=3（高阶 n-gram 无统计意义）"
    assert 'max_lines=2000' in src, "generate.py 必须限制语料行数（防内存爆）"
    assert 'min_count=2' in src, "generate.py 必须剪掉单次 n-gram"
    assert 'vocab_size=' in src, "generate.py 必须对齐模型词表"


def test_r41_ngram_build_progress():
    """_build 包含进度输出（每 5000 行 + 每阶剪枝统计），避免无输出假死观感。"""
    src = open('models/ngram.py', encoding='utf-8').read()
    assert 'ngram build' in src, "构建过程必须打印进度"
    assert 'counting done' in src
