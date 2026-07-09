#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
词汇编码诊断脚本
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.data_utils import load_data
from models.config_loader import load_config


def test_vocab_encoding():
    """测试词汇编码"""
    
    print("[1] 加载词汇...")
    vocab_size = load_config()['model']['vocab_size']
    dataset, vocab = load_data('data/train_data_final.txt', vocab_size=vocab_size)
    print(f"  词汇表大小: {len(vocab)}")
    print(f"  特殊词汇: BOS={vocab.bos_idx}, EOS={vocab.eos_idx}, PAD={vocab.pad_idx}, UNK={vocab.unk_idx}\n")
    
    # 测试一些词
    test_words = ["你好", "天气", "人工", "智能", "今天", "学习", "编程", "是什么", "朋友"]
    
    print("[2] 测试单个词汇编码:")
    for word in test_words:
        # 直接查找
        if word in vocab.word2idx:
            idx = vocab.word2idx[word]
            print(f"  '{word}' -> ID {idx} (在词汇表中)")
        else:
            print(f"  '{word}' -> UNK (不在词汇表中)")
    
    # 测试编码
    print("\n[3] 测试句子编码:")
    test_sentences = [
        "你好",
        "今天天气如何",
        "什么是人工智能",
        "我是一个AI",
        "怎样学习编程"
    ]
    
    for sentence in test_sentences:
        tokens = vocab.encode(sentence)
        decoded_words = [vocab.idx2word.get(t, '[UNK]') for t in tokens]
        print(f"\n  输入: {sentence}")
        print(f"  Tokens: {tokens}")
        print(f"  解码: {' '.join(decoded_words)}")
    
    # 检查词汇表覆盖率
    print(f"\n[4] 词汇表统计:")
    print(f"  总词汇量: {len(vocab)}")
    print(f"  特殊词汇: {5}")
    print(f"  常规词汇: {len(vocab) - 5}")


if __name__ == '__main__':
    test_vocab_encoding()
