"""生成字符级词表（学习型分词的词表基座）。

用法:
    python scripts/build_char_vocab.py --data data/pretrain_corpus/merged.txt \
        --vocab-size 5000 --out checkpoints_dml/vocab_char.json

词表 = 高频单字 + 256 byte token，零 OOV；相邻字符的"合并成词"由模型侧
CharMergeLayer 学习（受 LM loss 监督），本脚本只做字符→索引映射。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.data_utils import CharTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description='生成字符级词表')
    ap.add_argument('--data', default='data/pretrain_corpus/merged.txt')
    ap.add_argument('--vocab-size', type=int, default=5000)
    ap.add_argument('--min-freq', type=int, default=1)
    ap.add_argument('--out', default='checkpoints_dml/vocab_char.json')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    texts = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                texts.append(line)
            if args.limit and len(texts) >= args.limit:
                break
    print(f'载入语料 {len(texts)} 行')

    tok = CharTokenizer(vocab_size=args.vocab_size)
    tok.train(texts, min_freq=args.min_freq)
    print(f'字符词表大小 {len(tok)}')

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    print(f'已导出 -> {out_path}')

    sample = texts[:200]
    total = sum(len(t) for t in sample)
    toks = sum(len(tok.tokenize(t)) for t in sample)
    oov = sum(1 for t in sample for s in tok.tokenize(t)
              if tok.word2idx.get(s, tok.unk_idx) == tok.unk_idx)
    print(f'抽样: 字符 {total} token {toks} 压缩比 {toks/max(total,1):.3f} '
          f'OOV {oov/max(toks,1):.4%}')


if __name__ == '__main__':
    main()
