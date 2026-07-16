"""根据训练语料生成 BPE 词表并导出 vocab.json。

用法:
    python scripts/build_bpe_vocab.py --data data/pretrain_corpus/merged.txt --vocab-size 8000 --out checkpoints_dml/vocab.json

词表完全由训练数据分布决定（BPE 字节对合并），中文以单字起步、
高频相邻字/词合并为子词，英文/数字也被合并，OOV 趋近 0。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.data_utils import BPETokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description='根据训练语料生成 BPE 词表')
    ap.add_argument('--data', default='data/pretrain_corpus/merged.txt', help='训练语料文本')
    ap.add_argument('--vocab-size', type=int, default=8000, help='目标词表大小')
    ap.add_argument('--min-freq', type=int, default=2, help='单字/子词最低出现次数')
    ap.add_argument('--out', default='checkpoints_dml/vocab.json', help='输出 vocab.json 路径')
    ap.add_argument('--limit', type=int, default=0, help='仅用前 N 行（0=全部），调试用')
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
    print(f'载入语料 {len(texts)} 行，来自 {data_path}')

    tok = BPETokenizer(vocab_size=args.vocab_size)
    tok.train(texts, min_freq=args.min_freq)
    print(f'训练完成：词表大小 {len(tok)}，合并规则 {len(tok.merges)} 条')

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    print(f'已导出词表 -> {out_path}')

    # 简单质量速览：抽样编码看压缩比与 OOV
    sample = texts[: min(200, len(texts))]
    total_chars = sum(len(t) for t in sample)
    total_tokens = sum(len(tok.tokenize(t)) for t in sample)
    oov = sum(1 for t in sample for s in tok.tokenize(t) if tok.word2idx.get(s) == tok.unk_idx)
    print(f'抽样 {len(sample)} 行：字符数 {total_chars}，token 数 {total_tokens}，'
          f'压缩比 {total_tokens / max(total_chars, 1):.3f} tok/char，'
          f'OOV token 占比 {oov / max(total_tokens, 1):.4%}')


if __name__ == '__main__':
    main()
