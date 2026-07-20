"""
数据集管理统一入口（单入口）：查看、合并、统计、构建词表、转 jsonl。

用法：
    python scripts/data_manager.py merge [--input_dir data/datasets] [--output_file data/train_data_combined.txt] [--dedup] [--build-vocab] ...
    python scripts/data_manager.py stats [--input_dir data/datasets]
    python scripts/data_manager.py vocab --data_file ... [--vocab_size 5000] [--vocab_output data/vocab.json]
    python scripts/data_manager.py sample [--input_dir data/datasets] [--theme xxx]
    python scripts/data_manager.py to-jsonl [--input_folder data/datasets] [--output_folder data/processed]

兼容旧脚本：scripts/merge_data.py 与 scripts/process_data.py 现均为本文件的薄包装。
"""

import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Set

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.data_utils import CharTokenizer


# --------------------------------------------------------------------------
# 公共工具
# --------------------------------------------------------------------------
def find_data_files(input_dir: str, exclude_patterns: List[str], exclude_readme: bool = True) -> List[Path]:
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f'错误: 输入目录不存在: {input_dir}')
        sys.exit(1)
    txt_files = sorted(input_path.glob('*.txt'))
    if exclude_readme:
        exclude_patterns = list(exclude_patterns) + ['README.txt', 'README.md', 'readme.txt', 'readme.md']
    filtered = []
    for f in txt_files:
        if f.name not in exclude_patterns:
            filtered.append(f)
        else:
            print(f'  排除文件: {f.name}')
    return filtered


def read_file_lines(file_path: Path) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def merge_files(file_paths: List[Path], output_file: str, dedup: bool = False):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_lines: List[str] = []
    print('\n' + '=' * 60)
    print('正在合并数据文件...')
    print('=' * 60)
    for fp in file_paths:
        lines = read_file_lines(fp)
        all_lines.extend(lines)
        print(f'  {fp.name:30s} : {len(lines):5d} 行')
    total = len(all_lines)
    if dedup:
        print('\n正在去重...')
        all_lines = list(dict.fromkeys(all_lines))
        duplicate_count = total - len(all_lines)
    else:
        duplicate_count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        if all_lines:
            f.write('\n'.join(all_lines) + '\n')
    print('\n' + '-' * 60)
    print(f'{"总行数":<12} : {total:6d}')
    if dedup:
        print(f'{"去重后":<12} : {len(all_lines):6d}')
        print(f'{"重复行数":<12} : {duplicate_count:6d}')
    print(f'{"输出文件":<12} : {output_file}')
    print('-' * 60)
    return total, len(all_lines), duplicate_count


def show_stats(file_paths: List[Path]) -> int:
    print('\n' + '=' * 60)
    print('数据文件统计')
    print('=' * 60)
    total = 0
    vocab = set()
    for f in file_paths:
        lines = read_file_lines(f)
        total += len(lines)
        for line in lines:
            vocab.update(line.lower().split())
        size_kb = f.stat().st_size / 1024
        print(f'  {f.name:30s} : {len(lines):5d} 行  ({size_kb:6.1f} KB)')
    print('-' * 60)
    print(f'  {"总计":30s} : {total:5d} 行')
    print(f'  {"词表(粗略)":30s} : {len(vocab):5d}')
    print('=' * 60)
    return total


def build_vocabulary(data_file: str, vocab_size: int, min_freq: int, vocab_output: str):
    print('\n' + '=' * 60)
    print('正在构建词汇表...')
    print('=' * 60)
    with open(data_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    # 统一走字符级 BaseTokenizer（零 OOV），输出标准 vocab.json（带 char 标志，
    # 可被 load_vocab 直接加载）。
    vocab = CharTokenizer(vocab_size=vocab_size)
    vocab.train(texts, min_freq=min_freq)
    vocab_path = Path(vocab_output)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(str(vocab_path))
    print(f'\n词汇表已保存: {vocab_output}')
    print('=' * 60)
    return vocab


def update_config(output_file: str) -> None:
    config_path = Path('configs/pretrain.yaml')
    if not config_path.exists():
        print(f'\n警告: 配置文件不存在: {config_path}')
        return
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        new_content = re.sub(r'train_file:\s*"[^"]+"', f'train_file: "{output_file}"', content)
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'\n配置已更新: train_file: "{output_file}"')
    except Exception as e:
        print(f'\n警告: 更新配置文件失败: {e}')


def process_qa_to_jsonl(input_file: str, output_file: str):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    pairs = []
    for i in range(0, len(lines) - 1, 2):
        q, a = lines[i], lines[i + 1]
        if q and a:
            pairs.append({"question": q, "answer": a})
    with open(output_file, 'w', encoding='utf-8') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(f"  {os.path.basename(input_file):30s} -> {os.path.basename(output_file)} ({len(pairs)} 对)")


# --------------------------------------------------------------------------
# 子命令
# --------------------------------------------------------------------------
def cmd_merge(args):
    files = find_data_files(args.input_dir, args.exclude, args.exclude_readme)
    if not files:
        print('\n错误: 未找到任何数据文件')
        sys.exit(1)
    print(f'\n找到 {len(files)} 个数据文件')
    merge_files(files, args.output_file, args.dedup)
    if args.build_vocab:
        build_vocabulary(args.output_file, args.vocab_size, args.min_freq, args.vocab_output)
    if args.update_config:
        update_config(args.output_file)
    print('\n' + '=' * 60)
    print('数据合并完成!')
    print('=' * 60)


def cmd_stats(args):
    files = find_data_files(args.input_dir, [], args.exclude_readme)
    if not files:
        print('\n错误: 未找到任何数据文件')
        sys.exit(1)
    show_stats(files)


def cmd_vocab(args):
    build_vocabulary(args.data_file, args.vocab_size, args.min_freq, args.vocab_output)


def cmd_sample(args):
    datasets_dir = Path(args.input_dir)
    txt_files = [f for f in sorted(datasets_dir.glob('*.txt')) if f.name != 'README.md']
    if args.theme:
        file_path = datasets_dir / f'{args.theme}.txt'
    else:
        print('可用的主题:')
        for f in txt_files:
            print(f'  - {f.stem}')
        choice = input('\n选择主题 (输入名称): ').strip()
        if not choice:
            return
        file_path = datasets_dir / f'{choice}.txt'
    if not file_path.exists():
        print(f'找不到主题: {file_path}')
        return
    lines = read_file_lines(file_path)
    print(f'\n【{file_path.stem}】 - 总共 {len(lines)} 句话')
    print('=' * 60)
    print('\n前5句样本:')
    for i, line in enumerate(lines[:5], 1):
        print(f'{i}. {line}')
    if len(lines) > 5:
        print('\n随机5句样本:')
        for i, line in enumerate(random.sample(lines, min(5, len(lines))), 1):
            print(f'{i}. {line}')


def cmd_to_jsonl(args):
    os.makedirs(args.output_folder, exist_ok=True)
    files = glob.glob(os.path.join(args.input_folder, '*.txt'))
    print(f'找到 {len(files)} 个 txt 文件\n')
    for file_path in files:
        output_path = os.path.join(args.output_folder, os.path.basename(file_path).replace('.txt', '.jsonl'))
        process_qa_to_jsonl(file_path, output_path)
    print('\n全部完成。')


# --------------------------------------------------------------------------
# 命令行 / 兼容入口
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='数据集管理统一入口')
    sub = p.add_subparsers(dest='command')

    pm = sub.add_parser('merge', help='合并 datasets/ 为单一语料')
    pm.add_argument('--input_dir', default='data/datasets')
    pm.add_argument('--output_file', default='data/train_data_combined.txt')
    pm.add_argument('--dedup', action='store_true')
    pm.add_argument('--exclude', nargs='*', default=[])
    pm.add_argument('--build-vocab', dest='build_vocab', action='store_true')
    pm.add_argument('--vocab_size', type=int, default=5000)
    pm.add_argument('--min_freq', type=int, default=1)
    pm.add_argument('--vocab_output', default='data/vocab.json')
    pm.add_argument('--update-config', dest='update_config', action='store_true', default=True)
    pm.add_argument('--no-update-config', dest='update_config', action='store_false')
    pm.add_argument('--exclude_readme', action='store_true', default=True)
    pm.add_argument('--no_exclude_readme', action='store_false', dest='exclude_readme')
    pm.set_defaults(func=cmd_merge)

    ps = sub.add_parser('stats', help='显示统计信息')
    ps.add_argument('--input_dir', default='data/datasets')
    ps.add_argument('--exclude_readme', action='store_true', default=True)
    ps.set_defaults(func=cmd_stats)

    pv = sub.add_parser('vocab', help='构建词表')
    pv.add_argument('--data_file', required=True)
    pv.add_argument('--vocab_size', type=int, default=5000)
    pv.add_argument('--min_freq', type=int, default=1)
    pv.add_argument('--vocab_output', default='data/vocab.json')
    pv.set_defaults(func=cmd_vocab)

    pp = sub.add_parser('sample', help='显示数据样本')
    pp.add_argument('--input_dir', default='data/datasets')
    pp.add_argument('--theme', default=None)
    pp.set_defaults(func=cmd_sample)

    pj = sub.add_parser('to-jsonl', help='将 datasets/ 下的 txt 转为 jsonl')
    pj.add_argument('--input_folder', default='data/datasets')
    pj.add_argument('--output_folder', default='data/processed')
    pj.set_defaults(func=cmd_to_jsonl)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, 'command', None):
        parser.print_help()
        return
    args.func(args)


if __name__ == '__main__':
    main()
