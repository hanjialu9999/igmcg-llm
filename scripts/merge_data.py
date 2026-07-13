#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并训练数据脚本
合并 data/datasets/ 目录下的所有数据文件，支持去重、排除模式、词汇表构建
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Set, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.data_utils import Vocabulary


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='合并训练数据文件')
    parser.add_argument(
        '--input_dir',
        type=str,
        default='data/datasets',
        help='输入数据目录 (默认: data/datasets)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='data/train_data_combined.txt',
        help='输出文件路径 (默认: data/train_data_combined.txt)'
    )
    parser.add_argument(
        '--dedup',
        action='store_true',
        help='启用去重 (默认: False)'
    )
    parser.add_argument(
        '--exclude',
        nargs='*',
        default=[],
        help='排除的文件模式列表 (如: math_training.txt noisy_input_robust.txt)'
    )
    parser.add_argument(
        '--build_vocab',
        action='store_true',
        help='合并后构建词汇表 (默认: False)'
    )
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=5000,
        help='词汇表大小 (默认: 5000)'
    )
    parser.add_argument(
        '--min_freq',
        type=int,
        default=1,
        help='最小词频 (默认: 1)'
    )
    parser.add_argument(
        '--vocab_output',
        type=str,
        default='data/vocab.json',
        help='词汇表输出路径 (默认: data/vocab.json)'
    )
    parser.add_argument(
        '--stats_only',
        action='store_true',
        help='仅显示统计信息，不合并文件 (默认: False)'
    )
    parser.add_argument(
        '--exclude_readme',
        action='store_true',
        default=True,
        help='自动排除 README 文件 (默认: True)'
    )
    parser.add_argument(
        '--no_exclude_readme',
        action='store_false',
        dest='exclude_readme',
        help='不自动排除 README 文件'
    )
    return parser.parse_args()


def find_data_files(input_dir: str, exclude_patterns: List[str], exclude_readme: bool = True) -> List[Path]:
    """查找所有数据文件"""
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f'错误: 输入目录不存在: {input_dir}')
        sys.exit(1)

    txt_files = sorted(input_path.glob('*.txt'))

    if exclude_readme:
        exclude_patterns = list(exclude_patterns) + ['README.txt', 'README.md', 'readme.txt', 'readme.md']

    filtered_files = []
    for f in txt_files:
        if f.name not in exclude_patterns:
            filtered_files.append(f)
        else:
            print(f'  排除文件: {f.name}')

    return filtered_files


def read_file_lines(file_path: Path) -> List[str]:
    """读取文件所有非空行"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def merge_files(
    file_paths: List[Path],
    output_file: str,
    dedup: bool = False
) -> tuple[int, int, int]:
    """
    合并文件
    返回: (总行数, 去重后行数, 重复行数)
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_lines: List[str] = []
    file_stats = []

    print('\n' + '=' * 60)
    print('正在合并数据文件...')
    print('=' * 60)

    for file_path in file_paths:
        lines = read_file_lines(file_path)
        count = len(lines)
        file_stats.append((file_path.name, count))
        all_lines.extend(lines)
        print(f'  {file_path.name:30s} : {count:5d} 行')

    total_lines = len(all_lines)

    if dedup:
        print('\n正在去重...')
        unique_lines = list(dict.fromkeys(all_lines))  # 保持顺序去重
        duplicate_count = total_lines - len(unique_lines)
        all_lines = unique_lines
    else:
        duplicate_count = 0
        unique_lines = total_lines

    # 写入输出文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_lines))
        if all_lines:
            f.write('\n')

    # 打印统计信息
    print('\n' + '-' * 60)
    print(f'{"总行数":<12} : {total_lines:6d}')
    if dedup:
        print(f'{"去重后":<12} : {unique_lines:6d}')
        print(f'{"重复行数":<12} : {duplicate_count:6d}')
    print(f'{"输出文件":<12} : {output_file}')
    print('-' * 60)

    return total_lines, unique_lines, duplicate_count


def show_stats(file_paths: List[Path]) -> int:
    """显示文件统计信息"""
    print('\n' + '=' * 60)
    print('数据文件统计')
    print('=' * 60)

    total = 0
    for f in file_paths:
        lines = read_file_lines(f)
        count = len(lines)
        total += count
        size_kb = f.stat().st_size / 1024
        print(f'  {f.name:30s} : {count:5d} 行  ({size_kb:6.1f} KB)')

    print('-' * 60)
    print(f'  {"总计":30s} : {total:5d} 行')
    print('=' * 60)

    return total


def build_vocabulary(data_file: str, vocab_size: int, min_freq: int, vocab_output: str) -> Vocabulary:
    """构建词汇表"""
    print('\n' + '=' * 60)
    print('正在构建词汇表...')
    print('=' * 60)

    with open(data_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]

    vocab = Vocabulary(vocab_size=vocab_size, min_freq=min_freq)
    vocab.build_vocab(texts)

    # 保存词汇表
    vocab_path = Path(vocab_output)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    vocab_data = {
        'word2idx': vocab.word2idx,
        'idx2word': {str(k): v for k, v in vocab.idx2word.items()},
        'special_tokens': vocab.special_tokens,
        'vocab_size': vocab.vocab_size,
        'min_freq': vocab.min_freq
    }
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    print(f'\n词汇表已保存: {vocab_output}')
    print('=' * 60)

    return vocab


def update_config(output_file: str) -> None:
    """更新配置文件中的训练数据路径"""
    config_path = Path('configs/pretrain.yaml')
    if not config_path.exists():
        print(f'\n警告: 配置文件不存在: {config_path}')
        return

    try:
        import re
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()

        pattern = r'train_file:\s*"[^"]+"'
        replacement = f'train_file: "{output_file}"'
        new_content = re.sub(pattern, replacement, content)

        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f'\n配置已更新: train_file: "{output_file}"')
    except Exception as e:
        print(f'\n警告: 更新配置文件失败: {e}')


def main() -> None:
    args = parse_args()

    print('=' * 60)
    print('数据合并工具')
    print('=' * 60)
    print(f'输入目录: {args.input_dir}')
    print(f'输出文件: {args.output_file}')
    print(f'去重模式: {"开启" if args.dedup else "关闭"}')
    if args.exclude:
        print(f'排除模式: {", ".join(args.exclude)}')
    if args.exclude_readme:
        print('自动排除: README 文件')

    # 查找数据文件
    data_files = find_data_files(args.input_dir, args.exclude, args.exclude_readme)

    if not data_files:
        print('\n错误: 未找到任何数据文件')
        sys.exit(1)

    print(f'\n找到 {len(data_files)} 个数据文件')

    # 仅显示统计信息
    if args.stats_only:
        show_stats(data_files)
        return

    # 合并文件
    total_lines, unique_lines, duplicate_count = merge_files(
        data_files,
        args.output_file,
        args.dedup
    )

    # 构建词汇表
    if args.build_vocab:
        build_vocabulary(
            args.output_file,
            args.vocab_size,
            args.min_freq,
            args.vocab_output
        )

    # 更新配置文件
    update_config(args.output_file)

    print('\n' + '=' * 60)
    print('数据合并完成!')
    print('=' * 60)
    print('\n下一步:')
    print('  python scripts/train.py')
    print('=' * 60)


if __name__ == '__main__':
    main()