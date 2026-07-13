"""
数据集管理工具 - 查看、合并、统计数据
"""

import os
import sys
from pathlib import Path
from collections import Counter
import random

def show_menu():
    """显示菜单"""
    print("\n" + "="*50)
    print("数据集管理工具")
    print("="*50)
    print("1. 显示所有数据统计")
    print("2. 查看特定主题的数据片段")
    print("3. 合并所有数据集")
    print("4. 计算词表大小")
    print("5. 显示数据样本")
    print("6. 退出")
    print("="*50)
    choice = input("选择 (1-6): ").strip()
    return choice


def count_lines(file_path):
    """统计文件行数"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [l for l in f.readlines() if l.strip()]
        return len(lines)
    except Exception:
        return 0


def get_vocab_size(file_path):
    """计算词表大小"""
    words = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 简单的单词分割
                words_in_line = line.lower().split()
                words.update(words_in_line)
    except Exception:
        pass
    return len(words)


def get_avg_length(file_path):
    """计算平均句子长度"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            return 0
        total_words = sum(len(line.split()) for line in lines)
        return total_words / len(lines)
    except Exception:
        return 0


def show_stats():
    """显示统计信息"""
    print("\n" + "="*60)
    print("数据集统计信息")
    print("="*60)
    
    datasets_dir = Path('data/datasets')
    txt_files = sorted(datasets_dir.glob('*.txt'))
    
    print(f"\n{'文件名':<30} {'句数':<8} {'词表':<8} {'平均长度':<10}")
    print("-"*60)
    
    total_lines = 0
    total_vocab = set()
    
    for txt_file in txt_files:
        if txt_file.name == 'README.md':
            continue
        
        count = count_lines(txt_file)
        avg_len = get_avg_length(txt_file)
        
        # 读取单词
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    total_vocab.update(line.lower().split())
        except Exception:
            pass
        
        total_lines += count
        print(f"{txt_file.name:<30} {count:<8} {'-':<8} {avg_len:>6.1f} 词")
    
    print("-"*60)
    print(f"{'总计':<30} {total_lines:<8} {len(total_vocab):<8}")
    print("="*60)


def show_sample(theme=None):
    """显示数据样本"""
    datasets_dir = Path('data/datasets')
    
    if theme:
        file_path = datasets_dir / f'{theme}.txt'
        if not file_path.exists():
            print(f"找不到主题: {theme}")
            return
    else:
        print("\n可用的主题:")
        themes = []
        for f in sorted(datasets_dir.glob('*.txt')):
            if f.name != 'README.md':
                theme_name = f.stem
                themes.append(theme_name)
                print(f"  - {theme_name}")
        
        choice = input("\n选择主题 (输入名称): ").strip()
        if not choice:
            return
        file_path = datasets_dir / f'{choice}.txt'
        if not file_path.exists():
            print(f"找不到主题: {choice}")
            return
    
    # 读取和显示样本
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
    except Exception:
        print("读取文件失败")
        return
    
    print(f"\n【{file_path.stem}】 - 总共 {len(lines)} 句话")
    print("="*60)
    
    # 显示前5句
    print("\n前5句样本:")
    for i, line in enumerate(lines[:5], 1):
        print(f"{i}. {line}")
    
    # 显示随机5句
    if len(lines) > 5:
        print("\n随机5句样本:")
        for i, line in enumerate(random.sample(lines, min(5, len(lines))), 1):
            print(f"{i}. {line}")


def merge_datasets():
    """合并数据集"""
    datasets_dir = Path('data/datasets')
    txt_files = sorted(datasets_dir.glob('*.txt'))
    
    print("\n" + "="*60)
    print("合并数据集")
    print("="*60)
    
    # 选择输出文件名
    output = input("\n输出文件名 (默认: data/train_data_combined.txt): ").strip()
    if not output:
        output = 'data/train_data_combined.txt'
    
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n开始合并...")
    total_lines = 0
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for txt_file in txt_files:
            if txt_file.name == 'README.md':
                continue
            
            print(f"合并 {txt_file.name}...", end=' ')
            with open(txt_file, 'r', encoding='utf-8') as infile:
                lines = infile.readlines()
                outfile.writelines(lines)
                total_lines += len(lines)
                print(f"✓ ({len(lines)} 句)")
    
    print("="*60)
    print(f"✓ 合并完成!")
    print(f"输出文件: {output_path}")
    print(f"总句数: {total_lines}")
    print(f"\n接下来的步骤:")
    print(f"1. 编辑 configs/pretrain.yaml")
    print(f'2. 修改 train_file: "{output}"')
    print(f"3. 运行 python scripts/train.py")
    print("="*60)


def calc_vocab():
    """计算并显示词表信息"""
    print("\n" + "="*60)
    print("词表计算")
    print("="*60)
    
    # 选择文件
    print("\n1. 计算单个文件的词表")
    print("2. 计算合并后的总词表")
    choice = input("\n选择 (1-2): ").strip()
    
    if choice == '1':
        datasets_dir = Path('data/datasets')
        print("\n可用的数据文件:")
        files = list(datasets_dir.glob('*.txt'))
        for i, f in enumerate(files, 1):
            if f.name != 'README.md':
                print(f"  {i}. {f.name}")
        
        try:
            idx = int(input("选择 (输入数字): ")) - 1
            file_path = list(datasets_dir.glob('*.txt'))[idx]
        except Exception:
            print("选择无效")
            return
    else:
        file_path = Path('data/train_data_combined.txt')
        if not file_path.exists():
            print("未找到合并文件，请先合并数据")
            return
    
    # 计算词表
    print(f"\n分析 {file_path.name}...")
    words = Counter()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                words.update(line.lower().split())
    except Exception:
        print("读取文件失败")
        return
    
    vocab_size = len(words)
    total_words = sum(words.values())
    
    print("="*60)
    print(f"词表大小: {vocab_size}")
    print(f"总词数: {total_words}")
    print(f"平均频率: {total_words/vocab_size:.2f}")
    print("\n最频繁的10个词:")
    for word, count in words.most_common(10):
        print(f"  {word:<20} {count:>6} 次")
    print("="*60)


def main():
    """主菜单循环"""
    while True:
        choice = show_menu()
        
        if choice == '1':
            show_stats()
        elif choice == '2':
            show_sample()
        elif choice == '3':
            merge_datasets()
        elif choice == '4':
            calc_vocab()
        elif choice == '5':
            show_sample()
        elif choice == '6':
            print("再见!")
            break
        else:
            print("无效选择，请重试!")


if __name__ == '__main__':
    # 支持命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == 'stats':
            show_stats()
        elif sys.argv[1] == 'merge':
            merge_datasets()
        elif sys.argv[1] == 'vocab':
            calc_vocab()
        elif sys.argv[1] == 'sample':
            show_sample(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        # 菜单模式
        main()
