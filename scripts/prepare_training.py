#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练数据管理脚本 - 合并datasets文件夹中的所有QA数据
Train data management - merge all datasets from data/datasets/
"""

import os
import sys
from pathlib import Path
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_data_files():
    """检查datasets文件夹中的数据文件"""
    print("\n" + "="*80)
    print("  📂 数据文件检查 - Check Dataset Files")
    print("="*80)
    
    datasets_dir = Path('data/datasets')
    if not datasets_dir.exists():
        print(f"❌ datasets 文件夹不存在: {datasets_dir}")
        return []
    
    txt_files = sorted(datasets_dir.glob('*.txt'))
    
    print(f"\n数据集位置: {datasets_dir.absolute()}")
    print(f"找到 {len(txt_files)} 个数据文件:\n")
    
    total_lines = 0
    for i, f in enumerate(txt_files, 1):
        with open(f, 'r', encoding='utf-8') as file:
            lines = sum(1 for _ in file if _.strip())
        size_kb = f.stat().st_size / 1024
        print(f"  {i:2d}. {f.name:30s} | {lines:5d} 行 | {size_kb:7.1f} KB")
        total_lines += lines
    
    print(f"\n  总行数 (合并前): {total_lines}")
    return txt_files

def merge_data_files():
    """合并datasets文件夹中的所有数据，去重"""
    print("\n" + "="*80)
    print("  🔄 数据合并和去重 - Merge & Deduplicate")
    print("="*80)
    
    datasets_dir = Path('data/datasets')
    data_dir = Path('data')
    
    if not datasets_dir.exists():
        print(f"❌ datasets 文件夹不存在")
        return None
    
    # 收集所有训练数据
    all_lines = set()
    source_files = sorted(datasets_dir.glob('*.txt'))
    
    print(f"\n合并来自以下文件的数据:")
    for fpath in source_files:
        with open(fpath, 'r', encoding='utf-8') as f:
            file_lines = set(line.strip() for line in f if line.strip())
        print(f"  ✓ {fpath.name:30s} -> {len(file_lines):5d} 条")
        all_lines.update(file_lines)
    
    # 保存合并后的数据
    merged_path = data_dir / 'train_data_final.txt'
    with open(merged_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(all_lines)))
    
    print(f"\n  ✅ 合并完成:")
    print(f"     输出文件: {merged_path.name}")
    print(f"     总行数: {len(all_lines)}")
    print(f"     大小: {merged_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"     (去重后减少 {sum(len(set(open(f, 'r', encoding='utf-8').readlines())) for f in source_files) - len(all_lines)} 重复行)")
    
    return merged_path

def validate_data(filepath, sample_count=3):
    """验证数据格式"""
    print("\n" + "="*80)
    print("  数据格式验证 - Data Validation")
    print("="*80)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"\n  总行数: {len(lines)}")
    
    # 检查[SEP]分隔符
    sep_count = sum(1 for line in lines if '[SEP]' in line or '[sep]' in line)
    print(f"  包含[SEP]分隔符的行: {sep_count}/{len(lines)} ({sep_count*100//len(lines)}%)")
    
    # 显示样本
    print(f"\n  样本数据 (前{sample_count}条):")
    for i, line in enumerate(lines[:sample_count], 1):
        preview = line[:100] + ('...' if len(line) > 100 else '')
        print(f"    {i}. {preview}")
    
    return lines

def update_config(data_file):
    """更新配置文件指向新数据"""
    config_path = Path('configs/pretrain.yaml')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换所有可能的data_file路径
    import re
    pattern = r'train_file:\s*"[^"]+"'
    replacement = f'train_file: "{data_file}"'
    
    new_content = re.sub(pattern, replacement, content)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"\n✅ 配置已更新: train_file: {data_file}")

def main():
    print("\n" + "="*80)
    print("  AI文本生成模型 - 训练数据管理工具")
    print("  Training Data Management Tool for Text Generation")
    print("="*80)
    
    # 1. 检查现有数据
    txt_files = check_data_files()
    
    # 2. 合并数据
    merged_file = merge_data_files()
    
    # 3. 验证数据
    all_lines = validate_data(merged_file)
    
    # 4. 更新配置
    update_config(str(merged_file))
    
    # 5. 显示训练命令
    print("\n" + "="*80)
    print("  GPU训练命令 - Start Training with GPU")
    print("="*80)
    print(f"\n✅ 数据准备完成！现在可以开始训练")
    print(f"\n【方案1】直接启动训练 (使用默认GPU):")
    print(f"  python scripts/train.py --config configs/pretrain.yaml")
    
    print(f"\n【方案2】在后台运行训练 (推荐，不会被关闭):")
    print(f"  python -u scripts/train.py --config configs/pretrain.yaml > training.log 2>&1 &")
    
    print(f"\n【方案3】使用Windows Task Scheduler或nohup:")
    print(f"  nohup python scripts/train.py > training.log 2>&1 &")
    
    print(f"\n【监控训练进度】:")
    print(f"  tail -f training.log                  (实时查看日志)")
    print(f"  python monitor_training.py            (监控脚本)")
    
    print(f"\n【配置说明】:")
    print(f"  - GPU会自动使用 (配置文件device: 'cuda')")
    print(f"  - Batch size: 128 (优化GPU内存)")
    print(f"  - Learning rate: 0.0005")
    print(f"  - 混合精度训练: 启用 (24GB+ GPU推荐)")
    print(f"  - Num workers: 4 (并行数据加载)")
    print(f"\n" + "="*80)

if __name__ == '__main__':
    main()
