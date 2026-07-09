#!/usr/bin/env python3
"""
合并所有训练数据并生成统计报告
"""

import os
import glob

data_dir = 'data/datasets'
output_file = 'data/train_data_combined.txt'

# 获取所有txt文件
txt_files = sorted(glob.glob(os.path.join(data_dir, '*.txt')))
txt_files = [f for f in txt_files if 'README' not in f]  # 排除README

print("="*70)
print("📊 合并训练数据")
print("="*70)

total_sentences = 0
file_stats = []

# 读取并合并
with open(output_file, 'w', encoding='utf-8') as out:
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            count = len(lines)
            total_sentences += count
            file_stats.append((os.path.basename(txt_file), count))
            out.write('\n'.join(lines) + '\n')

# 打印统计
print("\n📈 各文件数据量：\n")
for filename, count in file_stats:
    print(f"  {filename:30s} : {count:4d} 行")

print(f"\n{'─'*70}")
print(f"{'📦 总计':<30s} : {total_sentences:4d} 行")
print(f"{'─'*70}\n")

# 检查重复
with open(output_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    unique_lines = len(set(line.strip() for line in lines if line.strip()))
    duplicate_count = len(lines) - unique_lines

print(f"✅ 数据合并完成！")
print(f"  • 总句子数: {total_sentences}")
print(f"  • 去重后: {unique_lines}")
print(f"  • 重复数: {duplicate_count}")
print(f"  • 输出文件: {output_file}")
print("\n现在可以运行训练：")
print("  python scripts/train.py")
print("\n" + "="*70)
