#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进数据格式: 将相邻的Q-A对合并为一行
"""

import os
import glob

data_dir = 'data/datasets'
output_file = 'data/train_data_formatted.txt'

# 获取所有txt文件，排除math和noisy
txt_files = sorted(glob.glob(os.path.join(data_dir, '*.txt')))
txt_files = [f for f in txt_files if 'README' not in f]
exclude_files = {'math_training.txt', 'noisy_input_robust.txt'}
txt_files = [f for f in txt_files if os.path.basename(f) not in exclude_files]

print("="*70)
print("🔄 改进数据格式 - 将相邻的Q-A对合并为一行")
print("="*70)

total_pairs = 0
pair_counts = []

with open(output_file, 'w', encoding='utf-8') as out:
    for txt_file in txt_files:
        filename = os.path.basename(txt_file)
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            all_lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # 按照问题形式配对
        # 规则: 以?结尾的是问题，否则是答案或语句
        pairs = []
        i = 0
        while i < len(all_lines):
            line = all_lines[i]
            
            # 如果当前行以?结尾，认为是问题
            if line.endswith('?'):
                question = line
                # 查找下一个答案（非问句）
                if i + 1 < len(all_lines):
                    answer = all_lines[i + 1]
                    # 合并为一对，用 [SEP] 分隔
                    pairs.append(f"{question} [SEP] {answer}")
                    total_pairs += 1
                    i += 2
                else:
                    i += 1
            else:
                # 非问题句子，作为独立输入
                pairs.append(line)
                total_pairs += 1
                i += 1
        
        pair_counts.append((filename, len(pairs)))
        out.write('\n'.join(pairs) + '\n')

print("\n📈 各文件数据对数量：\n")
for filename, count in pair_counts:
    print(f"  {filename:30s} : {count:4d} 对/句")

print(f"\n{'─'*70}")
print(f"{'📦 总计':<30s} : {total_pairs:4d} 对/句")

# 检查重复
with open(output_file, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]
    unique_lines = len(set(lines))
    duplicate_count = len(lines) - unique_lines

print(f"{'─'*70}\n")
print(f"✅ 数据格式改进完成！")
print(f"  • 总对数: {total_pairs}")
print(f"  • 去重后: {unique_lines}")
print(f"  • 重复数: {duplicate_count}")
print(f"  • 输出文件: {output_file}")
print(f"  • 格式说明: <question> [SEP] <answer>")
print("\n" + "="*70)
