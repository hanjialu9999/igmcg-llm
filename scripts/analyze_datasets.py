#!/usr/bin/env python3
"""
数据集格式分析脚本
"""

import os
from pathlib import Path

datasets_dir = 'data/datasets'

print("=" * 70)
print("📊 数据集格式分析")
print("=" * 70)

results = []

for file in sorted(Path(datasets_dir).glob('*.txt')):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    line_count = len(lines)
    question_count = sum(1 for line in lines if '?' in line)
    statement_count = sum(1 for line in lines if '?' not in line and line.strip())
    
    # 检查是否有连续的问/答对（一问一答格式）
    has_qa_pattern = False
    for i in range(len(lines) - 1):
        if '?' in lines[i] and '?' not in lines[i + 1]:
            has_qa_pattern = True
            break
    
    first_line = lines[0].strip() if lines else ""
    
    results.append({
        'file': file.name,
        'lines': line_count,
        'questions': question_count,
        'statements': statement_count,
        'has_qa': has_qa_pattern,
        'first': first_line[:60]
    })

# 按是否有问题排序
qa_files = [r for r in results if r['has_qa']]
statement_files = [r for r in results if not r['has_qa']]

print("\n✓ 一问一答格式的文件:")
print("-" * 70)
for r in qa_files:
    print(f"  {r['file']:25s} {r['lines']:3d} 行 | "
          f"问题: {r['questions']:2d} | "
          f"陈述: {r['statements']:2d}")

print("\n✗ 纯陈述句格式的文件:")
print("-" * 70)
for r in statement_files:
    print(f"  {r['file']:25s} {r['lines']:3d} 行 | "
          f"陈述: {r['statements']:2d}")
    if r['first']:
        print(f"    首行: {r['first']}")

print("\n" + "=" * 70)
print(f"📈 统计:")
print(f"  一问一答格式: {len(qa_files)} 个文件")
print(f"  纯陈述句格式: {len(statement_files)} 个文件")
print("=" * 70)

print("\n💡 建议:")
if len(statement_files) > 0:
    print(f"  纯陈述句格式的文件可能不适合用于文本生成/对话模型：")
    for r in statement_files:
        print(f"    • {r['file']}")
    print("\n  这些文件可以：")
    print("    1. 转换为问答格式")
    print("    2. 作为知识库用于上下文增强")
    print("    3. 从训练数据中移除")
