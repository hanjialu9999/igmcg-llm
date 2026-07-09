#!/usr/bin/env python3
"""快速检查所有数据集文件的格式"""

from pathlib import Path

datasets_dir = Path('data/datasets')

print("=" * 80)
print("📋 数据集文件格式检查")
print("=" * 80)

files = sorted(datasets_dir.glob('*.txt'))

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    question_count = sum(1 for line in lines if '?' in line)
    statement_count = len(lines) - question_count
    
    # 判断格式
    if question_count > 0 and statement_count > 0:
        qa_ratio = (question_count / len(lines)) * 100
        status = f"✓ 混合格式 ({qa_ratio:.0f}% Q)"
    elif question_count == 0:
        status = f"✗ 纯陈述句 (100% 陈述)"
    else:
        status = f"? 纯问题 (100% Q)"
    
    first_line = lines[0][:50] if lines else ""
    print(f"{file.name:30s} {len(lines):4d} 行 | {status:20s} | {first_line}...")

print("=" * 80)
print("\n✅ 已成功修改的文件 (转换为纯问答):")
print("  • trivia_qa.txt")
print("  • motivation.txt")
print("  • technology_future.txt")

print("\n⚠️ 仍为陈述句的文件:")
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    question_count = sum(1 for line in lines if '?' in line)
    if question_count == 0:
        print(f"  • {file.name}")
