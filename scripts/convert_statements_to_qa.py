#!/usr/bin/env python3
"""
将纯陈述句转换为问答格式
"""

import re
from pathlib import Path

def convert_statement_to_qa(statement):
    """
    将陈述句转换为问答对
    例如: "Success comes from hard work." 
    变成: "What is the key to success?\nHard work is the key to success."
    """
    statement = statement.strip()
    if not statement or statement.endswith('?'):
        return statement
    
    # 移除句号和感叹号
    statement = re.sub(r'[.!]+$', '', statement)
    
    # 通用问题生成（可以根据陈述句模式优化）
    templates = [
        (r"^(\w+) is (.+)$", lambda m: f"What is {m.group(1)}?\n{statement}."),
        (r"^(\w+) are (.+)$", lambda m: f"What are {m.group(1)}?\n{statement}."),
        (r"^(\w+) helps? (.+)$", lambda m: f"How does {m.group(1)} help?\n{statement}."),
        (r"^(\w+) makes? (.+)$", lambda m: f"What makes {m.group(1)}?\n{statement}."),
        (r"^(\w+) creates? (.+)$", lambda m: f"What does {m.group(1)} create?\n{statement}."),
        (r"^(\w+) provides? (.+)$", lambda m: f"What does {m.group(1)} provide?\n{statement}."),
        (r"^(\w+) allows? (.+)$", lambda m: f"What does {m.group(1)} allow?\n{statement}."),
        (r"^(\w+) can (.+)$", lambda m: f"What can {m.group(1)} do?\n{statement}."),
        (r"^(\w+) brings? (.+)$", lambda m: f"What does {m.group(1)} bring?\n{statement}."),
    ]
    
    for pattern, formatter in templates:
        match = re.match(pattern, statement, re.IGNORECASE)
        if match:
            return formatter(match)
    
    # 默认格式
    return f"Tell me about this: {statement}?\n{statement}."

def process_file(input_file, output_file):
    """处理一个文件，转换纯陈述句为问答格式"""
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    result_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # 如果是问题，直接保留
        if '?' in line:
            result_lines.append(line)
            i += 1
        # 如果是陈述句
        else:
            # 检查下一行是否也是陈述句（没有问号）
            next_is_answer = (i + 1 < len(lines) and 
                             '?' not in lines[i + 1] and 
                             lines[i + 1].strip())
            
            if next_is_answer:
                # 下一行是回答，跳过这个陈述句的转换
                result_lines.append(line)
                i += 1
            else:
                # 独立的陈述句，转换为问答
                if line:  # 非空行
                    qa_pair = convert_statement_to_qa(line)
                    result_lines.append(qa_pair)
                else:
                    result_lines.append(line)
                i += 1
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in result_lines:
            f.write(line + '\n')
    
    return len(result_lines)

# 处理需要转换的文件
files_to_convert = [
    'data/datasets/trivia_qa.txt',
    'data/datasets/motivation.txt',
    'data/datasets/technology_future.txt',
]

print("=" * 70)
print("🔄 陈述句转换为问答格式")
print("=" * 70)

for file_path in files_to_convert:
    if Path(file_path).exists():
        output_file = file_path.replace('.txt', '_converted.txt')
        line_count = process_file(file_path, output_file)
        print(f"\n✓ {Path(file_path).name}")
        print(f"  输出: {output_file}")
        print(f"  行数: {line_count}")
    else:
        print(f"\n✗ {file_path} 不存在")

print("\n" + "=" * 70)
print("接下来可以用转换后的文件替换原文件")
print("=" * 70)
