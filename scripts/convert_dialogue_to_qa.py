#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将对话行格式转换为 QA 对格式 (question [SEP] answer)
Convert dialogue lines to QA pair format
"""

from pathlib import Path

def convert_dialogue_lines_to_qa():
    """从datasets中的对话行转换为QA对格式"""
    print("\n" + "="*80)
    print("  📝 对话行 → QA对 转换")
    print("="*80)
    
    datasets_dir = Path('data/datasets')
    all_qa_pairs = []
    
    # 处理每个数据文件
    for txt_file in sorted(datasets_dir.glob('*.txt')):
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # 判断格式类型
        if _is_dialogue_lines(lines):
            # 对话行格式: 奇数行是问题，偶数行是答案
            qa_pairs = _convert_dialogue_lines(lines)
            print(f"  ✓ {txt_file.name:30s} -> {len(qa_pairs):5d} QA对")
        else:
            # 已经是QA对或其他格式，直接使用
            qa_pairs = lines
            print(f"  ✓ {txt_file.name:30s} -> {len(qa_pairs):5d} 行")
        
        all_qa_pairs.extend(qa_pairs)
    
    # 去重
    all_qa_pairs = list(set(all_qa_pairs))
    
    # 保存为 [SEP] 格式的QA对
    output_file = Path('data/train_data_final.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(all_qa_pairs)))
    
    print(f"\n✅ 转换完成:")
    print(f"   输出文件: {output_file.name}")
    print(f"   总QA对: {len(all_qa_pairs)}")
    print(f"   大小: {output_file.stat().st_size / (1024*1024):.2f} MB")
    
    return output_file, len(all_qa_pairs)

def _is_dialogue_lines(lines, sample_size=10):
    """判断是否为对话行格式"""
    if not lines:
        return False
    
    # 对话行通常没有 [SEP] 分隔符
    sep_count = sum(1 for line in lines[:sample_size] if '[SEP]' in line)
    return sep_count == 0

def _convert_dialogue_lines(lines):
    """将对话行转换为QA对"""
    qa_pairs = []
    
    i = 0
    while i < len(lines) - 1:
        question = lines[i].strip()
        answer = lines[i + 1].strip()
        
        if question and answer:
            qa_pair = f"{question} [SEP] {answer}"
            qa_pairs.append(qa_pair)
        
        i += 2
    
    return qa_pairs

if __name__ == '__main__':
    convert_dialogue_lines_to_qa()
    print("\n💡 提示: 现在可以运行 python scripts/train.py --config configs/pretrain.yaml 开始训练")
