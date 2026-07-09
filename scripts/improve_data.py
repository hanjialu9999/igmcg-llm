#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强和质量改进
"""

import os
import glob
import random

# 同义词映射（用于简单的数据增强）
SYNONYM_MAP = {
    'what': ['tell me', 'explain', 'describe', 'what'],
    'how': ['in what way', 'how', 'by what method'],
    'why': ['for what reason', 'why', 'what is the reason'],
    'can you': ['could you', 'can you', 'would you', 'could you please'],
    'is': ['is', 'appears to be', 'seems to be'],
    'artificial intelligence': ['AI', 'artificial intelligence', 'machine intelligence'],
    'machine learning': ['ML', 'machine learning', 'algorithmic learning'],
    'neural network': ['neural network', 'neural net', 'network'],
    'deep learning': ['deep learning', 'deep neural learning'],
}

def augment_data_pair(question, answer):
    """Simple data augmentation for a Q-A pair"""
    augmented = [(question, answer)]  # Original pair
    
    # Try synonym replacement (10% chance)
    if random.random() < 0.1:
        aug_q = question.lower()
        aug_a = answer.lower()
        
        # Replace some common terms
        for key, values in SYNONYM_MAP.items():
            if key in aug_q and len(values) > 1:
                other_synonym = random.choice([v for v in values if v != key])
                aug_q = aug_q.replace(key, other_synonym, 1)
        
        if aug_q != question.lower():
            augmented.append((aug_q.capitalize(), aug_a.capitalize()))
    
    return augmented

def clean_text(text):
    """Clean and normalize text"""
    # Remove extra spaces
    text = ' '.join(text.split())
    
    # Ensure capitalization at start if it's a sentence
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    
    return text

def improve_data(input_file, output_file, augment=True, dedup=True):
    """Improve training data with cleaning and optional augmentation"""
    
    print("="*70)
    print("📊 数据改进和增强")
    print("="*70)
    print(f"\n📖 输入文件: {input_file}")
    
    # Read data
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"📈 原始行数: {len(lines)}")
    
    # Process data
    processed = []
    for line in lines:
        # Skip short or invalid lines
        if len(line) < 5:
            continue
        
        # Clean text
        line = clean_text(line)
        
        # Add to processed
        processed.append(line)
    
    print(f"📈 清理后: {len(processed)} 行")
    
    # Deduplication
    if dedup:
        unique = []
        seen = set()
        for line in processed:
            if line not in seen:
                unique.append(line)
                seen.add(line)
        processed = unique
        print(f"📈 去重后: {len(processed)} 行")
    
    # Augmentation
    if augment:
        augmented = []
        for line in processed:
            if '[SEP]' in line:
                parts = line.split(' [SEP] ')
                if len(parts) == 2:
                    q, a = parts
                    pairs = augment_data_pair(q, a)
                    for aug_q, aug_a in pairs:
                        augmented.append(f"{aug_q} [SEP] {aug_a}")
            else:
                augmented.append(line)
        processed = augmented
        print(f"📈 增强后: {len(processed)} 行")
    
    # Save improved data
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(processed))
    
    print(f"\n✅ 数据改进完成！")
    print(f"   输出文件: {output_file}")
    print(f"   最终行数: {len(processed)}")
    print("\n" + "="*70)

def analyze_data(input_file):
    """Analyze data statistics"""
    
    print("\n📊 数据分析:")
    print("-"*70)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # 基本统计
    print(f"总行数: {len(lines)}")
    
    # Q-A对数统计
    qa_pairs = [l for l in lines if '[SEP]' in l]
    others = [l for l in lines if '[SEP]' not in l]
    
    print(f"Q-A 对: {len(qa_pairs)}")
    print(f"其他句子: {len(others)}")
    
    # 长度统计
    lengths = [len(l.split()) for l in lines]
    print(f"\n句子长度统计:")
    print(f"  最短: {min(lengths)} 词")
    print(f"  最长: {max(lengths)} 词")
    print(f"  平均: {sum(lengths)/len(lengths):.1f} 词")
    
    # 问句统计
    question_count = sum(1 for l in lines if l.strip().endswith('?'))
    print(f"\n包含问号的行: {question_count}")
    
    print("-"*70)

if __name__ == '__main__':
    # Improve data
    improve_data(
        'data/train_data_combined.txt',
        'data/train_data_improved.txt',
        augment=True,
        dedup=True
    )
    
    # Analyze
    analyze_data('data/train_data_improved.txt')
