#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temperature Tuning Script - 系统性地测试不同温度值找最优点
控制变量：固定 top_k=42, repetition_penalty=2.0
"""
import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import json
import argparse
import numpy as np
from models.device import get_device
from models.checkpoint import load_model

parser = argparse.ArgumentParser(description='Temperature 调参')
parser.add_argument('--model', default='checkpoints/final_model.pt')
parser.add_argument('--vocab', default='checkpoints/vocab.json')
parser.add_argument('--device', default=None)
args = parser.parse_args()

device = get_device(args.device)

# 复用 generate.load_model：从 *_config.yaml 透传增强开关（qk_norm/attn_temp 等），
# 避免 state_dict 不匹配；strict=False 兼容旧权重。比 build_model(load_config()) 更贴合实际权重。
model, vocab = load_model(args.model, args.vocab, device=device, quantize=False, compile_model=False)
model.eval()

test_questions = [
    'What is machine learning',
    'How does deep learning work',
    'Artificial intelligence can help',
]

# Temperature range: 0.3 to 1.2 in 0.1 increments
temperatures = [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2]

print('\n' + '='*110)
print('  TEMPERATURE TUNING - Finding Optimal Temperature')
print('  Control Variables: Top-K=42, Repetition Penalty=2.0')
print('='*110)

results = {}

for temp in temperatures:
    print('\n' + '-'*110)
    print(f'  TEMPERATURE = {temp}')
    print('-'*110 + '\n')
    
    temp_results = []
    
    with torch.no_grad():
        for i, q in enumerate(test_questions, 1):
            tokens = [vocab.bos_idx] + vocab.encode(q, add_special_tokens=False)
            out = model.generate(
                tokens, 
                max_length=50, 
                temperature=temp, 
                top_k=42, 
                device=device,
                repetition_penalty=2.0
            )
            resp = vocab.decode(out, skip_special=True)
            
            print(f'  {i}. [{q}]')
            print(f'     -> {resp[:125]}')
            print()
            
            temp_results.append(resp)
    
    results[temp] = temp_results

print('\n' + '='*110)
print('  ANALYSIS SUMMARY')
print('='*110)
print('\n  Temperature Characteristics:')
print('  • 0.3-0.5:   Very deterministic, may be repetitive')
print('  • 0.6-0.8:   Balanced, good for most use cases')
print('  • 0.9-1.2:   More creative, higher randomness')
print('\n  Recommended next step: Run top-k tuning with optimal temperature')
print('='*110 + '\n')
