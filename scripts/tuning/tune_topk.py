#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-K Tuning Script - 系统性地测试不同top-k值找最优点
控制变量：固定 temperature=0.65（根据温度测试的最佳值）, repetition_penalty=2.0
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import json
from models.config_loader import load_config, build_model, load_vocab

vocab = load_vocab('checkpoints/vocab.json')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = build_model(load_config()).to(device)
cp = torch.load('checkpoints/final_model.pt', map_location=device)
model.load_state_dict(cp['model_state_dict'])
model.to(device).eval()

test_questions = [
    'What is machine learning',
    'How does deep learning work',
    'Artificial intelligence can help',
]

# Top-K range: 25 to 60 in 5 increments
top_ks = [25, 30, 35, 40, 42, 45, 50, 55, 60]

print('\n' + '='*110)
print('  TOP-K TUNING - Finding Optimal Top-K Value')
print('  Control Variables: Temperature=0.65, Repetition Penalty=2.0')
print('='*110)

results = {}

for top_k in top_ks:
    print('\n' + '-'*110)
    print(f'  TOP-K = {top_k}')
    print('-'*110 + '\n')
    
    topk_results = []
    
    with torch.no_grad():
        for i, q in enumerate(test_questions, 1):
            tokens = [vocab.bos_idx] + vocab.encode(q, add_special_tokens=False)
            out = model.generate(
                tokens, 
                max_length=50, 
                temperature=0.65, 
                top_k=top_k, 
                device=device,
                repetition_penalty=2.0
            )
            resp = vocab.decode(out, skip_special=True)
            
            print(f'  {i}. [{q}]')
            print(f'     -> {resp[:125]}')
            print()
            
            topk_results.append(resp)
    
    results[top_k] = topk_results

print('\n' + '='*110)
print('  ANALYSIS SUMMARY')
print('='*110)
print('\n  Top-K Characteristics:')
print('  • 25-35:     Conservative, less random, may be repetitive')
print('  • 40-50:     Balanced diversity and coherence')
print('  • 55-60:     More creative, higher randomness')
print('\n  Recommendation: Choose top-k that balances diversity + coherence')
print('='*110 + '\n')
