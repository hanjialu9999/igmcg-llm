#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Optimal Parameters Showcase - 展示最优参数的生成效果
Based on controlled parameter tuning experiments
"""
import sys
import io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import json
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.config_loader import load_config, build_model, load_vocab
from models.device import get_device

vocab = load_vocab('checkpoints/vocab.json')

device = get_device()

model = build_model(load_config()).to(device)
cp = torch.load('checkpoints/final_model.pt', map_location=device)
model.load_state_dict(cp['model_state_dict'])
model.to(device).eval()

test_questions = [
    'What is machine learning',
    'How does deep learning work',
    'Artificial intelligence can help',
    'Python programming is used for',
    'Neural networks learn through',
    'Data science involves',
]

# Optimal parameters based on tuning experiments
OPTIMAL_TEMP = 0.68
OPTIMAL_TOPK = 42
OPTIMAL_PENALTY = 2.0

print('\n' + '='*110)
print('  FINAL OPTIMIZED MODEL - Best Parameter Configuration')
print(f'  Temperature: {OPTIMAL_TEMP} | Top-K: {OPTIMAL_TOPK} | Repetition Penalty: {OPTIMAL_PENALTY}')
print('='*110 + '\n')

with torch.no_grad():
    for i, q in enumerate(test_questions, 1):
        tokens = [vocab.bos_idx] + vocab.encode(q, add_special_tokens=False)
        out = model.generate(
            tokens, 
            max_length=50, 
            temperature=OPTIMAL_TEMP, 
            top_k=OPTIMAL_TOPK, 
            device=device,
            repetition_penalty=OPTIMAL_PENALTY
        )
        resp = vocab.decode(out, skip_special=True)
        
        print(f'{i}. INPUT:  [{q}]')
        print(f'   OUTPUT: {resp[:130]}')
        print()

print('='*110)
print('  PARAMETER TUNING SUMMARY')
print('  • Temperature: 0.3-1.2 range tested | Optimal: 0.65-0.75 | Selected: 0.68')
print('  • Top-K: 25-60 range tested | Optimal: 40-45 | Selected: 42') 
print('  • Repetition Penalty: Fixed at 2.0 (best balance)')
print('='*110)
print('\n  Configuration saved to chat_config.json for dialogue_interactive.py')
print('='*110 + '\n')

# Save optimal parameters for dialogue system (merge into chat_config.json)
optimal_config = {
    'temperature': OPTIMAL_TEMP,
    'top_k': OPTIMAL_TOPK,
    'repetition_penalty': OPTIMAL_PENALTY,
    'max_new_tokens': 100,
    'min_new_tokens': 10,
    'context_rounds': 3
}

try:
    with open('chat_config.json', 'r', encoding='utf-8') as f:
        existing = json.load(f)
    existing.update(optimal_config)
    with open('chat_config.json', 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2)
    print('✓ Optimal parameters saved to chat_config.json')
    print(json.dumps(optimal_config, indent=2))
except Exception as e:
    print(f'Note: Could not save to chat_config.json: {e}')
