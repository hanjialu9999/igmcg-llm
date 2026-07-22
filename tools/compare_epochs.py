#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比不同epoch的模型生成效果
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from models.device import get_device
from models.checkpoint import load_model
import os

device = get_device()

# 全局词表（每个 checkpoint 共用同一词表；final_model 与 model_epoch_*.pt 同源）
_, vocab = load_model('checkpoints/final_model.pt', 'checkpoints/vocab.json', device=device) \
    if os.path.exists('checkpoints/final_model.pt') else (None, None)

def generate_response(model, user_input, temperature=0.7, top_k=50, repetition_penalty=2.0, max_length=20):
    """生成模型回复"""
    tokens = vocab.encode(user_input, add_special_tokens=False)
    tokens = [vocab.bos_idx] + tokens

    with torch.no_grad():
        output_ids = model.generate(
            tokens,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            device=device,
            repetition_penalty=repetition_penalty
        )

    response = vocab.decode(output_ids, skip_special=True)
    input_text = vocab.decode(tokens, skip_special=True)

    if response.startswith(input_text):
        response = response[len(input_text):].strip()

    return response if response else "..."

# 找所有的checkpoint
checkpoint_files = sorted([f for f in Path('checkpoints/').glob('*.pt') if 'epoch' in f.name])

print("\n" + "="*80)
print("对比不同Epoch模型的生成效果")
print("="*80)

test_prompt = "What is artificial intelligence"

# 测试所有available的模型
models_to_test = [
    ('final_model.pt', '最终模型'),
] + [(f.name, f'Epoch {f.name.split("_")[2].split(".")[0]}') for f in checkpoint_files[-3:]]  # 最后3个epoch

for model_file, label in models_to_test:
    checkpoint_path = f'checkpoints/{model_file}'

    if not os.path.exists(checkpoint_path):
        continue

    print(f"\n{label}: {model_file}")
    print("-"*80)

    try:
        # 每个 checkpoint 用 load_model 加载，自动从同目录 *_config.yaml 透传增强开关
        model, _ = load_model(checkpoint_path, 'checkpoints/vocab.json', device=device)
        model.eval()

        response = generate_response(model, test_prompt, temperature=0.7, top_k=50, max_length=25)
        print(f"Q: {test_prompt}")
        print(f"A: {response}")

    except Exception as e:
        print(f"加载失败: {e}")

print("\n" + "="*80)
