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
from models.transformer import TransformerModel
from models.data_utils import Vocabulary
from models.device import get_device
import yaml
import os
from pathlib import Path

device = get_device()

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load vocabulary
with open('checkpoints/vocab.json', 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)

vocab = Vocabulary()
vocab.word2idx = vocab_data['word2idx']
vocab.idx2word = vocab_data['idx2word']

# Initialize model
model_config = config['model']

def create_model():
    return TransformerModel(
        vocab_size=len(vocab.word2idx),
        embedding_dim=model_config['embedding_dim'],
        num_heads=model_config['num_heads'],
        num_layers=model_config['num_layers'],
        hidden_dim=model_config['hidden_dim'],
        max_seq_length=config['data']['max_seq_length'],
        dropout=model_config['dropout']
    )

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
print("🔍 对比不同Epoch模型的生成效果")
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
    
    print(f"\n📌 {label}: {model_file}")
    print("-"*80)
    
    try:
        model = create_model()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        response = generate_response(model, test_prompt, temperature=0.7, top_k=50, max_length=25)
        print(f"Q: {test_prompt}")
        print(f"A: {response}")
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")

print("\n" + "="*80)
