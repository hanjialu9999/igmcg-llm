#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试当前模型的生成效果
"""

import torch
import json
from models.transformer import TransformerModel
from models.data_utils import Vocabulary
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load config
with open('configs/pretrain.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load vocabulary
with open('checkpoints/vocab.json', 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)

vocab = Vocabulary()
vocab.word2idx = vocab_data['word2idx']
vocab.idx2word = vocab_data['idx2word']

# Initialize model
model_config = config['model']
model = TransformerModel(
    vocab_size=len(vocab.word2idx),
    embedding_dim=model_config['embedding_dim'],
    num_heads=model_config['num_heads'],
    num_layers=model_config['num_layers'],
    hidden_dim=model_config['hidden_dim'],
    max_seq_length=config['data']['max_seq_length'],
    dropout=model_config['dropout']
)

# Load latest checkpoint
checkpoint = torch.load('checkpoints/model_epoch_79.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

def deduplicate_response(text):
    """Remove consecutive duplicate words"""
    words = text.split()
    if not words:
        return text
    
    deduped = [words[0]]
    for word in words[1:]:
        if word.lower() != deduped[-1].lower():
            deduped.append(word)
    
    return " ".join(deduped)

def generate_response(user_input, temperature=0.7, top_k=50, repetition_penalty=2.0, max_length=20):
    """Generate model response"""
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
    
    return deduplicate_response(response) if response else "..."

print("\n" + "="*70)
print("🧪 清洁数据训练效果 (Epoch 79)")
print("="*70 + "\n")

test_cases = [
    "What is artificial intelligence",
    "How does machine learning work",
    "What are neural networks",
    "Can you help me learn",
    "Tell me about Python programming"
]

for prompt in test_cases:
    response = generate_response(prompt, temperature=0.7, top_k=50, max_length=25)
    print(f"Q: {prompt}")
    print(f"A: {response}")
    print()

print("="*70)
