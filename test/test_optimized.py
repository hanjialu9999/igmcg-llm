#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试优化后的模型性能
"""

import torch
import json
import yaml
from pathlib import Path
from models.transformer import TransformerModel
from models.data_utils import Vocabulary

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load config
with open('configs/pretrain.yaml') as f:
    cfg = yaml.safe_load(f)

# Load vocab
with open('checkpoints/vocab.json') as f:
    vdata = json.load(f)

vocab = Vocabulary(vocab_size=cfg['model'].get('vocab_size', 10000))
vocab.word2idx = vdata['word2idx']
vocab.idx2word = vdata['idx2word']

# Load model
model = TransformerModel(
    vocab_size=len(vocab.word2idx), 
    embedding_dim=cfg['model']['embedding_dim'],
    num_heads=cfg['model']['num_heads'],
    num_layers=cfg['model']['num_layers'],
    hidden_dim=cfg['model']['hidden_dim'],
    max_seq_length=cfg['data']['max_seq_length'],
    dropout=cfg['model']['dropout']
)

# Find latest checkpoint
checkpoint_dir = Path('checkpoints')
epoch_files = sorted(checkpoint_dir.glob('model_epoch_*.pt'))

if not epoch_files:
    print("No checkpoints found!")
    exit(1)

latest = epoch_files[-1]
epoch_num = latest.name.split('_')[2].split('.')[0]

cp = torch.load(latest, map_location=device)
model_vocab_size = cp.get('vocab_size', len(vocab.word2idx))

# Load model with correct vocab size
model = TransformerModel(
    vocab_size=model_vocab_size, 
    embedding_dim=cfg['model']['embedding_dim'],
    num_heads=cfg['model']['num_heads'],
    num_layers=cfg['model']['num_layers'],
    hidden_dim=cfg['model']['hidden_dim'],
    max_seq_length=cfg['data']['max_seq_length'],
    dropout=cfg['model']['dropout']
)

model.load_state_dict(cp['model_state_dict'])
model = model.to(device)
model.eval()

print(f"\n{'='*70}")
print(f"Testing Model from Epoch {epoch_num}")
print(f"{'='*70}\n")

def test(prompt):
    tokens = vocab.encode(prompt, add_special_tokens=False)
    tokens = [vocab.bos_idx] + tokens
    
    with torch.no_grad():
        out = model.generate(tokens, max_length=35, temperature=0.8, top_k=50, device=device)
    
    response = vocab.decode(out, skip_special=True)
    print(f"Q: {prompt}")
    print(f"A: {response[:100]}\n")

# Test prompts
prompts = [
    'What is machine learning',
    'How to learn programming',
    'Artificial intelligence is',
    'Python is useful for'
]

for p in prompts:
    test(p)

print(f"{'='*70}\n")
