#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import json
from pathlib import Path
from models.transformer import TransformerModel
from models.data_utils import Vocabulary

# Load vocab
with open('checkpoints/vocab.json') as f:
    vocab_data = json.load(f)

vocab = Vocabulary(vocab_size=8731)
vocab.word2idx = vocab_data['word2idx']
vocab.idx2word = vocab_data['idx2word']

# Load latest model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epoch_files = sorted(Path('checkpoints').glob('model_epoch_*.pt'))

if not epoch_files:
    print("No model found!")
    exit(1)

latest = epoch_files[-1]
epoch_num = latest.name.split('_')[2].split('.')[0]

model = TransformerModel(
    vocab_size=8731, embedding_dim=512, num_heads=8,
    num_layers=6, hidden_dim=1024, max_seq_length=64, dropout=0.1
)

cp = torch.load(latest, map_location=device)
model.load_state_dict(cp['model_state_dict'])
model.to(device).eval()

# Test prompts
prompts = [
    'What is artificial intelligence',
    'How does machine learning work',
    'Python programming is useful for',
    'Neural networks are',
    'The future of AI',
]

print("\n" + "="*70)
print(f"[Batch 128] Epoch {epoch_num} - Generation Quality Test")
print("="*70 + "\n")

for prompt in prompts:
    tokens = vocab.encode(prompt, add_special_tokens=False)
    tokens = [vocab.bos_idx] + tokens
    
    with torch.no_grad():
        out = model.generate(tokens, max_length=35, temperature=0.8, top_k=50, device=device)
    
    response = vocab.decode(out, skip_special=True)
    print(f"Q: {prompt}")
    print(f"A: {response[:110]}\n")

print("="*70)
print(f"Model: Epoch {epoch_num} (Batch 128)")
print("="*70 + "\n")
