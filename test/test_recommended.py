#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import json
from pathlib import Path
from models.transformer import TransformerModel
from models.data_utils import Vocabulary

with open('checkpoints/vocab.json') as f:
    vocab_data = json.load(f)

vocab = Vocabulary(vocab_size=8731)
vocab.word2idx = vocab_data['word2idx']
vocab.idx2word = vocab_data['idx2word']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = TransformerModel(
    vocab_size=8731, embedding_dim=512, num_heads=8,
    num_layers=6, hidden_dim=1024, max_seq_length=64, dropout=0.1
)

# Load latest
latest = sorted(Path('checkpoints').glob('model_epoch_*.pt'))[-1]
epoch = latest.name.split('_')[2].split('.')[0]

cp = torch.load(latest, map_location=device)
model.load_state_dict(cp['model_state_dict'])
model.to(device).eval()

questions = [
    'What is machine learning',
    'How does deep learning work',
    'Artificial intelligence is useful for',
    'Python programming is',
    'Neural networks process data by',
    'Data science involves',
]

print('\n' + '='*95)
print(f'  RECOMMENDED PARAMS - Epoch {epoch}/200 (50% Progress)')
print('  Temperature: 0.65 | Top-K: 42')
print('='*95 + '\n')

with torch.no_grad():
    for i, q in enumerate(questions, 1):
        tokens = [vocab.bos_idx] + vocab.encode(q, add_special_tokens=False)
        out = model.generate(tokens, max_length=50, temperature=0.65, top_k=42, device=device)
        resp = vocab.decode(out, skip_special=True)
        print(f'{i}. [{q}]')
        print(f'   -> {resp[:130]}')
        print()

print('='*95)
print(f'  Parameters: Temp=0.65, Top-K=42 (Balanced diversity & coherence)')
print('='*95 + '\n')
