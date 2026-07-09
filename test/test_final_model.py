#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import json
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

# Load final trained model
cp = torch.load('checkpoints/final_model.pt', map_location=device)
model.load_state_dict(cp['model_state_dict'])
model.to(device).eval()

test_questions = [
    'What is machine learning',
    'How does deep learning work',
    'Artificial intelligence can help',
    'Python programming is used for',
    'Neural networks learn through',
    'Data science is important because',
]

print('\n' + '='*100)
print('  FINAL MODEL - Full Training Complete (Epoch 200) ')
print('  Parameters: Temp=0.65, Top-K=42')
print('='*100 + '\n')

with torch.no_grad():
    for i, q in enumerate(test_questions, 1):
        tokens = [vocab.bos_idx] + vocab.encode(q, add_special_tokens=False)
        out = model.generate(tokens, max_length=50, temperature=0.65, top_k=42, device=device)
        resp = vocab.decode(out, skip_special=True)
        
        print(f'{i}. INPUT:  [{q}]')
        print(f'   OUTPUT: {resp[:130]}')
        print()

print('='*100)
print('  TRAINING SUMMARY')
print('  - Architecture: 6 Transformer Layers | 512D Embeddings | 8 Attention Heads')
print('  - Parameters: 21.5M total')
print('  - Training: 200 Epochs | Batch Size: 128 | Learning Rate: 0.0005')
print('  - Data: 5974 Q&A pairs | Vocab Size: 8731')
print('  - Optimizations: Mixed Precision, Gradient Checkpointing, Warmup, Early Stopping')
print('='*100 + '\n')
