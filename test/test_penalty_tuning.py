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

cp = torch.load('checkpoints/final_model.pt', map_location=device)
model.load_state_dict(cp['model_state_dict'])
model.to(device).eval()

# Test questions
test_questions = [
    'What is machine learning',
    'How does deep learning work',
    'Artificial intelligence can help',
    'Python programming is used for',
    'Neural networks learn through',
]

# Fine-grained penalty testing: 1.8 to 2.2
penalties = [1.8, 1.9, 2.0, 2.1, 2.2]

print('\n' + '='*105)
print('  FINE-GRAINED REPETITION PENALTY TUNING (1.8 - 2.2)')
print('  Finding optimal balance between diversity and coherence')
print('='*105)

for penalty in penalties:
    print('\n' + '-'*105)
    print(f'  PENALTY = {penalty}')
    print('-'*105 + '\n')

    with torch.no_grad():
        for i, q in enumerate(test_questions, 1):
            tokens = [vocab.bos_idx] + vocab.encode(q, add_special_tokens=False)
            out = model.generate(
                tokens, 
                max_length=50, 
                temperature=0.65, 
                top_k=42, 
                device=device,
                repetition_penalty=penalty
            )
            resp = vocab.decode(out, skip_special=True)
            
            print(f'  {i}. [{q}]')
            print(f'     -> {resp[:128]}')
            print()

print('='*105)
print('  ANALYSIS: Look for penalty with best balance of repetition-free + coherent output')
print('='*105 + '\n')
