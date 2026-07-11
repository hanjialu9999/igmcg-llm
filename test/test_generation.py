#!/usr/bin/env python3

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
with open('checkpoints/vocab.json', 'r') as f:
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

# Load checkpoint
checkpoint = torch.load('checkpoints/final_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print("="*50)
print("Generation Tests with Optimized Model")
print("="*50)

prompts = [
    "Machine learning is",
    "The future of technology",
    "Python is a powerful",
    "Artificial intelligence can",
    "Learning is the key"
]

with torch.no_grad():
    for prompt in prompts:
        tokens = vocab.encode(prompt)
        if tokens:
            output_ids = model.generate(
                tokens,
                max_length=15,
                temperature=0.8,
                top_k=40,
                device=device,
                repetition_penalty=1.4
            )
            generated_text = vocab.decode(output_ids)
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generated_text}")
            print("-" * 50)

print("\nDone!")

# Pause so the user can read output before the script exits
try:
    input('\nPress Enter to exit...')
except Exception:
    pass
