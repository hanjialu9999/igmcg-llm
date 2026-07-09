#!/usr/bin/env python3

import torch
import json
from models.transformer import TransformerModel
from models.data_utils import Vocabulary
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load vocabulary
with open('checkpoints/vocab.json', 'r') as f:
    vocab_data = json.load(f)

vocab = Vocabulary()
vocab.word2idx = vocab_data['word2idx']
vocab.idx2word = vocab_data['idx2word']

print(f"Vocab size: {len(vocab.word2idx)}")
print(f"Sample token: 'learning' -> {vocab.word2idx.get('learning', 'NOT FOUND')}")

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
print(f"Checkpoint keys: {checkpoint.keys()}")
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Test: Forward pass only
sample_prompt = "Machine learning"
tokens = vocab.encode(sample_prompt)
print(f"\nPrompt: {sample_prompt}")
print(f"Encoded tokens: {tokens}")

if tokens:
    input_tensor = torch.tensor([tokens], device=device, dtype=torch.long)
    print(f"Input shape: {input_tensor.shape}")
    
    with torch.no_grad():
        logits = model.forward(input_tensor)
        print(f"Logits shape: {logits.shape}")
        
        # Look at last position
        last_logits = logits[0, -1, :]
        print(f"\nLast position logits stats:")
        print(f"  Min: {last_logits.min().item():.4f}")
        print(f"  Max: {last_logits.max().item():.4f}")
        print(f"  Mean: {last_logits.mean().item():.4f}")
        
        # Top 5 tokens
        top_5_vals, top_5_idx = torch.topk(last_logits, 5)
        print(f"\nTop 5 next tokens:")
        for i, (logit, idx) in enumerate(zip(top_5_vals, top_5_idx)):
            token_str = vocab.idx2word.get(str(idx.item()), f"<idx:{idx.item()}>")
            print(f"  {i+1}. '{token_str}' (idx={idx.item()}): logit={logit.item():.4f}")
