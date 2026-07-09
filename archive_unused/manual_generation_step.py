#!/usr/bin/env python3

import torch
import json
from models.transformer import TransformerModel
from models.data_utils import Vocabulary
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load config
with open('config/config.yaml', 'r') as f:
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

# Manually run one generation step
prompt = "Machine learning"
tokens = vocab.encode(prompt, add_special_tokens=False)
tokens = [vocab.bos_idx] + tokens  # [2, 458, 130]

print(f"Prompt: {prompt}")
print(f"Tokens: {tokens}")
print(f"Decoded back: {vocab.decode(tokens)}")

input_tensor = torch.tensor([tokens], device=device, dtype=torch.long)

with torch.no_grad():
    logits = model.forward(input_tensor)
    print(f"\nLogits shape: {logits.shape}")
    
    # Get last position logits
    next_token_logits = logits[0, -1, :]
    print(f"Last position logits - min: {next_token_logits.min():.2f}, max: {next_token_logits.max():.2f}")
    
    # Top 10 candidates
    top_10_vals, top_10_idx = torch.topk(next_token_logits, 10)
    print(f"\nTop 10 next tokens (NO FILTERING):")
    for i, (logit, idx) in enumerate(zip(top_10_vals, top_10_idx)):
        token_str = vocab.idx2word.get(str(idx.item()), f'<idx:{idx.item()}'>)
        print(f"  {i+1}. '{token_str}' (logit={logit.item():.2f})")
    
    # Apply softmax
    probs = torch.softmax(next_token_logits, dim=-1)
    print(f"\nProbs - min: {probs.min():.6f}, max: {probs.max():.6f}, sum: {probs.sum():.4f}")
    
    # Top 5 by probability
    top_5_probs, top_5_idx_prob = torch.topk(probs, 5)
    print(f"\nTop 5 by probability:")
    for i, (prob, idx) in enumerate(zip(top_5_probs, top_5_idx_prob)):
        token_str = vocab.idx2word.get(str(idx.item()), f'<idx:{idx.item()}'>)
        print(f"  {i+1}. '{token_str}' (prob={prob.item():.6f})")
    
    # Sample
    sampled = torch.multinomial(probs, num_samples=1).item()
    sampled_token_str = vocab.idx2word.get(str(sampled), '<unk>')
    print(f"\nSampled token: '{sampled_token_str}' (idx={sampled})")
