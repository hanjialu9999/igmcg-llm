"""
诊断脚本 - 检查模型输出
"""

import torch
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer import TransformerModel
from models.data_utils import Vocabulary
from models.device import get_device

# Load model and vocab
device = get_device()
print(f"Device: {device}")

model_path = 'checkpoints/final_model.pt'
vocab_path = 'checkpoints/vocab.json'

# Load vocab
with open(vocab_path, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)

vocab = Vocabulary()
vocab.word2idx = vocab_data['word2idx']
vocab.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}

# Load model
checkpoint = torch.load(model_path, map_location=device)
model_config = checkpoint.get('config', {
    'vocab_size': checkpoint['vocab_size'],
    'embedding_dim': 128,
    'num_heads': 4,
    'num_layers': 2,
    'hidden_dim': 256,
    'max_seq_length': 32,
    'dropout': 0.1
})

model = TransformerModel(
    vocab_size=checkpoint['vocab_size'],
    embedding_dim=model_config.get('embedding_dim', 128),
    num_heads=model_config.get('num_heads', 4),
    num_layers=model_config.get('num_layers', 2),
    hidden_dim=model_config.get('hidden_dim', 256),
    max_seq_length=model_config.get('max_seq_length', 32),
    dropout=model_config.get('dropout', 0.1)
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Model loaded successfully!")
print(f"Model vocab size: {checkpoint['vocab_size']}")
print(f"Vocab size: {len(vocab)}")
print()

# Test
prompt = "Hello world"
tokens = vocab.encode(prompt)
print(f"Prompt: {prompt}")
print(f"Token IDs: {tokens}")
print(f"Decoded back: {vocab.decode(tokens)}")
print()

# Manual generation step
print("Manual generation test:")
print("="*50)

with torch.no_grad():
    # First step
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    print(f"Input shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids}")
    
    logits = model(input_ids)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits min/max: {logits.min():.4f} / {logits.max():.4f}")
    
    # Get next token
    next_logits = logits[0, -1, :]
    probs = torch.softmax(next_logits, dim=-1)
    
    print(f"\nTop 10 prob tokens:")
    top_probs, top_indices = torch.topk(probs, 10)
    for prob, idx in zip(top_probs, top_indices):
        word = vocab.idx2word.get(idx.item(), '<unk>')
        print(f"  {word:<20} {prob:.4f}")
    
    # Sample next token
    next_token = torch.multinomial(probs, 1).item()
    print(f"\nSampled next token: {next_token} ({vocab.idx2word.get(next_token, '<unk>')})")
    print()

print("="*50)
print("\nConclusion:")
print("If the model generates mostly <pad> or <eos> tokens, it hasn't learned well.")
print("If it generates relevant words, the model is learning correctly.")
