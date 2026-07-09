#!/usr/bin/env python3
"""
调试脚本 - 深度分析为什么总是返回 "I understand"
"""

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

# Load checkpoint
checkpoint = torch.load('checkpoints/final_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print("="*70)
print("🔍 调试：为什么返回 'I understand'")
print("="*70)

test_prompts = [
    "What is machine learning?",
    "Tell me more",
    "How can I learn programming?"
]

for prompt in test_prompts:
    print(f"\n{'-'*70}")
    print(f"📝 Prompt: {prompt}")
    print(f"{'-'*70}")
    
    # Encode without context
    tokens = vocab.encode(prompt, add_special_tokens=False)
    tokens = [vocab.bos_idx] + tokens
    
    print(f"1️⃣ Encoded tokens: {tokens}")
    input_text = vocab.decode(tokens, skip_special=True)
    print(f"   Decoded back: {input_text}")
    
    # Forward pass
    input_tensor = torch.tensor([tokens], device=device, dtype=torch.long)
    
    with torch.no_grad():
        logits = model.forward(input_tensor)
        print(f"\n2️⃣ Model forward pass:")
        print(f"   Logits shape: {logits.shape}")
        
        # Check last position
        last_logits = logits[0, -1, :]
        print(f"   Last position logits stat:")
        print(f"     Min: {last_logits.min().item():.4f}")
        print(f"     Max: {last_logits.max().item():.4f}")
        print(f"     Mean: {last_logits.mean().item():.4f}")
        
        # Top 10 candidates
        top_10_vals, top_10_indices = torch.topk(last_logits, min(10, last_logits.shape[0]))
        print(f"\n   Top 10 candidates:")
        for i, (val, idx) in enumerate(zip(top_10_vals, top_10_indices)):
            word = vocab.idx2word.get(str(idx.item()), f'<UNK:{idx.item()}'>)
            print(f"     {i+1}. '{word}' (logit={val.item():.2f})")
    
    # Now run generation
    print(f"\n3️⃣ Generation with max_length=20:")
    output_ids = model.generate(
        tokens,
        max_length=20,
        temperature=0.8,
        top_k=40,
        device=device,
        repetition_penalty=1.5
    )
    
    print(f"   Raw output IDs: {output_ids}")
    generated = vocab.decode(output_ids, skip_special=True)
    print(f"   Decoded: {generated}")
    
    # Check if input is in output
    if generated.startswith(input_text):
        trimmed = generated[len(input_text):].strip()
        print(f"   After removing input: {trimmed}")
    else:
        print(f"   (Input not found in output)")

print("\n" + "="*70)
print("🔍 分析：检查 'I understand' 出现情况")
print("="*70)

# Check if "I understand" is in vocabulary
understand_idx = vocab.word2idx.get('understand', -1)
i_idx = vocab.word2idx.get('i', -1)

print(f"\n词汇表中的词：")
print(f"  'understand': idx={understand_idx}")
print(f"  'i': idx={i_idx}")

if understand_idx >= 0:
    print(f"\n🔎 检查为什么 'understand' 这么容易被选中...")
    
    # Check what word appears most often in training data
    with open('data/train_data_combined.txt', 'r', encoding='utf-8') as f:
        text = f.read().lower()
        understand_count = text.count(' understand ')
        print(f"   训练数据中 'understand' 出现: {understand_count} 次")
        
        # Count top words
        words = text.split()
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        print(f"\n   训练数据中前15个高频词：")
        for word, count in top_words:
            print(f"     {word}: {count}次")

print("\n" + "="*70)
