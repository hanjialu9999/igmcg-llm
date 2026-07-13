#!/usr/bin/env python3
"""
快速体验脚本 - 快速查看对话效果（配置可调）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from models.transformer import TransformerModel
from models.data_utils import Vocabulary
from models.device import get_device
import yaml

device = get_device()

# Load config
with open('configs/pretrain.yaml', 'r', encoding='utf-8') as f:
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
checkpoint = torch.load('checkpoints/final_model.pt', map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

def deduplicate_response(text):
    """Remove consecutive duplicate words"""
    words = text.split()
    if not words:
        return text
    
    deduped = [words[0]]
    for word in words[1:]:
        if word.lower() != deduped[-1].lower():
            deduped.append(word)
    
    return " ".join(deduped)

def generate_response(user_input, temperature=0.7, top_k=30, repetition_penalty=2.0, max_length=10):
    """Generate model response"""
    tokens = vocab.encode(user_input, add_special_tokens=False)
    tokens = [vocab.bos_idx] + tokens
    
    with torch.no_grad():
        output_ids = model.generate(
            tokens,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            device=device,
            repetition_penalty=repetition_penalty
        )
    
    response = vocab.decode(output_ids, skip_special=True)
    input_text = vocab.decode(tokens, skip_special=True)
    
    if response.startswith(input_text):
        response = response[len(input_text):].strip()
    
    return deduplicate_response(response) if response else "..."

print("\n" + "="*70)
print("🚀 AI 对话系统 - 快速演示")
print("="*70)

test_inputs = [
    "Machine learning is",
    "Python is powerful",
    "Today I learned something",
    "The future of technology",
    "Innovation creates",
]

print("\n📋 使用默认参数 (temperature=0.7, top_k=30, repetition_penalty=2.0):\n")
for prompt in test_inputs:
    response = generate_response(prompt)
    print(f"📝 Input:  {prompt}")
    print(f"💬 Output: {response}\n")

print("\n" + "="*70)
print("🎚️  不同参数的效果对比")
print("="*70)

test_prompt = "Success is achieved through"

configs = [
    ("保守模式 (温度低，重复惩罚高)", 0.5, 20, 2.5),
    ("平衡模式 (默认参数)", 0.7, 30, 2.0),
    ("创意模式 (温度高，重复惩罚低)", 0.9, 50, 1.3),
]

for config_name, temp, topk, rep_pen in configs:
    response = generate_response(test_prompt, temperature=temp, top_k=topk, repetition_penalty=rep_pen)
    print(f"\n{config_name}:")
    print(f"  Input:  {test_prompt}")
    print(f"  Output: {response}")

print("\n" + "="*70)
print("✅ 现在你可以运行以下命令进行交互式对话：")
print("="*70)
print("\n  python dialogue_interactive.py")
print("\n这样可以：")
print("  ✓ 进行多轮对话（完整上下文记忆）")
print("  ✓ 实时调整生成参数")
print("  ✓ 查看对话历史")
print("  ✓ 重置对话状态")
print("\n" + "="*70 + "\n")
