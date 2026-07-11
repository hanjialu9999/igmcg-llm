#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的生成效果测试 - 测试句子连贯性和对话上下文理解
"""

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

def deduplicate_response(text):
    """移除连续重复的词"""
    words = text.split()
    if not words:
        return text
    
    deduped = [words[0]]
    for word in words[1:]:
        if word.lower() != deduped[-1].lower():
            deduped.append(word)
    
    return " ".join(deduped)

def generate_response(user_input, temperature=0.7, top_k=50, repetition_penalty=2.0, max_length=25):
    """生成模型回复"""
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

print("\n" + "="*75)
print("🧪 改进的生成效果测试 - 测试句子连贯性和上下文理解能力")
print("="*75)

# ============= 测试1: 单轮问答 =============
print("\n\n【测试1】单轮问答 - 基础生成能力")
print("-"*75)

single_turn_prompts = [
    "What is machine learning",
    "How does neural network work",
    "What are embeddings used for",
    "Explain the transformer architecture",
    "Why is Python popular for AI"
]

for prompt in single_turn_prompts:
    response = generate_response(prompt, temperature=0.7, top_k=50, max_length=25)
    print(f"Q: {prompt}")
    print(f"A: {response[:100]}...")
    print()

# ============= 测试2: 多轮对话 - 测试上下文连贯性 =============
print("\n\n【测试2】多轮对话 - 测试上下文记忆和连贯性")
print("-"*75)

dialogue_context = ""
dialogue_turns = [
    ("What is machine learning", "我在问什么是机器学习"),
    ("Can you give me examples", "追问能不能给我举例"),
    ("How do I start learning it", "追问我该怎样开始学习"),
]

for question, description in dialogue_turns:
    # 构建带有上下文的prompt
    if dialogue_context:
        full_prompt = dialogue_context + " " + question
    else:
        full_prompt = question
    
    response = generate_response(full_prompt, temperature=0.7, top_k=50, max_length=30)
    
    print(f"\n轮次: {description}")
    print(f"Q: {question}")
    print(f"A: {response[:120]}")
    
    # 更新对话上下文（简化：只使用最后一个Q+A）
    dialogue_context = question + " " + response

# ============= 测试3: 参数对比 =============
print("\n\n【测试3】不同参数对生成质量的影响")
print("-"*75)

test_prompt = "Artificial intelligence can"

configs = [
    ("保守模式", {"temperature": 0.5, "top_k": 20, "repetition_penalty": 3.0}),
    ("平衡模式", {"temperature": 0.7, "top_k": 50, "repetition_penalty": 2.0}),
    ("创意模式", {"temperature": 0.9, "top_k": 100, "repetition_penalty": 1.0}),
]

for mode_name, params in configs:
    response = generate_response(test_prompt, max_length=25, **params)
    print(f"\n{mode_name}:")
    print(f"  参数: {params}")
    print(f"  输出: {response[:100]}")

# ============= 测试4: 长序列生成 =============
print("\n\n【测试4】长序列生成 - 测试模型维度句子")
print("-"*75)

long_prompts = [
    "The process of learning a new skill requires",
    "In the field of artificial intelligence and machine learning",
]

for prompt in long_prompts:
    response = generate_response(prompt, temperature=0.7, top_k=50, max_length=40)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")

print("\n\n" + "="*75)
print("✅ 测试完成！")
print("="*75 + "\n")
