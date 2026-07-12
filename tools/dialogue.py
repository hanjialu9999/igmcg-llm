#!/usr/bin/env python3
"""
连续对话脚本 - 支持多轮对话交互
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
try:
    checkpoint = torch.load('checkpoints/final_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
except:
    print("Warning: Failed to load final_model.pt, trying model_epoch_20.pt")
    checkpoint = torch.load('checkpoints/model_epoch_20.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

model = model.to(device)
model.eval()

print("="*60)
print("🤖 连续对话系统 - Continuous Dialogue System")
print("="*60)
print("\n命令列表:")
print("  • 输入对话内容继续交互")
print("  • 'reset' 或 'r': 重置对话历史")
print("  • 'history' 或 'h': 显示对话历史")
print("  • 'exit' 或 'q': 退出程序")
print("\n" + "="*60 + "\n")

# Conversation history
conversation = []
max_history_tokens = 28  # 留一些空间给新提示

def format_context(history):
    """格式化对话历史作为模型输入上下文"""
    if not history:
        return ""
    
    # 合并最后的对话，限制长度
    context_parts = []
    total_tokens = 0
    
    # 反向遍历历史，添加最近的消息
    for user_msg, bot_msg in reversed(history):
        tokens = len(vocab.tokenize(user_msg)) + len(vocab.tokenize(bot_msg)) + 2
        if total_tokens + tokens > max_history_tokens:
            break
        context_parts.insert(0, f"{user_msg} {bot_msg}")
        total_tokens += tokens
    
    return " ".join(context_parts) if context_parts else ""

def generate_response(user_input, context=""):
    """生成模型响应"""
    if context:
        full_input = context + " " + user_input
    else:
        full_input = user_input
    
    # 对输入进行编码
    tokens = vocab.encode(full_input, add_special_tokens=False)
    tokens = [vocab.bos_idx] + tokens
    
    # 生成响应
    with torch.no_grad():
        output_ids = model.generate(
            tokens,
            max_length=10,
            temperature=0.75,
            top_k=35,
            device=device,
            repetition_penalty=1.5
        )
    
    # 解码响应
    response = vocab.decode(output_ids, skip_special=True)
    
    # 清理响应：移除输入部分
    input_text = vocab.decode(tokens, skip_special=True)
    if response.startswith(input_text):
        response = response[len(input_text):].strip()
    
    return response if response else "I understand."

turn = 0

while True:
    # 获取用户输入
    user_input = input("\n👤 You: ").strip()
    
    if not user_input:
        continue
    
    # 处理特殊命令
    if user_input.lower() in ['exit', 'q']:
        print("\n👋 拜拜！Goodbye!")
        break
    
    if user_input.lower() in ['reset', 'r']:
        conversation = []
        print("\n🔄 对话历史已重置 (History reset)")
        continue
    
    if user_input.lower() in ['history', 'h']:
        if conversation:
            print("\n📋 对话历史 (Conversation History):")
            print("-" * 60)
            for i, (user_msg, bot_msg) in enumerate(conversation, 1):
                print(f"Turn {i}:")
                print(f"  You: {user_msg}")
                print(f"  Bot: {bot_msg}")
            print("-" * 60)
        else:
            print("\n📋 对话历史为空 (Empty history)")
        continue
    
    # 生成响应
    turn += 1
    context = format_context(conversation)
    response = generate_response(user_input, context)
    
    # 添加到历史
    conversation.append((user_input, response))
    
    # 显示响应
    print(f"🤖 Bot: {response}")
    print(f"   [Turn {turn}, Context length: {len(context.split()) if context else 0} words]")

print("\n" + "="*60)
print("Thank you for using the dialogue system!")
print("="*60)
