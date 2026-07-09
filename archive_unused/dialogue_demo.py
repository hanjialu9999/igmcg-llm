#!/usr/bin/env python3
"""
连续对话演示 - 演示多轮对话功能
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
try:
    checkpoint = torch.load('checkpoints/final_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
except:
    try:
        checkpoint = torch.load('checkpoints/model_epoch_20.pt', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        print("Error: No model checkpoint found!")
        exit(1)

model = model.to(device)
model.eval()

print("="*65)
print("🤖 连续对话演示 Demo - Continuous Dialogue System")
print("="*65)

# Conversation history
conversation = []
max_history_tokens = 28

def format_context(history):
    """Format conversation history as context"""
    if not history:
        return ""
    
    context_parts = []
    total_tokens = 0
    
    for user_msg, bot_msg in reversed(history):
        tokens = len(vocab.tokenize(user_msg)) + len(vocab.tokenize(bot_msg)) + 2
        if total_tokens + tokens > max_history_tokens:
            break
        context_parts.insert(0, f"{user_msg} {bot_msg}")
        total_tokens += tokens
    
    return " ".join(context_parts) if context_parts else ""

def generate_response(user_input, context=""):
    """Generate model response"""
    if context:
        full_input = context + " " + user_input
    else:
        full_input = user_input
    
    tokens = vocab.encode(full_input, add_special_tokens=False)
    tokens = [vocab.bos_idx] + tokens
    
    with torch.no_grad():
        output_ids = model.generate(
            tokens,
            max_length=10,
            temperature=0.75,
            top_k=35,
            device=device,
            repetition_penalty=1.5
        )
    
    response = vocab.decode(output_ids, skip_special=True)
    
    input_text = vocab.decode(tokens, skip_special=True)
    if response.startswith(input_text):
        response = response[len(input_text):].strip()
    
    # Fallback: try short input if empty
    if not response:
        short_tokens = tokens[-min(4, len(tokens)):]
        with torch.no_grad():
            output_ids = model.generate(
                short_tokens,
                max_length=10,
                temperature=0.75,
                top_k=35,
                device=device,
                repetition_penalty=1.5
            )
        response = vocab.decode(output_ids, skip_special=True)
        short_input = vocab.decode(short_tokens, skip_special=True)
        if response.startswith(short_input):
            response = response[len(short_input):].strip()
    
    return response if response else "..."

# Demo conversation
demo_prompts = [
    "What is machine learning?",
    "Tell me more about it",
    "How can I learn programming?",
    "What about Python?",
    "Is artificial intelligence important?"
]

print("\n演示对话 Demo Dialogue\n")
print("-" * 65)

for turn, user_input in enumerate(demo_prompts, 1):
    # Get context
    context = format_context(conversation)
    
    # Generate response
    response = generate_response(user_input, context)
    
    # Add to history
    conversation.append((user_input, response))
    
    # Display
    print(f"\n[ Turn {turn} ]")
    print(f"👤 You:  {user_input}")
    print(f"🤖 Bot:  {response}")
    context_len = len(context.split()) if context else 0
    print(f"   └─ 上下文长度 Context: {context_len} words")

print("\n" + "="*65)
print("📋 完整对话历史 Full Conversation History:")
print("="*65)

for i, (user_msg, bot_msg) in enumerate(conversation, 1):
    print(f"\nTurn {i}:")
    print(f"  User: {user_msg}")
    print(f"  Bot:  {bot_msg}")

print("\n" + "="*65)
print(f"✅ 演示完成！总共 {len(conversation)} 轮对话")
print("="*65)
