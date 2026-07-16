#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式连续对话系统 - Interactive Continuous Dialogue System
支持多轮对话、对话历史管理和上下文记忆
"""

import torch
import json
import os
import sys
from pathlib import Path

# 注入项目根目录，确保可 import models（脚本位于 tools/，上一级即根）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.transformer import TransformerModel
from models.config_loader import load_vocab, build_model
import yaml

from models.device import get_device

device = get_device()  # 自动适配 CUDA / DirectML(AMD) / CPU

# Load config
with open('configs/pretrain.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Path for persisted dialogue parameters (single source of truth: chat_config.json)
# 与 showcase_optimal_params.py / load_generation_config 默认路径保持一致（仓库根目录）
params_path = 'chat_config.json'

# default generation configuration (may be overwritten by saved settings)
gen_config = {
    'temperature': 0.65,
    'top_k': 40,
    'repetition_penalty': 2.0,
    'min_new_tokens': 10,
    'max_new_tokens': 100,
    'context_rounds': 3
}

# helper functions to load/save parameter file
def load_gen_config():
    global gen_config
    if os.path.exists(params_path):
        try:
            with open(params_path, 'r', encoding='utf-8') as pf:
                saved = json.load(pf)
            gen_config.update(saved)
            print(f"✅ Loaded persisted dialogue params from {params_path}")
        except Exception as e:
            print(f"⚠️  Failed to load persisted params: {e}")

def save_gen_config():
    try:
        with open(params_path, 'w', encoding='utf-8') as pf:
            json.dump(gen_config, pf, indent=2)
        print(f"💾 Generation parameters saved to {params_path}")
    except Exception as e:
        print(f"⚠️  Could not save params: {e}")

# attempt to load any existing parameter overrides
load_gen_config()

# Load vocabulary（复用 config_loader.load_vocab，正确处理 BPE/char 词表）
vocab = load_vocab('checkpoints/vocab.json')

# Initialize model
model = build_model(config, device=device)

# Load checkpoint
try:
    checkpoint = torch.load('checkpoints/final_model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Loaded: final_model.pt (Best model)")
except Exception as e:
    print(f"⚠️  Warning: {e}")
    try:
        checkpoint = torch.load('checkpoints/model_epoch_50.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Loaded: model_epoch_50.pt")
    except Exception:
        print("❌ Error: No model checkpoint found!")
        sys.exit(1)

model = model.to(device)
model.eval()

print("\n" + "="*70)
print("🤖 连续对话系统 v2.0 - Continuous Dialogue System")
print("="*70)
print("\n📝 命令说明:")
print("  • 直接输入消息进行对话")
print("  • 'reset' / 'r':    重置对话历史")
print("  • 'history' / 'h':  显示完整对话历史")
print("  • 'config' / 'c':   调整生成参数")
print("  • 'exit' / 'q':     退出程序\n")

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

def generate_response(user_input, context=""):
    """Generate model response with configuration"""
    # Build input
    if context:
        full_input = context + " " + user_input
    else:
        full_input = user_input
    
    # Encode
    tokens = vocab.encode(full_input, add_special_tokens=False)
    tokens = [vocab.bos_idx] + tokens
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            tokens,
            max_length=gen_config['max_new_tokens'],
            temperature=gen_config['temperature'],
            top_k=gen_config['top_k'],
            device=device,
            repetition_penalty=gen_config['repetition_penalty']
        )
    
    # Decode
    response = vocab.decode(output_ids, skip_special=True)
    
    # Remove input from response
    input_text = vocab.decode(tokens, skip_special=True)
    if response.startswith(input_text):
        response = response[len(input_text):].strip()
    
    # Deduplicate consecutive words
    response = deduplicate_response(response)
    
    # If still empty, try alternative with reduced input context
    if not response:
        # Try with shorter input
        short_tokens = tokens[-min(4, len(tokens)):]
        with torch.no_grad():
            output_ids = model.generate(
                short_tokens,
                max_length=gen_config['max_new_tokens'],
                temperature=gen_config['temperature'],
                top_k=gen_config['top_k'],
                device=device,
                repetition_penalty=gen_config['repetition_penalty']
            )
        response = vocab.decode(output_ids, skip_special=True)
        short_input = vocab.decode(short_tokens, skip_special=True)
        if response.startswith(short_input):
            response = response[len(short_input):].strip()
        response = deduplicate_response(response)
    
    return response if response else "..."

def show_config():
    """Display generation configuration"""
    print("\n" + "-"*70)
    print("⚙️  生成参数配置 Generation Config:")
    print("-"*70)
    for key, value in gen_config.items():
        if isinstance(value, float):
            print(f"  • {key:20s}: {value:.2f}")
        else:
            print(f"  • {key:20s}: {value}")
    print("-"*70)

turn = 0

while True:
    try:
        user_input = input("\n👤 You: ").strip()
    except EOFError:
        break
    
    if not user_input:
        continue
    
    # Handle special commands
    if user_input.lower() in ['exit', 'q']:
        print("\n👋 Goodbye! See you next time!")
        break
    
    if user_input.lower() in ['reset', 'r']:
        conversation = []
        turn = 0
        print("🔄 对话历史已重置 ✓")
        continue
    
    if user_input.lower() in ['history', 'h']:
        if conversation:
            print("\n" + "="*70)
            print("📋 对话历史 Conversation History:")
            print("="*70)
            for i, (user_msg, bot_msg) in enumerate(conversation, 1):
                print(f"\n[Turn {i}]")
                print(f"  You: {user_msg}")
                print(f"  Bot: {bot_msg}")
            print("\n" + "="*70)
        else:
            print("\n📋 对话历史为空 (Empty history)")
        continue
    
    if user_input.lower() in ['config', 'c']:
        show_config()
        print("\n⚙️  调整参数 (输入格式: 参数名=值，如: temperature=0.8)")
        try:
            param_input = input("→ ").strip()
            if param_input:
                key, value = param_input.split('=')
                key = key.strip()
                value = value.strip()
                
                if key in gen_config:
                    if isinstance(gen_config[key], float):
                        gen_config[key] = float(value)
                    elif isinstance(gen_config[key], int):
                        gen_config[key] = int(value)
                    else:
                        gen_config[key] = value
                    print(f"✅ {key} 已更新为 {value}")
                    # persist changes
                    save_gen_config()
                else:
                    print(f"❌ 未知参数: {key}")
        except Exception as e:
            print(f"❌ 参数错误: {e}")
        continue
    
    # Generate response
    turn += 1
    context = format_context(conversation)
    response = generate_response(user_input, context)
    
    conversation.append((user_input, response))
    
    # Display response
    print(f"🤖 Bot: {response}")
    context_len = len(context.split()) if context else 0
    print(f"    └─ Turn {turn} | Context: {context_len} words | Model: final_model.pt")

# Save dialogue parameters before exit
save_gen_config()

print("\n" + "="*70)
print("感谢使用！Thank you!")
print("="*70)
