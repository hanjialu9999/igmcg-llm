#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细的生成效果测试脚本
"""

import torch
import json
from pathlib import Path
from models.transformer import TransformerModel
from models.data_utils import load_vocab, tokenize, detokenize

def load_model_and_vocab():
    """加载模型和词汇表"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载词汇表
    vocab = load_vocab('./checkpoints/vocab.json')
    
    # 初始化模型
    model = TransformerModel(
        vocab_size=len(vocab),
        embedding_dim=256,
        num_heads=8,
        num_layers=4,
        hidden_dim=512,
        max_seq_length=32,
        dropout=0.1
    )
    
    # 加载最好的模型
    checkpoint_path = './checkpoints/final_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    print(f"✅ 模型加载完成: {checkpoint_path}")
    print(f"   设备: {device}")
    print(f"   词汇表大小: {len(vocab)}")
    
    return model, vocab, device

def generate_response(model, vocab, prompt, device, temperature=0.7, top_k=30, max_length=30):
    """生成回复"""
    tokens = tokenize(prompt, vocab)
    tokens = torch.tensor(tokens[:31], dtype=torch.long).unsqueeze(0).to(device)
    
    generated = list(tokens[0].cpu().numpy())
    
    with torch.no_grad():
        for _ in range(max_length - len(generated)):
            logits = model(tokens)
            logits = logits[0, -1, :] / temperature
            
            # top_k采样
            top_k_vals, top_k_inds = torch.topk(logits, min(top_k, logits.shape[0]))
            logits_top_k = torch.full_like(logits, float('-inf'))
            logits_top_k[top_k_inds] = top_k_vals
            
            probabilities = torch.softmax(logits_top_k, dim=0)
            next_token = torch.multinomial(probabilities, 1).item()
            
            if next_token == vocab.get('<eos>', len(vocab)-1):
                break
            
            generated.append(next_token)
            tokens = torch.tensor([generated], dtype=torch.long).to(device)
    
    # 解码
    word_list = []
    for token_id in generated:
        if token_id < len(vocab):
            for word, idx in vocab.items():
                if idx == token_id:
                    word_list.append(word)
                    break
    
    return ' '.join(word_list)

def main():
    print("\n" + "="*70)
    print("🧪 模型生成效果详细测试")
    print("="*70 + "\n")
    
    model, vocab, device = load_model_and_vocab()
    
    # 测试问题集
    test_prompts = [
        "机器学习是",
        "今天天气很",
        "我想告诉你",
        "人工智能可以",
        "学习编程的第一步是",
        "这个问题很",
        "最好的方法是",
        "让我解释一下",
        "根据研究显示",
        "在未来，科技将"
    ]
    
    print("\n📝 测试生成效果 (Temperature=0.7, Top-K=30):\n")
    print("-" * 70)
    
    for prompt in test_prompts:
        output = generate_response(model, vocab, prompt, device, 
                                   temperature=0.7, top_k=30)
        # 截断显示
        output_short = output[:60] + "..." if len(output) > 60 else output
        print(f"输入: {prompt}")
        print(f"输出: {output_short}")
        print("-" * 70)
    
    # 多轮对话测试
    print("\n\n💬 多轮对话测试（检查句子连贯性）:\n")
    print("="*70)
    
    conversation_context = ""
    questions = [
        "什么是机器学习",
        "它有什么应用",
        "我该怎样学习它"
    ]
    
    for i, question in enumerate(questions, 1):
        prompt = conversation_context + question if conversation_context else question
        response = generate_response(model, vocab, prompt, device, 
                                    temperature=0.7, top_k=30)
        response_short = response[:80] + "..." if len(response) > 80 else response
        
        print(f"\n轮次 {i}:")
        print(f"  Q: {question}")
        print(f"  A: {response_short}")
        
        conversation_context = prompt + " " + response + " "
    
    print("\n" + "="*70)
    print("✅ 测试完成")
    print("="*70)

if __name__ == '__main__':
    main()
