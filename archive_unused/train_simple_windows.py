#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版训练脚本 - 忽略多进程问题，在Windows上直接运行
Simplified training script for Windows
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.transformer import TransformerModel
from models.data_utils import load_data, create_dataloader

def main():
    print("\n" + "="*80)
    print("  🚀 简化版GPU训练脚本 (Windows版)")
    print("="*80 + "\n")
    
    # 配置
    config = {
        'data_file': 'data/train_data_final.txt',
        'vocab_size': 10000,
        'batch_size': 128,
        'epochs': 200,
        'learning_rate': 0.0005,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'embedding_dim': 512,
        'num_heads': 8,
        'num_layers': 6,
        'hidden_dim': 1024,
        'dropout': 0.1,
        'max_seq_length': 64
    }
    
    # 检查GPU
    device = torch.device(config['device'])
    print(f"[1] 设备: {device}")
    if device.type == 'cuda':
        print(f"    GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"    显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 加载数据
    print(f"\n[2] 加载数据...")
    dataset, vocab = load_data(
        config['data_file'],
        vocab_size=config['vocab_size'],
        max_seq_length=config['max_seq_length']
    )
    print(f"    数据集: {len(dataset)} 样本")
    print(f"    词汇表: {len(vocab)} 词汇")
    
    # 创建DataLoader (不使用num_workers)
    print(f"\n[3] 创建数据加载器...")
    dataloader = create_dataloader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0  # Windows上必须为0
    )
    print(f"    批次数量: {len(dataloader)}")
    
    # 创建模型
    print(f"\n[4] 创建模型...")
    model = TransformerModel(
        vocab_size=len(vocab),
        embedding_dim=config['embedding_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        hidden_dim=config['hidden_dim'],
        max_seq_length=config['max_seq_length'],
        dropout=config['dropout']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    参数数量: {total_params:,}")
    
    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    # 混合精度训练
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    print(f"\n[5] 开始训练...")
    print(f"    批大小: {config['batch_size']}")
    print(f"    学习率: {config['learning_rate']}")
    print(f"    混合精度: {device.type == 'cuda'}")
    print(f"\n" + "="*80 + "\n")
    
    # 训练循环
    for epoch in range(1, min(config['epochs'] + 1, 6)):  # 仅训练5个epoch用于测试
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            if device.type == 'cuda' and scaler:
                with torch.amp.autocast('cuda'):
                    logits = model(input_ids)
                    logits = logits.view(-1, logits.size(-1))
                    target_ids_flat = target_ids.view(-1)
                    loss = criterion(logits, target_ids_flat)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids)
                logits = logits.view(-1, logits.size(-1))
                target_ids_flat = target_ids.view(-1)
                loss = criterion(logits, target_ids_flat)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | Loss: {avg_loss:.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"\n✅ Epoch {epoch}/{min(config['epochs'], 5)} 完成 | 平均Loss: {avg_loss:.4f}\n")
    
    # 保存模型
    print("="*80)
    print("  ✅ 训练成功！模型已保存")
    print("="*80)
    
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': len(vocab),
    }, 'checkpoints/final_model.pt')
    print("\n✅ 模型位置: checkpoints/final_model.pt")

if __name__ == '__main__':
    main()
