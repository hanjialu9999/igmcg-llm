#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU 训练脚本 - Windows 兼容版本
强制 num_workers=0，避免 Windows 多进程问题
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
import yaml
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# 项目根目录
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.transformer import TransformerModel
from models.data_utils import load_data, TextDataset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_gpu.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path='config/config.yaml'):
    """加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_windows_dataloader(dataset, batch_size=16, shuffle=True):
    """为 Windows 创建数据加载器（强制 num_workers=0）"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Windows 上必须为 0
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle
    )


def train_epoch(model, dataloader, optimizer, scaler, device, epoch, max_epochs):
    """训练单个 epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        # 处理字典格式的批次数据
        if isinstance(batch, dict):
            inputs = batch['input_ids'].to(device)
            targets = batch['target_ids'].to(device)
        else:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # 混合精度训练
        with autocast(device_type='cuda', dtype=torch.float16):
            logits = model(inputs)
            loss = nn.CrossEntropyLoss()(logits.view(-1, model.vocab_size), targets.view(-1))
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            logger.info(f"Epoch [{epoch}/{max_epochs}] Batch [{batch_idx+1}/{num_batches}] Loss: {avg_loss:.4f}")
    
    return total_loss / num_batches


def validate(model, dataloader, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            # 处理字典格式的批次数据
            if isinstance(batch, dict):
                inputs = batch['input_ids'].to(device)
                targets = batch['target_ids'].to(device)
            else:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
            
            logits = model(inputs)
            loss = nn.CrossEntropyLoss()(logits.view(-1, model.vocab_size), targets.view(-1))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    """主训练循环"""
    
    logger.info("=" * 80)
    logger.info("[START] GPU 训练开始 (Windows 兼容)")
    logger.info("=" * 80)
    
    # 1. 加载配置
    logger.info("加载配置...")
    config = load_config()
    
    # 设置设备
    device = torch.device(config.get('device', 'cuda'))
    logger.info(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 2. 加载数据
    logger.info(f"加载训练数据 ({config['data']['train_file']})...")
    dataset, vocab = load_data(config['data']['train_file'], vocab_size=config['model']['vocab_size'])
    logger.info(f"数据集大小: {len(dataset)}")
    logger.info(f"词汇表大小: {len(vocab)}")
    
    # 3. 分割数据集
    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 4. 创建数据加载器 (Windows: num_workers=0)
    logger.info("创建数据加载器...")
    batch_size = config['training']['batch_size']
    train_loader = create_windows_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = create_windows_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    logger.info(f"训练批次数: {len(train_loader)}")
    logger.info(f"验证批次数: {len(val_loader)}")
    
    # 5. 创建模型
    logger.info("创建模型...")
    model = TransformerModel(
        vocab_size=config['model']['vocab_size'],
        embedding_dim=config['model']['embedding_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        hidden_dim=config['model']['hidden_dim'],
        max_seq_length=config['model']['max_seq_length'],
        dropout=config['model']['dropout']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数总数: {total_params:,}")
    
    # 6. 设置优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    warmup_steps = config['training']['warmup_steps']
    total_steps = len(train_loader) * config['training']['epochs']
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()
    
    # 7. 训练循环
    epochs = config['training']['epochs']
    best_loss = float('inf')
    patience = config['training']['early_stop_patience']
    patience_counter = 0
    
    logger.info("开始训练...")
    logger.info("=" * 80)
    
    for epoch in range(1, epochs + 1):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch, epochs)
        scheduler.step(len(train_loader))
        
        # 验证
        val_loss = validate(model, val_loader, device)
        
        logger.info(f"Epoch [{epoch}/{epochs}] 训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")
        
        # 保存检查点
        checkpoint_dir = Path(config['paths']['checkpoint_dir'])
        checkpoint_dir.mkdir(exist_ok=True)
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            checkpoint_path = checkpoint_dir / f"best_model_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'vocab': vocab
            }, checkpoint_path)
            logger.info(f"[BEST] 保存最佳模型: {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"早停触发（无改进 {patience} 个 epoch）")
                break
    
    final_path = checkpoint_dir / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'config': config
    }, final_path)
    logger.info(f"[FINAL] 保存最终模型: {final_path}")
    
    logger.info("=" * 80)
    logger.info(f"[DONE] 训练完成! 最佳验证损失: {best_loss:.4f}")
    logger.info("=" * 80)
    
    return final_path


if __name__ == '__main__':
    """
    Windows 多进程安全入口点
    必须将所有初始化代码放在 if __name__ == '__main__': 下面
    """
    torch.multiprocessing.freeze_support()
    
    try:
        final_model = main()
        logger.info(f"最终模型路径: {final_model}")
    except Exception as e:
        logger.error(f"错误: {type(e).__name__}: {e}", exc_info=True)
        sys.exit(1)
