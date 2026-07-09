#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从检查点恢复 GPU 训练
从保存的状态继续 200-epoch 训练
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
import glob

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
        logging.FileHandler('training_resume.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path='config/config.yaml'):
    """加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def find_latest_checkpoint(checkpoint_dir='checkpoints'):
    """找到最新的 epoch 检查点"""
    epoch_files = glob.glob(os.path.join(checkpoint_dir, 'model_epoch_*.pt'))
    if not epoch_files:
        return None
    
    # 提取 epoch 号并找最新的
    epochs = []
    for f in epoch_files:
        try:
            epoch_num = int(os.path.basename(f).replace('model_epoch_', '').replace('.pt', ''))
            epochs.append((epoch_num, f))
        except:
            pass
    
    if epochs:
        epochs.sort(reverse=True)
        logger.info(f"找到检查点: Epoch {epochs[0][0]}")
        return epochs[0]
    return None


def load_checkpoint(checkpoint_path, model, optimizer, device):
    """加载检查点"""
    logger.info(f"[RESUME] 从检查点加载: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型状态
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("  [OK] 模型状态已加载")
    
    # 加载优化器状态（如果存在）
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("  [OK] 优化器状态已加载")
    
    # 获取 epoch 信息
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_loss = checkpoint.get('loss', float('inf'))
    
    logger.info(f"  [OK] 继续从 Epoch {start_epoch} 开始训练")
    logger.info(f"  [OK] 前一个最佳损失: {best_loss:.4f}")
    
    return start_epoch, best_loss


def create_windows_dataloader(dataset, batch_size=16, shuffle=True):
    """为 Windows 创建数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle
    )


def train_epoch(model, dataloader, optimizer, scaler, device, epoch, max_epochs):
    """训练单个 epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        if isinstance(batch, dict):
            inputs = batch['input_ids'].to(device)
            targets = batch['target_ids'].to(device)
        else:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
        
        optimizer.zero_grad()
        
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
    """主训练循环（恢复模式）"""
    
    logger.info("=" * 80)
    logger.info("[RESUME] GPU 训练恢复 (从检查点继续)")
    logger.info("=" * 80)
    
    # 1. 加载配置
    logger.info("加载配置...")
    config = load_config()
    
    device = torch.device(config.get('device', 'cuda'))
    logger.info(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 2. 加载数据
    logger.info(f"加载训练数据 ({config['data']['train_file']})...")
    dataset, vocab = load_data(config['data']['train_file'], vocab_size=config['model']['vocab_size'])
    logger.info(f"数据集大小: {len(dataset)}")
    
    # 3. 分割数据集
    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 4. 创建数据加载器
    logger.info("创建数据加载器...")
    batch_size = config['training']['batch_size']
    train_loader = create_windows_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = create_windows_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    logger.info(f"训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}")
    
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
    
    # 6. 设置优化器
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], 
                           weight_decay=config['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs']
    )
    scaler = GradScaler()
    
    # 7. 尝试加载检查点
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True)
    
    start_epoch = 1
    best_loss = float('inf')
    patience_counter = 0
    
    latest = find_latest_checkpoint(str(checkpoint_dir))
    if latest:
        epoch_num, checkpoint_path = latest
        try:
            start_epoch, best_loss = load_checkpoint(checkpoint_path, model, optimizer, device)
        except Exception as e:
            logger.warning(f"加载检查点失败: {e}, 将从头开始")
            start_epoch = 1
    else:
        logger.info("未找到检查点, 从 Epoch 1 开始训练")
    
    # 8. 训练循环
    epochs = config['training']['epochs']
    patience = config['training']['early_stop_patience']
    
    logger.info("[START] 开始训练...")
    logger.info("=" * 80)
    
    for epoch in range(start_epoch, epochs + 1):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch, epochs)
        scheduler.step()
        
        # 验证
        val_loss = validate(model, val_loader, device)
        
        logger.info(f"Epoch [{epoch}/{epochs}] 训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")
        
        # 保存检查点
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            checkpoint_path = checkpoint_dir / f"model_epoch_{epoch}.pt"
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
    
    # 保存最终模型
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
    torch.multiprocessing.freeze_support()
    
    try:
        final_model = main()
        logger.info(f"最终模型路径: {final_model}")
    except Exception as e:
        logger.error(f"错误: {type(e).__name__}: {e}", exc_info=True)
        sys.exit(1)
