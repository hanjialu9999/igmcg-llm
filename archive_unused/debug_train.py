#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""直接运行训练，带详细调试输出"""

import sys
import os
sys.path.insert(0, '.')

print("[START] 训练脚本开始执行...", flush=True)

try:
    print("[1/10] 导入 argparse...", flush=True)
    import argparse
    
    print("[2/10] 导入 torch 和相关模块...", flush=True)
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.amp import autocast as torch_autocast, GradScaler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    print("[3/10] 导入 yaml...", flush=True)
    import yaml
    
    print("[4/10] 导入项目模块...", flush=True)
    from models.transformer import TransformerModel
    from models.data_utils import load_data, create_dataloader
    
    print("[5/10] 定义辅助函数...", flush=True)
    from pathlib import Path
    from datetime import datetime
    import json
    import numpy as np
    
    print("[6/10] 加载配置...", flush=True)
    config_path = 'config/config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"      - 配置文件: {config_path}")
    print(f"      - 数据文件: {config['data']['train_file']}")
    
    print("[7/10] 设置设备...", flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
    print(f"      - 设备: {device}")
    
    print("[8/10] 加载数据...", flush=True)
    dataset, vocab = load_data(
        config['data']['train_file'],
        vocab_size=config['data']['vocab_size'],
        max_seq_length=config['data']['max_seq_length']
    )
    print(f"      - 数据加载成功")
    print(f"      - 数据集大小: {len(dataset)}")
    print(f"      - 词汇表大小: {len(vocab)}")
    
    print("[9/10] 创建数据加载器...", flush=True)
    dataloader = create_dataloader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 0)  # Windows上用0
    )
    print(f"      - Dataloader 创建成功")
    print(f"      - Batch 数量: {len(dataloader)}")
    
    print("[10/10] 创建模型...", flush=True)
    model = TransformerModel(
        vocab_size=len(vocab),
        embedding_dim=config['model']['embedding_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        hidden_dim=config['model']['hidden_dim'],
        max_seq_length=config['model']['max_seq_length'],
        dropout=config['model']['dropout']
    ).to(device)
    print(f"      - 模型创建成功")
    
    print("\n✅ 所有初始化步骤完成！")
    print("可以开始训练。\n")
    
    # 测试单个batch
    print("[测试] 运行单个 batch...", flush=True)
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        print(f"      - Input shape: {input_ids.shape}")
        print(f"      - Target shape: {target_ids.shape}")
        
        with torch.no_grad():
            output = model(input_ids)
        print(f"      - Output shape: {output.shape}")
        print("✅ 单个batch 测试成功！")
        break
    
except Exception as e:
    print(f"\n❌ 错误: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✅ 训练环境准备完毕，可以启动完整训练")
print("="*80)
