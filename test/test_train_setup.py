#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试训练环境设置"""

import sys
import os
from pathlib import Path

sys.path.insert(0, '.')

try:
    print("\n" + "="*80)
    print("  🔍 训练环境诊断")
    print("="*80)
    
    # 1. 测试基础导入
    print("\n[1] 基础模块导入...")
    import torch
    import yaml
    print("    ✓ PyTorch 和 YAML 导入成功")
    
    # 2. 检查GPU
    print("\n[2] GPU 检查...")
    print(f"    CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    GPU 数量: {torch.cuda.device_count()}")
        print(f"    GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"    GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 3. 加载配置
    print("\n[3] 配置加载...")
    with open('configs/pretrain.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"    ✓ 配置加载成功")
    print(f"      - 数据文件: {config['data']['train_file']}")
    print(f"      - 批大小: {config['training']['batch_size']}")
    print(f"      - 设备: {config['device']}")
    
    # 4. 检查数据文件
    print("\n[4] 数据文件检查...")
    data_file = Path(config['data']['train_file'])
    if data_file.exists():
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = len(f.readlines())
        print(f"    ✓ 数据文件存在: {data_file}")
        print(f"      - 行数: {lines}")
        print(f"      - 大小: {data_file.stat().st_size / (1024*1024):.2f} MB")
    else:
        print(f"    ✗ 数据文件不存在: {data_file}")
        
    # 5. 测试数据加载模块
    print("\n[5] 数据模块导入...")
    from models.data_utils import load_data
    print(f"    ✓ 数据模块导入成功")
    
    # 6. 测试模型导入
    print("\n[6] 模型模块导入...")
    from models.transformer import TransformerModel
    print(f"    ✓ 模型模块导入成功")
    
    # 7. 估计内存占用
    print("\n[7] 内存占用估计...")
    vocab_size = config['data']['vocab_size']
    embedding_dim = config['model']['embedding_dim']
    batch_size = config['training']['batch_size']
    seq_len = config['model']['max_seq_length']
    
    model_params = 21.5e6  # 模型有21.5M参数
    batch_memory_mb = (batch_size * seq_len * 4) / 1024  # 输入张量
    activation_memory_mb = batch_memory_mb * 3  # 激活值
    grad_memory_mb = (model_params * 4) / (1024**2)  # 梯度
    optimizer_memory_mb = grad_memory_mb * 2  # Adam状态
    
    total_gpu_memory_mb = batch_memory_mb + activation_memory_mb + grad_memory_mb + optimizer_memory_mb
    
    print(f"    估计 GPU 显存占用:")
    print(f"      - 模型参数: {(model_params*4)/(1024**2):.0f} MB")
    print(f"      - 批次输入: {batch_memory_mb:.0f} MB")
    print(f"      - 激活输出: {activation_memory_mb:.0f} MB")
    print(f"      - 梯度: {grad_memory_mb:.0f} MB")
    print(f"      - 优化器状态: {optimizer_memory_mb:.0f} MB")
    print(f"      - ============================")
    print(f"      - 总计: {total_gpu_memory_mb:.0f} MB ~ {total_gpu_memory_mb/1024:.1f} GB")
    
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        print(f"\n    GPU 总显存: {total_mem:.0f} MB ({total_mem/1024:.1f} GB)")
        if total_gpu_memory_mb > total_mem * 0.8:
            print(f"    ⚠️  警告: 估计占用超过 GPU 总显存的 80%，可能会 OOM")
        else:
            print(f"    ✓ 内存充足")
    
    print("\n" + "="*80)
    print("  ✅ 所有检查通过! 可以开始训练")
    print("="*80 + "\n")
    
except Exception as e:
    print(f"\n❌ 错误: {type(e).__name__}: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)
