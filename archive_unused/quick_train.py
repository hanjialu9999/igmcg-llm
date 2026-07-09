#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速GPU训练启动脚本 - One-Click Training Launcher
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("\n" + "="*80)
    print("  🚀 AI文本生成模型 - GPU训练系统")
    print("="*80)
    
    project_root = Path(__file__).parent.absolute()
    
    # 步骤1: 准备数据
    print("\n[步骤 1/3] 准备训练数据...")
    print("-" * 80)
    result = subprocess.run(
        [sys.executable, 'prepare_training.py'],
        cwd=project_root
    )
    if result.returncode != 0:
        print("❌ 数据准备失败")
        return False
    
    # 步骤2: 验证GPU
    print("\n[步骤 2/3] 验证GPU环境...")
    print("-" * 80)
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            print(f"✅ GPU已检测: {device_name} (共{device_count}个)")
            print(f"   CUDA版本: {torch.version.cuda}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️  未检测到GPU，将使用CPU（极慢）")
    except Exception as e:
        print(f"⚠️  GPU检查失败: {e}")
    
    # 步骤3: 启动训练
    print("\n[步骤 3/3] 启动GPU训练...")
    print("-" * 80)
    print("📝 请选择运行方式:")
    print("  1. 前台运行 (直接看日志)")
    print("  2. 后台运行 (不干扰系统)")
    print("  3. 取消")
    
    choice = input("\n你的选择 (1/2/3)? ").strip()
    
    if choice == '1':
        print("\n✅ 启动前台训练...")
        print("提示: 按 Ctrl+C 可以停止，但会保留进度")
        print("-" * 80)
        subprocess.run(
            [sys.executable, 'scripts/train.py', '--config', 'config/config.yaml'],
            cwd=project_root
        )
    elif choice == '2':
        print("\n✅ 启动后台训练...")
        if os.name == 'nt':  # Windows
            # 在Windows上用detached进程启动
            import subprocess
            subprocess.Popen(
                [sys.executable, 'scripts/train.py', '--config', 'config/config.yaml'],
                cwd=project_root,
                stdout=open('training.log', 'w'),
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if hasattr(subprocess, 'CREATE_NEW_PROCESS_GROUP') else 0
            )
            print("✅ 训练已在后台启动")
            print("📝 日志文件: training.log")
            print("🔍 监控进度: python monitor_training.py")
        else:  # Linux/Mac
            subprocess.Popen(
                ['nohup', sys.executable, 'scripts/train.py', '--config', 'config/config.yaml'],
                cwd=project_root,
                stdout=open('training.log', 'w'),
                stderr=subprocess.STDOUT
            )
            print("✅ 训练已在后台启动 (nohup)")
            print("📝 日志文件: training.log")
            print("🔍 监控进度: python monitor_training.py")
    else:
        print("\n取消训练")
        return False
    
    print("\n" + "="*80)
    print("  训练配置:")
    print("  - 数据: data/train_data_final.txt")
    print("  - 模型: Transformer 6层 (21.5M参数)")
    print("  - Batch Size: 128")
    print("  - 优化: GPU加速 + 混合精度训练")
    print("  - Early Stop: 10 epochs无改进时停止")
    print("="*80 + "\n")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
