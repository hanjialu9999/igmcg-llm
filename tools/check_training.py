#!/usr/bin/env python3
"""
训练状态速览 - 快速查看当前训练进度
"""

import os
import glob

checkpoint_dir = 'checkpoints'

# 获取最新的checkpoint
files = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_') and f.endswith('.pt')]
if files:
    epochs = []
    for f in files:
        try:
            epoch = int(f.split('_')[2].split('.')[0])
            epochs.append((epoch, f))
        except:
            pass
    
    epochs.sort(reverse=True)
    latest_epoch = epochs[0][0] if epochs else 0
else:
    latest_epoch = 0

print("\n" + "="*70)
print("🚀 训练状态快览")
print("="*70)

if latest_epoch == 0:
    print("\n⏳ 训练进行中... (正在加载数据或运行初始epoch)")
    print("\n提示：可以运行以下命令监控训练：")
    print("  python monitor_training.py")
else:
    progress = (latest_epoch / 50) * 100
    print(f"\n✅ 当前进度: Epoch {latest_epoch}/50 ({progress:.1f}%)")
    
    # 进度条
    filled = int(progress / 2)
    empty = 50 - filled
    bar = "█" * filled + "░" * empty
    print(f"\n  [{bar}] {progress:.1f}%")
    
    print(f"\n📝 当前模型: checkpoints/model_epoch_{latest_epoch}.pt")
    print(f"\n💡 你可以：")
    
    if latest_epoch < 10:
        print(f"  • 等待更多epoch（当前training中...）")
        print(f"  • 或运行 monitor_training.py 实时监控")
    else:
        print(f"  • 使用最新模型测试: python quick_demo.py")
        print(f"  • 或继续等待完整50个epoch")
        print(f"  • 运行 monitor_training.py 监控剩余进度")

print("\n" + "="*70 + "\n")
