#!/usr/bin/env python3
"""
新数据训练总结报告
"""

import os
import json
from datetime import datetime

print("\n" + "="*70)
print("📊 新数据训练总结报告")
print("="*70)

print("\n📈 数据情况：")
print("  • 原始数据: 2561 句")
print("  • 新增数据: 766 句")
print("    - identity_coweive.txt: 200 句")
print("    - natural_chat.txt: 562 句")
print("  • 合并后数据: 3327 句 (去重后)")
print("  • 增长幅度: +30%")

print("\n🧠 模型配置：")
print("  • 嵌入维度: 256")
print("  • 注意力头数: 8")
print("  • Transformer层数: 4")
print("  • 隐层维度: 512")
print("  • 训练轮数: 50")

print("\n🎯 当前状态：")

# 检查最新checkpoint
checkpoint_dir = 'checkpoints'
files = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_') and f.endswith('.pt')]
if files:
    epochs = []
    for f in files:
        try:
            epoch = int(f.split('_')[2].split('.')[0])
            mtime = os.path.getmtime(os.path.join(checkpoint_dir, f))
            epochs.append((epoch, f, mtime))
        except Exception:
            pass
    
    epochs.sort(reverse=True)
    latest_epoch, latest_file, mtime = epochs[0]
    mtime_str = datetime.fromtimestamp(mtime).strftime('%H:%M:%S')
    
    progress = (latest_epoch / 50) * 100
    print(f"  ✅ 已完成: Epoch {latest_epoch}/50 ({progress:.1f}%)")
    print(f"  ⏰ 最后更新: {mtime_str}")
    
    if latest_epoch == 50:
        print("\n✨ 训练已完成！")
    else:
        print(f"\n⏳ 还需 {50 - latest_epoch} 个 epoch...")

print("\n💡 建议：")
print("  1. 使用最新模型测试:")
print("     python quick_demo.py")
print()
print("  2. 进行交互式对话:")
print("     python dialogue_interactive.py")
print()
print("  3. 监控训练进度:")
print("     python monitor_training.py")

print("\n" + "="*70 + "\n")
