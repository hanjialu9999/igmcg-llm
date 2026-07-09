#!/usr/bin/env python3
"""
训练监控 - 实时查看训练进度的脚本
"""

import time
import os

checkpoint_dir = 'checkpoints'

def get_latest_checkpoint():
    """获取最新的checkpoint"""
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_') and f.endswith('.pt')]
    if not files:
        return None, 0
    
    # 按epoch数排序
    epochs = []
    for f in files:
        try:
            epoch = int(f.split('_')[2].split('.')[0])
            epochs.append((epoch, f))
        except:
            pass
    
    if not epochs:
        return None, 0
    
    epochs.sort()
    return epochs[-1][1], epochs[-1][0]

def format_time(seconds):
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.0f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分"
    else:
        return f"{seconds/3600:.1f}小时"

print("\n" + "="*70)
print("📊 训练监控面板")
print("="*70)

print("\n⏳ 等待训练数据加载...\n")

start_time = time.time()
last_epoch = 0

while True:
    checkpoint, current_epoch = get_latest_checkpoint()
    elapsed = time.time() - start_time
    
    if current_epoch > last_epoch:
        elapsed_epoch = elapsed / max(1, current_epoch)
        remaining = elapsed_epoch * (50 - current_epoch)
        
        print(f"✅ Epoch {current_epoch}/50 完成")
        print(f"  • 已用时: {format_time(elapsed)}")
        print(f"  • 预计剩余: {format_time(remaining)}")
        print(f"  • 预计完成: {format_time(elapsed + remaining)}")
        print()
        
        last_epoch = current_epoch
        
        if current_epoch >= 50:
            print("="*70)
            print("🎉 训练完成！")
            print("="*70)
            print("\n现在可以运行对话系统：")
            print("  python dialogue_interactive.py")
            print()
            break
    
    time.sleep(5)  # 每5秒检查一次
