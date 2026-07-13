#!/usr/bin/env python3
"""
实时训练监控面板 - 显示详细的训练进度和预计时间
"""

import os
import time
from datetime import datetime, timedelta

def get_latest_epoch():
    """获取最新完成的epoch"""
    checkpoint_dir = 'checkpoints'
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_') and f.endswith('.pt')]
    
    if not files:
        return 0, None
    
    epochs = []
    for f in files:
        try:
            epoch = int(f.split('_')[2].split('.')[0])
            mtime = os.path.getmtime(os.path.join(checkpoint_dir, f))
            epochs.append((epoch, f, mtime))
        except Exception:
            pass
    
    if not epochs:
        return 0, None
    
    epochs.sort()
    return epochs[-1][0], epochs[-1][2]

def format_time(seconds):
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.0f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    else:
        hours = seconds / 3600
        if hours < 24:
            return f"{hours:.1f}小时"
        else:
            days = hours / 24
            return f"{days:.1f}天"

def get_progress_bar(current, total, width=50):
    """生成进度条"""
    percentage = (current / total) * 100
    filled = int((percentage / 100) * width)
    empty = width - filled
    
    bar = "█" * filled + "░" * empty
    return bar, percentage

print("\n" + "="*75)
print("🚀 AI 模型训练监控面板")
print("="*75)
print("\n📊 训练信息：")
print("  • 数据量: 3767 句")
print("  • 模型: 256D embeddings, 4层Transformer")
print("  • 总训练轮数: 100 epochs")
print("  • 批次大小: 8")
print("\n" + "─"*75)

start_time = time.time()
first_epoch_time = None

print("\n⏳ 等待训练开始...\n")

while True:
    current_epoch, mtime = get_latest_epoch()
    elapsed_total = time.time() - start_time
    
    if current_epoch > 0:
        if first_epoch_time is None:
            first_epoch_time = elapsed_total
        
        bar, percentage = get_progress_bar(current_epoch, 100)
        
        # 计算平均每个epoch的时间
        avg_epoch_time = elapsed_total / current_epoch if current_epoch > 0 else 0
        remaining_epochs = 100 - current_epoch
        remaining_time = avg_epoch_time * remaining_epochs
        total_time = elapsed_total + remaining_time
        
        # 预计完成时间
        complete_time = datetime.now() + timedelta(seconds=remaining_time)
        
        print(f"\r✅ Epoch: {current_epoch:3d}/100 | {bar} | {percentage:5.1f}% | "
              f"已用: {format_time(elapsed_total):8s} | "
              f"剩余: {format_time(remaining_time):8s}", end='', flush=True)
        
        if current_epoch >= 100:
            print("\n\n" + "="*75)
            print("🎉 训练完成！")
            print("="*75)
            print(f"\n📊 最终统计：")
            print(f"  • 总耗时: {format_time(elapsed_total)}")
            print(f"  • 平均每个epoch: {format_time(avg_epoch_time)}")
            print(f"  • 完成时间: {datetime.now().strftime('%H:%M:%S')}")
            print(f"\n💾 最优模型: checkpoints/final_model.pt")
            print(f"\n🎯 接下来：")
            print(f"  • python dialogue_interactive.py  (交互式对话)")
            print(f"  • python quick_demo.py            (快速演示)")
            print(f"  • python dialogue_demo.py         (自动演示)")
            print("\n" + "="*75 + "\n")
            break
    
    time.sleep(10)  # 每10秒检查一次
