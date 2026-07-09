#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的训练进度监控
"""

import time
import re
from pathlib import Path
import subprocess

def tail_log(log_file, n=20):
    """获取日志文件的最后 n 行"""
    try:
        p = subprocess.run(['Get-Content', log_file, '-Tail', str(n)], 
                          capture_output=True, text=True, shell=True)
        return p.stdout.strip().split('\n')
    except:
        return []

def parse_epoch_loss(lines):
    """从日志行解析 epoch 损失信息"""
    pattern = r"Epoch \[(\d+)/(\d+)\].*训练损失: ([\d.]+) \| 验证损失: ([\d.]+)"
    for line in reversed(lines):
        m = re.search(pattern, line)
        if m:
            return {
                'epoch': int(m.group(1)),
                'max_epochs': int(m.group(2)),
                'train_loss': float(m.group(3)),
                'val_loss': float(m.group(4))
            }
    return None

def main():
    log_file = 'training_final.log'
    
    print("\n" + "="*60)
    print("📊 GPU 训练监控")
    print("="*60 + "\n")
    
    last_epoch = 0
    start_time = time.time()
    
    while True:
        if Path(log_file).exists():
            lines = tail_log(log_file, 30)
            epoch_info = parse_epoch_loss(lines)
            
            if epoch_info:
                epoch = epoch_info['epoch']
                max_epochs = epoch_info['max_epochs']
                
                if epoch != last_epoch:
                    elapsed = time.time() - start_time
                    hours = elapsed / 3600
                    epoch_time = elapsed / epoch if epoch > 0 else 0
                    eta_hours = epoch_time * (max_epochs - epoch) / 3600
                    
                    progress = (epoch / max_epochs) * 100
                    
                    print(f"✓ Epoch {epoch}/{max_epochs} ({progress:.1f}%)")
                    print(f"  训练损失: {epoch_info['train_loss']:.4f}")
                    print(f"  验证损失: {epoch_info['val_loss']:.4f}")
                    print(f"  运行时间: {hours:.1f}h | 预计剩余: {eta_hours:.1f}h")
                    print("-" * 60)
                    
                    last_epoch = epoch
                    
                    if epoch >= max_epochs:
                        print("\n✅ 训练完成！")
                        break
        
        time.sleep(30)

if __name__ == '__main__':
    main()
