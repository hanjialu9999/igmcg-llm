#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时训练监控脚本
追踪损失、学习率、GPU 内存等指标
"""

import re
import subprocess
import time
from pathlib import Path
from datetime import datetime
import psutil
import os

def get_gpu_stats():
    """获取 GPU 统计信息"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory', '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            return {
                'memory_used_mb': float(parts[0]),
                'memory_total_mb': float(parts[1]),
                'gpu_util': float(parts[2]),
                'mem_util': float(parts[3])
            }
    except Exception as e:
        print(f"GPU 查询失败: {e}")
    return None

def get_latest_training_metrics(log_file='training_gpu.log'):
    """从日志文件提取最新的训练指标"""
    if not Path(log_file).exists():
        return None
    
    try:
        # 尝试多种编码方式
        for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
            try:
                with open(log_file, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                break
            except UnicodeDecodeError:
                continue
        
        # 查找最后的 epoch 信息
        epoch_pattern = r"Epoch \[(\d+)/(\d+)\].*Loss: ([\d.]+)"
        batch_pattern = r"Epoch \[(\d+)/\d+\] Batch \[(\d+)/(\d+)\] Loss: ([\d.]+)"
        val_pattern = r"Epoch \[(\d+)/(\d+)\].*训练损失: ([\d.]+) \| 验证损失: ([\d.]+)"
        
        latest_epoch = None
        latest_batch = None
        latest_val = None
        
        for line in reversed(lines):
            if not latest_val and re.search(val_pattern, line):
                m = re.search(val_pattern, line)
                latest_val = {
                    'epoch': int(m.group(1)),
                    'max_epoch': int(m.group(2)),
                    'train_loss': float(m.group(3)),
                    'val_loss': float(m.group(4))
                }
            elif not latest_batch and re.search(batch_pattern, line):
                m = re.search(batch_pattern, line)
                latest_batch = {
                    'epoch': int(m.group(1)),
                    'batch': int(m.group(2)),
                    'total_batches': int(m.group(3)),
                    'loss': float(m.group(4))
                }
                break
        
        return {'batch': latest_batch, 'validation': latest_val}
    except Exception as e:
        return None

def get_process_memory(process_name='python'):
    """获取 Python 进程的内存占用"""
    try:
        total_mem = 0
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            try:
                if process_name.lower() in proc.info['name'].lower():
                    # 检查命令行是否包含 train_gpu.py
                    cmdline = proc.cmdline()
                    if any('train_gpu' in cmd for cmd in cmdline):
                        total_mem += proc.info['memory_info'].rss / (1024 * 1024)  # 转换为 MB
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total_mem
    except Exception as e:
        return 0

def monitor_training(interval=30, duration=None):
    """监控训练进度"""
    start_time = time.time()
    
    print("\n" + "="*80)
    print("🚀 GPU 训练监控开始")
    print("="*80 + "\n")
    
    while True:
        elapsed = time.time() - start_time
        
        # 获取指标
        gpu_stats = get_gpu_stats()
        training = get_latest_training_metrics()
        cpu_mem = get_process_memory()
        
        # 打印时间戳
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 运行时长: {elapsed/3600:.1f} 小时")
        
        # GPU 统计
        if gpu_stats:
            print(f"📊 GPU:")
            print(f"   显存: {gpu_stats['memory_used_mb']:.0f}MB / {gpu_stats['memory_total_mb']:.0f}MB ({gpu_stats['mem_util']:.1f}%)")
            print(f"   利用率: {gpu_stats['gpu_util']:.1f}%")
        
        # 训练进度
        if training:
            if training['batch']:
                b = training['batch']
                progress = (b['batch'] / b['total_batches']) * 100
                print(f"📈 训练进度:")
                print(f"   Epoch {b['epoch']}: Batch {b['batch']}/{b['total_batches']} ({progress:.1f}%)")
                print(f"   当前损失: {b['loss']:.4f}")
            
            if training['validation']:
                v = training['validation']
                print(f"✓ Epoch {v['epoch']} 完成:")
                print(f"   训练损失: {v['train_loss']:.4f}")
                print(f"   验证损失: {v['val_loss']:.4f}")
        
        # CPU 内存
        if cpu_mem > 0:
            print(f"💾 Python 进程内存: {cpu_mem:.0f}MB")
        
        print("-" * 80)
        
        # 检查是否应该停止
        if duration and elapsed > duration:
            break
        
        time.sleep(interval)

if __name__ == '__main__':
    # 监控 2 小时（7200 秒）或按 Ctrl+C 中断
    try:
        monitor_training(interval=30, duration=7200)
    except KeyboardInterrupt:
        print("\n\n✅ 监控已停止")
