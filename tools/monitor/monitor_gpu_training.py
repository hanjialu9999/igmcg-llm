#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时训练监控脚本
追踪损失、学习率、GPU 内存等指标。

日志格式由 scripts/train.py 决定：
  Epoch {epoch}/{epochs} | Train Loss: {x:.4f} | Val Loss: {x:.4f}
  Epoch {epoch} | Batch {batch_idx + 1}/{total_steps} | Loss: {avg:.4f} | LR: ...

用法:
  python tools/monitor/monitor_gpu_training.py                  # 默认 logs/train.log
  python tools/monitor/monitor_gpu_training.py --log my.log
  python tools/monitor/monitor_gpu_training.py --interval 60 --duration 3600
"""

import argparse
import re
import subprocess
import time
from pathlib import Path
from datetime import datetime
import os

try:
    import psutil
except ImportError:
    psutil = None


def get_gpu_stats():
    """获取 GPU 统计信息（nvidia-smi，AMD/Intel iGPU 不可用时返回 None）。"""
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
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    except Exception as e:
        print(f"GPU 查询失败: {e}")
    return None


def get_latest_training_metrics(log_file='logs/train.log'):
    """从日志文件提取最新的训练指标。

    匹配 scripts/train.py 实际输出格式（无方括号）：
      Epoch {epoch}/{epochs} | Train Loss: {x:.4f} | Val Loss: {x:.4f}
      Epoch {epoch} | Batch {batch_idx + 1}/{total_steps} | Loss: {avg:.4f} | LR: ...
    """
    if not Path(log_file).exists():
        return None

    try:
        # 尝试多种编码方式（Windows GBK 终端重定向时可能混入非 UTF-8 字符）
        lines = []
        for encoding in ['utf-8', 'utf-8-sig', 'gbk', 'latin-1']:
            try:
                with open(log_file, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                break
            except UnicodeDecodeError:
                continue

        # 实际格式（scripts/train.py:596, 245-246）
        batch_pattern = r"Epoch (\d+) \| Batch (\d+)/(\d+) \| Loss: ([\d.]+)"
        val_pattern = r"Epoch (\d+)/(\d+) \| Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)"

        latest_batch = None
        latest_val = None

        for line in reversed(lines):
            if not latest_val:
                m = re.search(val_pattern, line)
                if m:
                    latest_val = {
                        'epoch': int(m.group(1)),
                        'max_epoch': int(m.group(2)),
                        'train_loss': float(m.group(3)),
                        'val_loss': float(m.group(4))
                    }
            if not latest_batch:
                m = re.search(batch_pattern, line)
                if m:
                    latest_batch = {
                        'epoch': int(m.group(1)),
                        'batch': int(m.group(2)),
                        'total_batches': int(m.group(3)),
                        'loss': float(m.group(4))
                    }
                    break

        return {'batch': latest_batch, 'validation': latest_val}
    except Exception:
        return None


def get_process_memory(process_name='python'):
    """获取 Python 进程的内存占用（无 psutil 时返回 0）。"""
    if psutil is None:
        return 0
    try:
        total_mem = 0
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            try:
                if process_name.lower() in proc.info['name'].lower():
                    cmdline = proc.cmdline()
                    if any('train' in cmd for cmd in cmdline):
                        total_mem += proc.info['memory_info'].rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total_mem
    except Exception:
        return 0


def monitor_training(log_file='logs/train.log', interval=30, duration=None):
    """监控训练进度"""
    start_time = time.time()

    print("\n" + "=" * 80)
    print(f"🚀 训练监控开始 (日志: {log_file})")
    print("=" * 80 + "\n")

    while True:
        elapsed = time.time() - start_time

        # 获取指标
        gpu_stats = get_gpu_stats()
        training = get_latest_training_metrics(log_file)
        cpu_mem = get_process_memory()

        # 打印时间戳
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 运行时长: {elapsed/3600:.1f} 小时")

        # GPU 统计
        if gpu_stats:
            print(f"📊 GPU:")
            print(f"   显存: {gpu_stats['memory_used_mb']:.0f}MB / {gpu_stats['memory_total_mb']:.0f}MB ({gpu_stats['mem_util']:.1f}%)")
            print(f"   利用率: {gpu_stats['gpu_util']:.1f}%")
        else:
            print("📊 GPU: (无 nvidia-smi，AMD/Intel iGPU 不支持此查询)")

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
                print(f"✓ Epoch {v['epoch']}/{v['max_epoch']} 完成:")
                print(f"   Train Loss: {v['train_loss']:.4f}")
                print(f"   Val Loss:   {v['val_loss']:.4f}")
        else:
            print("📈 训练进度: (尚未在日志中找到 epoch 记录)")

        # CPU 内存
        if cpu_mem > 0:
            print(f"💾 Python 进程内存: {cpu_mem:.0f}MB")

        print("-" * 80)

        # 检查是否应该停止
        if duration and elapsed > duration:
            break

        # 训练已完成则退出
        if training and training['validation'] and training['validation']['epoch'] >= training['validation']['max_epoch']:
            print("\n✅ 训练已完成（最后一个 epoch 已记录）")
            break

        time.sleep(interval)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='实时训练监控')
    ap.add_argument('--log', default='logs/train.log', help='训练日志路径')
    ap.add_argument('--interval', type=int, default=30, help='轮询间隔秒数')
    ap.add_argument('--duration', type=int, default=None, help='最长监控秒数（默认直到训练结束）')
    args = ap.parse_args()

    try:
        monitor_training(log_file=args.log, interval=args.interval, duration=args.duration)
    except KeyboardInterrupt:
        print("\n\n✅ 监控已停止")
