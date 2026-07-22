#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的训练进度监控

从训练日志（stdout 重定向）解析 epoch 损失并显示进度。
日志格式由 scripts/train.py 决定：
  Epoch {epoch}/{epochs} | Train Loss: {x:.4f} | Val Loss: {x:.4f}
  Epoch {epoch} | Batch {batch_idx + 1}/{total_steps} | Loss: {avg:.4f} | LR: ...

用法:
  python tools/monitor/simple_monitor.py                  # 默认 logs/train.log
  python tools/monitor/simple_monitor.py --log my.log
"""

import argparse
import re
import time
from pathlib import Path


def tail_log(log_file: str, n: int = 30):
    """读取日志文件最后 n 行（跨平台，不依赖 PowerShell）。"""
    try:
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        return lines[-n:] if len(lines) >= n else lines
    except FileNotFoundError:
        return []
    except Exception:
        return []


def parse_epoch_loss(lines):
    """从日志行解析 epoch 训练/验证损失信息。

    匹配 scripts/train.py:596 实际格式：
      `Epoch {epoch}/{epochs} | Train Loss: {x:.4f} | Val Loss: {x:.4f}`
    """
    pattern = r"Epoch (\d+)/(\d+) \| Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)"
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
    ap = argparse.ArgumentParser(description='简单训练进度监控')
    ap.add_argument('--log', default='logs/train.log',
                    help='训练日志路径（默认 logs/train.log；train.py 输出需重定向到此）')
    ap.add_argument('--interval', type=int, default=30, help='轮询间隔秒数')
    args = ap.parse_args()

    log_file = args.log

    print("\n" + "=" * 60)
    print(f"📊 训练监控 (日志: {log_file})")
    print("=" * 60 + "\n")

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
                    print(f"  Train Loss: {epoch_info['train_loss']:.4f}")
                    print(f"  Val Loss:   {epoch_info['val_loss']:.4f}")
                    print(f"  运行时间: {hours:.1f}h | 预计剩余: {eta_hours:.1f}h")
                    print("-" * 60)

                    last_epoch = epoch

                    if epoch >= max_epochs:
                        print("\n✅ 训练完成！")
                        break

        time.sleep(args.interval)


if __name__ == '__main__':
    main()
