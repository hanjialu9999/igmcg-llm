#!/usr/bin/env python3
"""
训练监控 - 通过观察 checkpoints/ 目录下的 model_epoch_*.pt 文件跟踪进度。

epoch 总数从配置文件读取（默认 configs/pretrain.yaml），避免硬编码。

用法:
  python tools/monitor/monitor_training.py
  python tools/monitor/monitor_training.py --config configs/config_hybrid.yaml
"""

import argparse
import os
import sys
import time
from pathlib import Path

# 加载配置以读取 epochs 总数
def _load_epochs_from_config(config_path: str) -> int:
    """从 YAML 配置读取 training.epochs；失败时返回 0（仅以 checkpoint 为准）。"""
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        return int(cfg.get('training', {}).get('epochs', 0))
    except Exception:
        return 0


def get_latest_checkpoint(checkpoint_dir: str = 'checkpoints'):
    """获取最新的 checkpoint 文件名与 epoch 号。"""
    if not os.path.isdir(checkpoint_dir):
        return None, 0
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_') and f.endswith('.pt')]
    if not files:
        return None, 0

    epochs = []
    for f in files:
        try:
            # model_epoch_{N}.pt -> N
            name = f[len('model_epoch_'):-len('.pt')]
            epoch = int(name)
            epochs.append((epoch, f))
        except ValueError:
            continue

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


def main():
    ap = argparse.ArgumentParser(description='训练 checkpoint 监控')
    ap.add_argument('--config', default='configs/pretrain.yaml',
                    help='配置文件路径（读取 training.epochs 作为总 epoch 数）')
    ap.add_argument('--checkpoint-dir', default='checkpoints', help='checkpoint 目录')
    ap.add_argument('--interval', type=int, default=5, help='轮询间隔秒数')
    args = ap.parse_args()

    total_epochs = _load_epochs_from_config(args.config)
    if total_epochs <= 0:
        print(f"⚠️  未能从 {args.config} 读取 training.epochs，将仅以 checkpoint 为准（无法预估剩余时间）。")
        total_epochs = None

    print("\n" + "=" * 70)
    print("📊 训练监控面板")
    if total_epochs:
        print(f"   总 epochs: {total_epochs}")
    print(f"   checkpoint 目录: {args.checkpoint_dir}")
    print("=" * 70)

    print("\n⏳ 等待训练数据加载...\n")

    start_time = time.time()
    last_epoch = 0

    while True:
        checkpoint, current_epoch = get_latest_checkpoint(args.checkpoint_dir)
        elapsed = time.time() - start_time

        if current_epoch > last_epoch:
            if total_epochs:
                elapsed_epoch = elapsed / max(1, current_epoch)
                remaining = elapsed_epoch * (total_epochs - current_epoch)

                print(f"✅ Epoch {current_epoch}/{total_epochs} 完成")
                print(f"  • 已用时: {format_time(elapsed)}")
                print(f"  • 预计剩余: {format_time(remaining)}")
                print(f"  • 预计完成: {format_time(elapsed + remaining)}")
            else:
                print(f"✅ Epoch {current_epoch} 完成")
                print(f"  • 已用时: {format_time(elapsed)}")
            print()

            last_epoch = current_epoch

            if total_epochs and current_epoch >= total_epochs:
                print("=" * 70)
                print("🎉 训练完成！")
                print("=" * 70)
                print("\n现在可以运行对话系统：")
                print("  python tools/dialogue_interactive.py")
                print()
                break

        time.sleep(args.interval)


if __name__ == '__main__':
    main()
