#!/usr/bin/env python3
"""
实时训练监控面板 - 显示详细的训练进度和预计时间。

通过观察 checkpoints/model_epoch_*.pt 文件计数；epoch 总数从配置文件读取
（默认 configs/pretrain.yaml），不再硬编码 100 / 50 等历史值。

用法:
  python tools/monitor/monitor_live.py
  python tools/monitor/monitor_live.py --config configs/config_hybrid.yaml
"""

import argparse
import os
import time
from datetime import datetime, timedelta


def _load_epochs_from_config(config_path: str) -> int:
    """从 YAML 配置读取 training.epochs；失败时返回 0。"""
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        return int(cfg.get('training', {}).get('epochs', 0))
    except Exception:
        return 0


def get_latest_epoch(checkpoint_dir: str = 'checkpoints'):
    """获取最新完成的 epoch 号与其文件 mtime。"""
    if not os.path.isdir(checkpoint_dir):
        return 0, None
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_') and f.endswith('.pt')]

    if not files:
        return 0, None

    epochs = []
    for f in files:
        try:
            name = f[len('model_epoch_'):-len('.pt')]
            epoch = int(name)
            mtime = os.path.getmtime(os.path.join(checkpoint_dir, f))
            epochs.append((epoch, f, mtime))
        except ValueError:
            continue

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


def main():
    ap = argparse.ArgumentParser(description='实时训练监控面板')
    ap.add_argument('--config', default='configs/pretrain.yaml',
                    help='配置文件路径（读取 training.epochs）')
    ap.add_argument('--checkpoint-dir', default='checkpoints', help='checkpoint 目录')
    ap.add_argument('--interval', type=int, default=10, help='轮询间隔秒数')
    args = ap.parse_args()

    total_epochs = _load_epochs_from_config(args.config)
    if total_epochs <= 0:
        print(f"⚠️  未能从 {args.config} 读取 training.epochs；将仅显示已完成的 epoch 数。")
        total_epochs = None

    print("\n" + "=" * 75)
    print("🚀 训练监控面板")
    if total_epochs:
        print(f"   总训练轮数: {total_epochs} epochs")
    print(f"   checkpoint 目录: {args.checkpoint_dir}")
    print("=" * 75 + "\n" + "-" * 75)

    start_time = time.time()
    first_epoch_time = None

    print("\n⏳ 等待训练开始...\n")

    while True:
        current_epoch, mtime = get_latest_epoch(args.checkpoint_dir)
        elapsed_total = time.time() - start_time

        if current_epoch > 0:
            if first_epoch_time is None:
                first_epoch_time = elapsed_total

            if total_epochs:
                bar, percentage = get_progress_bar(current_epoch, total_epochs)

                avg_epoch_time = elapsed_total / current_epoch if current_epoch > 0 else 0
                remaining_epochs = total_epochs - current_epoch
                remaining_time = avg_epoch_time * remaining_epochs

                complete_time = datetime.now() + timedelta(seconds=remaining_time)

                print(f"\r✅ Epoch: {current_epoch:3d}/{total_epochs} | {bar} | {percentage:5.1f}% | "
                      f"已用: {format_time(elapsed_total):8s} | "
                      f"剩余: {format_time(remaining_time):8s}", end='', flush=True)

                if current_epoch >= total_epochs:
                    print("\n\n" + "=" * 75)
                    print("🎉 训练完成！")
                    print("=" * 75)
                    print(f"\n📊 最终统计：")
                    print(f"  • 总耗时: {format_time(elapsed_total)}")
                    print(f"  • 平均每个epoch: {format_time(avg_epoch_time)}")
                    print(f"  • 完成时间: {datetime.now().strftime('%H:%M:%S')}")
                    print(f"\n💾 最优模型: {args.checkpoint_dir}/final_model.pt")
                    print(f"\n🎯 接下来：")
                    print(f"  • python tools/dialogue_interactive.py  (交互式对话)")
                    print(f"  • python tools/quick_demo.py            (快速演示)")
                    print("\n" + "=" * 75 + "\n")
                    break
            else:
                print(f"\r✅ Epoch: {current_epoch} | 已用: {format_time(elapsed_total):8s}", end='', flush=True)

        time.sleep(args.interval)


if __name__ == '__main__':
    main()
