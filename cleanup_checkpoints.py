#!/usr/bin/env python3
"""
独立清理脚本 - 可以单独运行来清理多余模型
"""

import os
import glob

def cleanup_old_checkpoints(checkpoint_dir='checkpoints', keep_last_n=5):
    """Clean up old checkpoints, keep only the best model and last N epochs"""
    
    # Find all epoch checkpoint files
    epoch_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'model_epoch_*.pt')))
    
    print(f"Found {len(epoch_files)} epoch checkpoint(s)")
    
    if len(epoch_files) <= keep_last_n:
        print(f"No cleanup needed (keep_last_n={keep_last_n})")
        return
    
    # Keep only the last N checkpoints
    files_to_delete = epoch_files[:-keep_last_n]
    
    print(f"\n将删除 {len(files_to_delete)} 个旧模型文件:")
    print("-" * 50)
    
    deleted_count = 0
    freed_size = 0
    
    for file_path in files_to_delete:
        try:
            file_size = os.path.getsize(file_path)
            os.remove(file_path)
            deleted_count += 1
            freed_size += file_size
            print(f"✓ {os.path.basename(file_path):30s} ({file_size/1024/1024:.1f}MB)")
        except Exception as e:
            print(f"✗ {os.path.basename(file_path):30s} (错误: {e})")
    
    print("-" * 50)
    print(f"\n✅ 清理完成:")
    print(f"   • 删除文件数: {deleted_count}")
    print(f"   • 释放空间: {freed_size/1024/1024:.1f}MB")
    print(f"   • 保留最后 {keep_last_n} 个 epoch 检查点")
    print(f"   • 保留 final_model.pt")
    
    # List remaining checkpoints
    remaining_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'model_epoch_*.pt')))
    if remaining_files:
        print(f"\n保留的模型文件:")
        for f in remaining_files:
            print(f"   {os.path.basename(f)}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--keep', type=int, default=5,
                        help='保留最后N个checkpoint (默认: 5)')
    parser.add_argument('--dir', type=str, default='checkpoints',
                        help='Checkpoint 目录 (默认: checkpoints)')
    args = parser.parse_args()
    
    cleanup_old_checkpoints(args.dir, args.keep)
