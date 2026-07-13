from __future__ import annotations

import os
import glob
import yaml
from typing import Any, Dict, Optional
import logging
import sys
import functools
import torch


def get_logger(name: str = "igmcg") -> logging.Logger:
    """返回统一配置的 logger（首次调用配置 root handler，避免重复输出）。"""
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    return logger


class IGMCGError(Exception):
    """用户级错误：配合 cli_guard 捕获后打印清晰信息并以非 0 退出。"""
    pass


def cli_guard(func):
    """装饰器：统一捕获 main() 异常，打印清晰信息并以非 0 退出。

    替代零散的 try/except：用户级 IGMCGError 与常见 IO/配置/运行时错误给 [ERROR]，
    其余异常给 [FATAL]；SystemExit / KeyboardInterrupt 透传（不影响 argparse 与 Ctrl-C）。
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except IGMCGError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            sys.exit(1)
        except (FileNotFoundError, KeyError, ValueError, yaml.YAMLError, RuntimeError) as e:
            print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"[FATAL] {type(e).__name__}: {e}", file=sys.stderr)
            sys.exit(1)
    return wrapper


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    best_loss: float,
                    checkpoint_dir: str,
                    vocab_size: int,
                    model_config: Optional[Dict] = None) -> str:
    """
    Save model checkpoint with separate config YAML for weights_only=True compatibility.

    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        best_loss: Best loss achieved so far
        checkpoint_dir: Directory to save checkpoint
        vocab_size: Vocabulary size
        model_config: Model configuration dict (saved as separate YAML)

    Returns:
        Path to saved checkpoint
    """
    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'vocab_size': vocab_size,
    }, checkpoint_path)

    # Save config separately for weights_only=True compatibility
    if model_config is not None:
        config_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}_config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(model_config, f, allow_unicode=True)

    print(f"Checkpoint saved at {checkpoint_path}")
    return checkpoint_path


def cleanup_old_checkpoints(checkpoint_dir: str, keep_last_n: int = 5):
    """Clean up old checkpoints, keep only the best model and last N epochs."""
    epoch_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'model_epoch_*.pt')))

    if len(epoch_files) <= keep_last_n:
        print(f"Found {len(epoch_files)} checkpoint(s). No cleanup needed (keep_last_n={keep_last_n})")
        return

    files_to_delete = epoch_files[:-keep_last_n]
    deleted_count = 0
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            # Also remove corresponding config YAML
            config_path = file_path.replace('.pt', '_config.yaml')
            if os.path.exists(config_path):
                os.remove(config_path)
            deleted_count += 1
            print(f"Removed: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Failed to remove {file_path}: {e}")

    print(f"\n✅ Cleanup complete: Deleted {deleted_count} old checkpoint(s)")
    print(f"   Kept last {keep_last_n} checkpoint(s) and final_model.pt")


def backup_existing_checkpoints(checkpoint_dir: str, backup_root: Optional[str] = None,
                                project_root: Optional[str] = None) -> Optional[str]:
    """Backup existing checkpoints before training starts."""
    if backup_root is None:
        if project_root is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        backup_root = os.path.join(project_root, 'archive_unused', 'checkpoints_backup')

    if not os.path.isdir(checkpoint_dir) or not os.listdir(checkpoint_dir):
        print("No existing checkpoints to back up; skipping backup.")
        return None

    os.makedirs(backup_root, exist_ok=True)
    base = os.path.basename(os.path.normpath(checkpoint_dir))
    dest = os.path.join(backup_root, base)
    n = 0
    while os.path.exists(dest):
        n += 1
        dest = os.path.join(backup_root, f"{base}_{n}")

    import shutil
    shutil.copytree(checkpoint_dir, dest)
    print(f"Backed up existing checkpoints -> {dest}")
    return dest


def save_final_model(model: torch.nn.Module,
                     vocab: Any,
                     checkpoint_dir: str,
                     model_config: Optional[Dict] = None) -> tuple:
    """
    Save final model and vocabulary.

    Returns:
        Tuple of (final_model_path, vocab_path)
    """
    final_model_path = os.path.join(checkpoint_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': len(vocab),
    }, final_model_path)

    if model_config is not None:
        config_path = os.path.join(checkpoint_dir, 'final_model_config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(model_config, f, allow_unicode=True)

    vocab_path = os.path.join(checkpoint_dir, 'vocab.json')
    vocab_data = {
        'word2idx': vocab.word2idx,
        'idx2word': {str(k): v for k, v in vocab.idx2word.items()}
    }
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    print(f"\nTraining completed!")
    print(f"Final model saved at {final_model_path}")
    print(f"Vocabulary saved at {vocab_path}")
    return final_model_path, vocab_path