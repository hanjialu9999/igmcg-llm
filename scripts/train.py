import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast as torch_autocast, GradScaler
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import json
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.transformer import TransformerModel
from models.data_utils import load_data, create_dataloader
from models.config_loader import build_model
from models.device import get_device, apply_cpu_threads

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, scheduler=None,
                warmup_steps=0, base_lr=0.0005, gradient_clip=1.0, scaler=None,
                use_amp=True, autocast_dtype=torch.float32, grad_accum_steps=1):
    """Train one epoch with warmup, gradient accumulation and mixed precision.

    - warmup_steps: 预热步数。若 <1 则按"占整个 epoch 有效步数的比例"解释（如 0.1=前 10% 步预热）。
    - grad_accum_steps: 梯度累积步数；有效 batch = batch_size * grad_accum_steps。
    - 余弦退火在每次"有效优化步"(累积满后)调用 scheduler.step()，单轮内平滑衰减。
    """
    model.train()
    loss_meter = AverageMeter()

    total_steps = len(dataloader)
    total_eff = (total_steps + grad_accum_steps - 1) // grad_accum_steps
    warmup_target = int(warmup_steps * total_eff) if 0 < warmup_steps < 1 else int(warmup_steps)

    optimizer.zero_grad()
    accumulated = 0
    eff_step = 0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)

        # Forward pass (optionally under autocast for CUDA mixed precision)
        if use_amp:
            with torch_autocast('cuda', dtype=autocast_dtype):
                logits = model(input_ids).view(-1, model.vocab_size)
                loss = criterion(logits, target_ids.view(-1))
        else:
            logits = model(input_ids).view(-1, model.vocab_size)
            loss = criterion(logits, target_ids.view(-1))

        # Scale loss for gradient accumulation, then backward
        loss = loss / grad_accum_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accumulated += 1

        # Only optimize every grad_accum_steps
        if accumulated % grad_accum_steps == 0:
            eff_step += 1

            # Warmup: linear ramp of LR over the first warmup_target effective steps
            if epoch == 1 and warmup_target > 0 and eff_step <= warmup_target:
                lr = base_lr * (eff_step / max(1, warmup_target))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            elif scheduler is not None:
                scheduler.step()

            # Gradient clipping + optimizer step
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                optimizer.step()

            optimizer.zero_grad()
            accumulated = 0

        loss_meter.update(loss.item() * grad_accum_steps)

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx + 1}/{total_steps} | "
                  f"Loss: {loss_meter.avg:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Flush any leftover accumulated gradients
    if accumulated % grad_accum_steps != 0:
        eff_step += 1
        if epoch == 1 and warmup_target > 0 and eff_step <= warmup_target:
            lr = base_lr * (eff_step / max(1, warmup_target))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        elif scheduler is not None:
            scheduler.step()
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()
        optimizer.zero_grad()

    return loss_meter.avg


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    loss_meter = AverageMeter()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            # Forward pass
            logits = model(input_ids)
            
            # Reshape for loss calculation
            logits = logits.view(-1, logits.size(-1))
            target_ids = target_ids.view(-1)
            
            # Calculate loss
            loss = criterion(logits, target_ids)
            loss_meter.update(loss.item())
    
    return loss_meter.avg


def save_checkpoint(model, optimizer, epoch, best_loss, checkpoint_dir, vocab_size):
    """Save model checkpoint"""
    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'vocab_size': vocab_size
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
    return checkpoint_path


def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=5):
    """Clean up old checkpoints, keep only the best model and last N epochs"""
    import glob
    
    # Find all epoch checkpoint files
    epoch_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'model_epoch_*.pt')))
    
    if len(epoch_files) <= keep_last_n:
        print(f"Found {len(epoch_files)} checkpoint(s). No cleanup needed (keep_last_n={keep_last_n})")
        return
    
    # Keep only the last N checkpoints
    files_to_delete = epoch_files[:-keep_last_n]
    
    deleted_count = 0
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            deleted_count += 1
            print(f"Removed: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Failed to remove {file_path}: {e}")
    
    print(f"\n✅ Cleanup complete: Deleted {deleted_count} old checkpoint(s)")
    print(f"   Kept last {keep_last_n} checkpoint(s) and final_model.pt")


def backup_existing_checkpoints(checkpoint_dir, backup_root=None):
    """训练开始前自动备份已有 checkpoints，避免覆盖旧模型。
    备份到 <项目根>/archive_unused/checkpoints_backup 下，名字冲突时自动追加数字。
    返回备份目标路径；无内容可备份时返回 None。
    """
    if backup_root is None:
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

    shutil.copytree(checkpoint_dir, dest)
    print(f"Backed up existing checkpoints -> {dest}")
    return dest


def main(config_path='config/config.yaml'):
    # Load configuration
    config = load_config(config_path)
    
    # Set seed
    set_seed(config['seed'])
    
    # Device: 自动适配 CUDA / DirectML(AMD) / CPU
    device = get_device(config.get('device', 'auto'))
    apply_cpu_threads(config['training'].get('cpu_threads'))
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = config['paths']['checkpoint_dir']

    # 训练前自动备份已有模型，避免覆盖旧 checkpoints
    backup_existing_checkpoints(checkpoint_dir)

    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    dataset, vocab = load_data(
        config['data']['train_file'],
        vocab_size=config['data']['vocab_size'],
        max_seq_length=config['data']['max_seq_length']
    )
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader with parallel data loading
    num_workers = config['data'].get('num_workers', 4)
    dataloader = create_dataloader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers
    )
    
    # Create model
    print("Creating model...")
    model = build_model(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function with label smoothing and optimizer
    # Label smoothing helps prevent overconfidence and improves generalization
    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.pad_idx,
        label_smoothing=config['training'].get('label_smoothing', 0.1)
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # ---- 精度 / 梯度累积 / 余弦退火 配置 ----
    precision = str(config['training'].get('precision', 'fp32')).lower()
    grad_accum_steps = int(config['training'].get('grad_accum_steps', 1))
    eta_min = float(config['training'].get('eta_min', 0.0))
    if grad_accum_steps < 1:
        grad_accum_steps = 1

    # 有效优化步数（用于余弦退火 T_max）
    total_eff = (len(dataloader) + grad_accum_steps - 1) // grad_accum_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=total_eff, eta_min=eta_min)

    # Mixed precision: 仅 NVIDIA CUDA 支持 torch.amp 自动混合精度(fp16/bf16)。
    # AMD DirectML / CPU 不支持 AMP，也没有 bf16 支持，自动回退 fp32。
    use_amp = False
    autocast_dtype = torch.float32
    scaler = None
    if precision in ('fp16', 'bf16'):
        if device.type == 'cuda':
            use_amp = True
            autocast_dtype = torch.float16 if precision == 'fp16' else torch.bfloat16
            # bf16 动态范围大，不需要 loss scaling；fp16 才用 GradScaler
            scaler = torch.amp.GradScaler('cuda') if precision == 'fp16' else None
        else:
            print(f"[警告] precision={precision} 仅在 NVIDIA CUDA 上支持混合精度；"
                  f"当前设备 {device} 不支持 AMP/bf16，自动回退 fp32 训练。")
    
    print(f"\n[Training Config]")
    print(f"  Device: {device}")
    print(f"  Precision: {precision} (AMP={use_amp}, scaler={'yes' if scaler else 'no'})")
    print(f"  Batch size: {config['training']['batch_size']}  (grad_accum={grad_accum_steps}, "
          f"effective={config['training']['batch_size'] * grad_accum_steps})")
    print(f"  Num workers: {num_workers}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Learning rate: {config['training']['learning_rate']}  (eta_min={eta_min})")
    print(f"  Early stop patience: {config['training'].get('early_stop_patience', 5)}")
    
    # Training loop
    print("\n[Training] Starting training...")
    best_loss = float('inf')
    history = {'train_loss': [], 'best_epoch': 0}
    no_improve_epochs = 0
    patience = config['training'].get('early_stop_patience', 5)
    
    for epoch in range(1, config['training']['epochs'] + 1):
        train_loss = train_epoch(
            model, dataloader, optimizer, criterion, device, epoch,
            scheduler=scheduler,
            warmup_steps=config['training'].get('warmup_steps', 0),
            base_lr=config['training']['learning_rate'],
            gradient_clip=config['training']['gradient_clip'],
            scaler=scaler,
            use_amp=use_amp,
            autocast_dtype=autocast_dtype,
            grad_accum_steps=grad_accum_steps
        )
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        
        print(f"\nEpoch {epoch}/{config['training']['epochs']} | Train Loss: {train_loss:.4f}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint and update best loss
        if train_loss < best_loss:
            best_loss = train_loss
            history['best_epoch'] = epoch
            no_improve_epochs = 0
            save_checkpoint(model, optimizer, epoch, best_loss, checkpoint_dir, len(vocab))
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epoch(s).")
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered after {no_improve_epochs} epochs without improvement.")
                break
        
        print("-" * 50)
    
    # Clean up old checkpoints before saving final model
    print("\n" + "="*50)
    print("Cleaning up old checkpoints...")
    print("="*50)
    cleanup_old_checkpoints(checkpoint_dir, keep_last_n=5)
    
    # Save final model and vocab
    final_model_path = os.path.join(checkpoint_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': len(vocab),
        'config': config['model']
    }, final_model_path)
    
    vocab_path = os.path.join(checkpoint_dir, 'vocab.json')
    vocab_data = {
        'word2idx': vocab.word2idx,
        'idx2word': {str(k): v for k, v in vocab.idx2word.items()}
    }
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best loss: {best_loss:.4f} (Epoch {history['best_epoch']})")
    print(f"Final model saved at {final_model_path}")
    print(f"Vocabulary saved at {vocab_path}")


if __name__ == '__main__':
    # 必须在 if __name__ == '__main__': 中调用 main()，以支持 Windows 多进程
    torch.multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/pretrain.yaml',
                        help='Path to config file (default: 基座模型预训练配置)')
    args = parser.parse_args()
    
    main(args.config)
