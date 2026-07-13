import os
import sys
import shutil
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast as torch_autocast, GradScaler
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import json
import random
import time
import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.transformer import TransformerModel
from models.data_utils import load_data, create_dataloader, split_dataset
from models.config_loader import build_model
from models.device import get_device, apply_cpu_threads
from models.utils import (save_checkpoint, cleanup_old_checkpoints,
                             backup_existing_checkpoints, save_final_model, cli_guard,
                             _cpu_offload)

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_lr(eff_step, total_eff, warmup_target, base_lr, eta_min, lr_schedule, wsd_decay_frac):
    """计算第 eff_step 个有效优化步的学习率（统一处理预热与各调度）。

    - warmup: 前 warmup_target 步线性升温到 base_lr
    - cosine: 之后按余弦衰减到 eta_min
    - constant (WSO): 之后保持 base_lr 不变（ICLR2026：利于后续微调的平缓极小值）
    - wsd: 之后保持 base_lr，最后 wsd_decay_frac 比例步内余弦衰减到 eta_min
    """
    if warmup_target > 0 and eff_step <= warmup_target:
        return base_lr * (eff_step / max(1, warmup_target))

    progress = (eff_step - warmup_target) / max(1, total_eff - warmup_target)
    if lr_schedule == 'constant':
        return base_lr
    if lr_schedule == 'cosine':
        return eta_min + 0.5 * (base_lr - eta_min) * (1 + math.cos(math.pi * progress))
    if lr_schedule == 'wsd':
        decay_start = 1.0 - wsd_decay_frac
        if progress >= decay_start:
            p = (progress - decay_start) / max(1e-6, wsd_decay_frac)
            return eta_min + 0.5 * (base_lr - eta_min) * (1 + math.cos(math.pi * p))
        return base_lr
    return base_lr


def train_epoch(model, dataloader, optimizer, criterion, device, epoch,
                 warmup_steps=0, base_lr=0.0005, gradient_clip=1.0, scaler=None,
                 use_amp=True, autocast_dtype=torch.float32, grad_accum_steps=1,
                 lr_schedule='cosine', eta_min=0.0, wsd_decay_frac=0.1,
                 show_progress=True, amp_device=None, enhancement_off_prob=0.0):
    """Train one epoch with warmup, gradient accumulation and mixed precision.

    - warmup_steps: 预热步数。若 <1 则按"占整个 epoch 有效步数的比例"解释（如 0.1=前 10% 步预热）。
    - grad_accum_steps: 梯度累积步数；有效 batch = batch_size * grad_accum_steps。
    - lr_schedule: cosine | constant | wsd（见 compute_lr）。
    """
    model.train()
    loss_sum = 0.0  # 用 Python float 累加，避免张量创建开销与潜在计算图残留
    loss_count = 0
    t_start = time.time()
    tokens_total = 0

    total_steps = len(dataloader)
    total_eff = (total_steps + grad_accum_steps - 1) // grad_accum_steps
    # warmup_steps 可能为小数（占 epoch 比例）或整数步数；钳制不超过总有效步数，
    # 避免误配过大预热导致全程线性升温、永不进入稳定/衰减期。
    warmup_target = min(int(warmup_steps * total_eff) if 0 < warmup_steps < 1 else int(warmup_steps), total_eff)

    optimizer.zero_grad()
    accumulated = 0
    eff_step = 0

    def step_optimizer():
        """执行一次优化器步进（含 warmup 学习率 + 梯度裁剪），循环内与 epoch 末共用，避免逻辑分叉。"""
        nonlocal eff_step, accumulated
        eff_step += 1
        lr = compute_lr(eff_step, total_eff, warmup_target, base_lr,
                        eta_min, lr_schedule, wsd_decay_frac)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
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

    progress = tqdm(dataloader, desc=f"Epoch {epoch}", total=total_steps,
                    leave=True) if (HAS_TQDM and show_progress) else None

    for batch_idx, batch in enumerate(progress if progress is not None else dataloader):
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)

        # 交替增强训练：以 enhancement_off_prob 的概率跳过本批次的增强（QK-Norm/温度/门控恒等）。
        # 关闭时增强参数本步不更新梯度；开启时正常训练。默认 0 = 始终开启。
        if enhancement_off_prob > 0.0:
            model.set_enhancements_active(random.random() >= enhancement_off_prob)

        # Forward pass (optionally under autocast for mixed precision)
        if use_amp and amp_device is not None:
            with torch_autocast(amp_device, dtype=autocast_dtype):
                logits = model(input_ids).view(-1, model.vocab_size)
                loss = criterion(logits, target_ids.view(-1))
        else:
            logits = model(input_ids).view(-1, model.vocab_size)
            loss = criterion(logits, target_ids.view(-1))

        # Scale loss for gradient accumulation, then backward
        scaled = loss / grad_accum_steps
        if scaler is not None:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        accumulated += 1

        # Only optimize every grad_accum_steps
        if accumulated % grad_accum_steps == 0:
            step_optimizer()

        # 累加损失（用 Python float，避免张量创建开销与潜在计算图残留）
        # 注意：这里累加的是未缩放的原始 loss（backward 内部已用 /grad_accum_steps 缩放梯度），
        # 不要乘以 grad_accum_steps，否则在累积步数>1 时上报值被错误地放大约 grad_accum_steps 倍。
        loss_sum += loss.detach().item()
        loss_count += 1
        tokens_total += int(input_ids.numel())

        if (batch_idx + 1) % 10 == 0:
            avg = loss_sum / loss_count
            elapsed = time.time() - t_start
            tps = tokens_total / elapsed if elapsed > 0 else 0.0
            if progress is not None:
                progress.set_postfix(loss=f"{avg:.4f}",
                                     lr=f"{optimizer.param_groups[0]['lr']:.6f}",
                                     tok_s=f"{tps:.0f}")
            else:
                print(f"Epoch {epoch} | Batch {batch_idx + 1}/{total_steps} | "
                      f"Loss: {avg:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | "
                      f"Speed: {tps:.0f} tok/s ({elapsed:.0f}s)")

    if progress is not None:
        progress.close()

    # Flush any leftover accumulated gradients
    if accumulated % grad_accum_steps != 0:
        step_optimizer()

    return (loss_sum / loss_count) if loss_count else 0.0


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    model.set_enhancements_active(True)  # 验证用增强开启模式，反映训练所得“开”行为
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





@cli_guard
def main(config_path='configs/pretrain.yaml'):
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
    
    # Split into train/validation
    test_split = config['data'].get('test_split', 0.0)
    if test_split > 0:
        train_dataset, val_dataset = split_dataset(dataset, train_ratio=1.0 - test_split,
                                                    seed=config['seed'])
        print(f"Split: train={len(train_dataset)}, val={len(val_dataset)} "
              f"(ratio {1.0-test_split:.1f}/{test_split:.1f})")
    else:
        train_dataset, val_dataset = dataset, None
    
    # Create dataloader with parallel data loading
    num_workers = config['data'].get('num_workers', 4)
    dataloader = create_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers
    )
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = create_dataloader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=num_workers
        )
    
    # Create model
    print("Creating model...")
    model = build_model(config, device=device)

    # 可选：torch.compile 加速（仅 CPU/CUDA 支持；与梯度检查点易冲突，自动关闭后者）
    if config['training'].get('compile', False) and hasattr(torch, 'compile') \
            and device.type in ('cpu', 'cuda'):
        compile_ok = True
        if device.type == 'cpu':
            # Inductor CPU 后端需要 C++ 编译器（g++/clang++/MSVC），缺失则直接跳过避免空耗
            if not any(shutil.which(c) for c in
                       ('g++.exe', 'g++', 'clang++.exe', 'clang++', 'cl.exe', 'cl')):
                compile_ok = False
                print("[提示] 未检测到 C++ 编译器，torch.compile(CPU) 不可用，已跳过（安装 MSVC/g++ 后可加速）")
        if compile_ok:
            try:
                torch._dynamo.config.suppress_errors = True  # 编译失败自动回退 eager
                model.set_gradient_checkpointing(False)
                model = torch.compile(model)
                print("torch.compile 已启用（梯度检查点已关闭；编译失败会自动回退 eager）")
            except Exception as e:
                print(f"[警告] torch.compile 初始化失败，回退普通模型: {e}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function with label smoothing and optimizer
    # Label smoothing helps prevent overconfidence and improves generalization
    # 注意：PyTorch 的 nn.CrossEntropyLoss 不支持同时使用 label_smoothing > 0 和 ignore_index
    # 因为我们使用 ignore_index=vocab.pad_idx 忽略 padding token，所以必须移除 label_smoothing
    # 若配置中设置了 label_smoothing，将在此处忽略并打印警告
    label_smoothing = config['training'].get('label_smoothing', 0.0)
    if label_smoothing > 0:
        import warnings
        warnings.warn(
            f'label_smoothing={label_smoothing} 已被忽略：'
            f'PyTorch 的 CrossEntropyLoss 不支持同时使用 label_smoothing 和 ignore_index（用于 padding）'
        )
        label_smoothing = 0.0
    
    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.pad_idx,
        # label_smoothing=config['training'].get('label_smoothing', 0.1)  # 已移除：与 ignore_index 不兼容
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

    lr_schedule = str(config['training'].get('lr_schedule', 'cosine')).lower()
    wsd_decay_frac = float(config['training'].get('wsd_decay_frac', 0.1))

    # ---- 混合精度（可选）----
    # bf16：CUDA 与 CPU 均支持（CPU 走 oneDNN bf16 matmul，可提速且动态范围大无需 loss scaling）
    # fp16：仅 NVIDIA CUDA 支持，需要 GradScaler 做 loss scaling
    # AMD DirectML（privateuseone）不支持 AMP/bf16，自动回退 fp32
    use_amp = False
    autocast_dtype = torch.float32
    scaler = None
    amp_device = None
    if precision in ('fp16', 'bf16'):
        if device.type == 'cuda':
            use_amp = True
            amp_device = 'cuda'
            autocast_dtype = torch.float16 if precision == 'fp16' else torch.bfloat16
            # bf16 动态范围大，不需要 loss scaling；fp16 才用 GradScaler
            scaler = torch.amp.GradScaler('cuda') if precision == 'fp16' else None
        elif device.type == 'cpu' and precision == 'bf16':
            # CPU bf16 混合精度（oneDNN 支持），可加速训练且无需 loss scaling
            use_amp = True
            amp_device = 'cpu'
            autocast_dtype = torch.bfloat16
            scaler = None
        else:
            print(f"[警告] precision={precision} 混合精度仅支持 CUDA(fp16/bf16) 与 CPU(bf16)；"
                  f"当前设备 {device} 不支持 AMP/bf16，自动回退 fp32 训练。")
    
    total_batches = len(dataloader)
    total_eff = (total_batches + grad_accum_steps - 1) // grad_accum_steps
    epochs = config['training']['epochs']
    print(f"\n[Training Config]")
    print(f"  Device: {device}")
    print(f"  Precision: {precision} (AMP={use_amp}, scaler={'yes' if scaler else 'no'})")
    print(f"  Batch size: {config['training']['batch_size']}  (grad_accum={grad_accum_steps}, "
          f"effective={config['training']['batch_size'] * grad_accum_steps})")
    print(f"  Steps per epoch: {total_batches} batches, {total_eff} effective (x{epochs} epochs = {total_eff * epochs} total)")
    print(f"  Num workers: {num_workers}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {config['training']['learning_rate']}  (schedule={lr_schedule}, eta_min={eta_min})")
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
            warmup_steps=config['training'].get('warmup_steps', 0),
            base_lr=config['training']['learning_rate'],
            gradient_clip=config['training']['gradient_clip'],
            scaler=scaler,
            use_amp=use_amp,
            autocast_dtype=autocast_dtype,
            amp_device=amp_device,
            grad_accum_steps=grad_accum_steps,
            lr_schedule=lr_schedule,
            eta_min=eta_min,
            wsd_decay_frac=wsd_decay_frac,
            show_progress=config['training'].get('show_progress', True),
            enhancement_off_prob=config['training'].get('enhancement_off_prob', 0.0)
        )

        history['train_loss'].append(train_loss)
        
        # Validation
        val_loss = None
        if val_dataloader is not None:
            val_loss = validate(model, val_dataloader, criterion, device)
            print(f"\nEpoch {epoch}/{config['training']['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        else:
            print(f"\nEpoch {epoch}/{config['training']['epochs']} | Train Loss: {train_loss:.4f}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Use val loss for best/early stopping if available, otherwise train loss
        epoch_loss = val_loss if val_loss is not None else train_loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            history['best_epoch'] = epoch
            no_improve_epochs = 0
            # Save per-epoch checkpoint (skipped when single epoch to avoid redundant file)
            if config['training']['epochs'] > 1:
                save_checkpoint(model, optimizer, epoch, best_loss, checkpoint_dir, len(vocab), config['model'])
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
    # CPU-offload 后再保存，确保任意设备（含 DML/CUDA）都能用 weights_only=True 加载
    torch.save({
        'model_state_dict': _cpu_offload(model.state_dict()),
        'vocab_size': len(vocab),
    }, final_model_path)
    # Save config separately for weights_only=True compatibility
    config_path = os.path.join(checkpoint_dir, 'final_model_config.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config['model'], f, allow_unicode=True)
    
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
    parser.add_argument('--config', type=str, default='configs/pretrain.yaml',
                        help='Path to config file (default: 基座模型预训练配置)')
    args = parser.parse_args()
    
    main(args.config)