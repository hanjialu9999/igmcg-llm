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

def find_latest_checkpoint(checkpoint_dir: str):
    """在 checkpoint_dir 中找最新的 model_epoch_*.pt，返回 (epoch, path) 或 (0, None)。"""
    import glob as _glob
    pattern = os.path.join(checkpoint_dir, 'model_epoch_*.pt')
    files = _glob.glob(pattern)
    if not files:
        return 0, None
    # 按 epoch 编号排序取最大
    def _epoch_num(fp):
        try:
            return int(os.path.basename(fp).replace('model_epoch_', '').replace('.pt', ''))
        except ValueError:
            return 0
    files.sort(key=_epoch_num)
    latest = files[-1]
    return _epoch_num(latest), latest

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
                 show_progress=True, amp_device=None, enhancement_off_prob=0.0,
                 enhancement_schedule=None, complexity_lambda=0.0,
                 complexity_budget=None,                  curriculum_anneal=None,
                 global_step=0, curriculum_total_steps=1,
                 igmcg_sel_prob=0.0):
    """Train one epoch with warmup, gradient accumulation and mixed precision.

    - warmup_steps: 预热步数。若 <1 则按"占整个 epoch 有效步数的比例"解释（如 0.1=前 10% 步预热）。
    - grad_accum_steps: 梯度累积步数；有效 batch = batch_size * grad_accum_steps。
    - lr_schedule: cosine | constant | wsd（见 compute_lr）。
    """
    model.train()
    loss_sum = 0.0  # 初始 float，首次 += loss.detach() 后自动提升为 GPU 张量，仅打印时 .item() 同步
    loss_count = 0
    t_start = time.time()
    tokens_total = 0

    # 课程式退火（阶段8.5）：早期全增强学表示，后期按比例随机关闭指定增强（替代固定 SEL）。
    # 仅当未设 enhancement_schedule / enhancement_off_prob 时生效，向后兼容。
    cur_warmup = cur_keys = None
    cur_off_max = 0.0
    if curriculum_anneal is not None and enhancement_schedule is None and enhancement_off_prob <= 0.0:
        cur_warmup = float(curriculum_anneal.get('warmup_frac', 0.3))
        cur_off_max = float(curriculum_anneal.get('off_prob_max', 0.5))
        cur_keys = curriculum_anneal.get('keys', None)  # None=全部增强

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
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        target_ids = batch['target_ids'].to(device, non_blocking=True)

        # 交替/分段增强训练：
        #  - enhancement_schedule（分段，按开关粒度）：按 batch_idx 循环取各分段的增强掩码（dict）。
        #    每个掩码仅切换指定增强（如只动 residual_gate/hybrid_gate，qk_norm/attn_temp 恒开），
        #    关闭的增强本步不更新梯度。多段循环无额外开销（仅切几个布尔开关）。
        #  - enhancement_off_prob（整体随机）：以该概率跳过本批次全部增强。
        # 默认两者皆无 = 始终全开。
        if enhancement_schedule is not None:
            model.set_enhancements_active(enhancement_schedule[batch_idx % len(enhancement_schedule)])
        elif enhancement_off_prob > 0.0:
            model.set_enhancements_active(random.random() >= enhancement_off_prob)
        elif cur_keys is not None or cur_warmup is not None:
            # 课程退火：frac∈[0,1] 当前训练进度；warmup 内恒开，之后 off 概率线性升到 off_prob_max。
            frac = (global_step + batch_idx) / max(1, curriculum_total_steps)
            if frac < cur_warmup:
                model.set_enhancements_active(True)
            else:
                p_off = cur_off_max * (frac - cur_warmup) / max(1e-6, 1.0 - cur_warmup)
                if random.random() < p_off:
                    model.set_enhancements_active(
                        {k: False for k in cur_keys} if cur_keys else False)
                else:
                    model.set_enhancements_active(True)

        # 阶段8.7 IGMCG-SEL：训练期以 igmcg_sel_prob 概率整批强制关闭 IGMCG 引导（igate 归零），
        # 让模型学会"何时依赖 IGMCG、何时靠自身"——否则 use 门控恒开、无自决意义。
        _igmcg_off = igmcg_sel_prob > 0.0 and random.random() < igmcg_sel_prob
        # Forward pass (optionally under autocast for mixed precision)
        if use_amp and amp_device is not None:
            with torch_autocast(amp_device, dtype=autocast_dtype):
                logits = model(input_ids, igmcg_force_off=_igmcg_off).view(-1, model.vocab_size)
                loss = criterion(logits, target_ids.view(-1))
        else:
            logits = model(input_ids, igmcg_force_off=_igmcg_off).view(-1, model.vocab_size)
            loss = criterion(logits, target_ids.view(-1))
        # 阶段8.2：复杂度约束（正则项）——把"小模型/提速"从弱乘奖励升级为预算硬约束。
        #  - 旧式弱乘：complexity_lambda>0 且未设 budget → loss += λ·comp（λ=1e-4，量级可忽略）。
        #  - 新式 hinge 预算：设 complexity_budget∈(0,1]（相对 max_complexity 的目标占比）→
        #    仅当 comp 超过 target 才惩罚 relu(comp-target)，梯度只在超预算时生效，
        #    且 λ 可设大（如 0.01~0.1），驱动 skip_gate/mixer/learn_window 真正压到低复杂度。
        if complexity_lambda and complexity_lambda > 0:
            comp = model.compute_complexity()
            if complexity_budget is not None and complexity_budget > 0:
                target = float(complexity_budget) * model.max_complexity()
                over = torch.relu(comp - target)
                loss = loss + complexity_lambda * over
            else:
                loss = loss + complexity_lambda * comp

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

        # 累加损失：始终用 GPU 张量累加，仅在打印时才 .item() 同步
        loss_sum = loss_sum + loss.detach()
        loss_count += 1
        tokens_total += int(input_ids.numel())

        if (batch_idx + 1) % 10 == 0:
            avg = (loss_sum / loss_count).item()  # 仅此处同步 DML→CPU
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

    return (loss_sum / loss_count).item() if loss_count else 0.0


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
def main(config_path='configs/pretrain.yaml', resume=False):
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
    # 阶段8.1：n-gram 神经融合——用训练语料（与模型训练同一份分布）构建统计 n-gram 缓冲，
    # 传入 build_model 供可学习门控融合（缺省关；开启时 ngram_fusion=True 且 ngram_corpus 指向语料）。
    _ngram_model = None
    if config['model'].get('ngram_fusion', False):
        try:
            from scripts.generate import NGramModel
            _ngram_corpus = config['model'].get('ngram_corpus', 'data/pretrain_corpus/merged.txt')
            # vocab_size 对齐模型词表（config 里可能远大于语料实际覆盖的 token 数），
            # 否则 logprob_matrix 维度与 output_head 不匹配会广播失败。
            _ngram_vocab_size = config['model'].get('vocab_size', getattr(vocab, 'vocab_size', None))
            _ngram_model = NGramModel(vocab, _ngram_corpus, max_order=10, smoothing=1.0,
                                      vocab_size=_ngram_vocab_size)
            print(f"[n-gram 融合] 已从 {_ngram_corpus} 构建统计 n-gram 缓冲（与训练分布对齐）")
        except Exception as e:
            print(f"[n-gram 融合] 构建失败，已跳过：{e}")
    model = build_model(config, device=device, ngram_model=_ngram_model)

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
    # 优化器工厂：支持 DML 友好的 SGD（避免 AdamW 的 CPU lerp 回退税）
    # 配置键：training.optimizer ∈ {adamw(默认), sgd, adam}；sgd 另读 training.momentum(默认0.9)
    opt_name = str(config['training'].get('optimizer', 'adamw')).lower()
    if opt_name == 'sgd':
        # SGD 学习率量级远大于 AdamW，未显式配置时给一个合理的字符级 LM 默认值
        sgd_lr = float(config['training'].get('sgd_learning_rate', config['training']['learning_rate']))
        momentum = float(config['training'].get('momentum', 0.9))
        optimizer = optim.SGD(
            model.parameters(),
            lr=sgd_lr,
            momentum=momentum,
            weight_decay=config['training']['weight_decay'],
        )
        print(f"Optimizer: SGD(lr={sgd_lr}, momentum={momentum})  [DML GPU-native, 无 CPU lerp 税]")
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        print(f"Optimizer: Adam(lr={config['training']['learning_rate']})")
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        print(f"Optimizer: AdamW(lr={config['training']['learning_rate']})")

    # 调度基准 lr 须与优化器实际初始 lr 一致：SGD 用 sgd_learning_rate，否则用 learning_rate
    opt_base_lr = (float(config['training'].get('sgd_learning_rate', config['training']['learning_rate']))
                   if opt_name == 'sgd' else float(config['training']['learning_rate']))

    # ---- 续训（resume）：加载最新 checkpoint 恢复训练 ----
    start_epoch = 1
    best_loss = float('inf')
    _resume_scaler_state = None
    if resume:
        resume_epoch, resume_path = find_latest_checkpoint(checkpoint_dir)
        if resume_path is not None:
            print(f"\n[Resume] 从 {resume_path} (epoch {resume_epoch}) 续训")
            ckpt = torch.load(resume_path, map_location='cpu', weights_only=True)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            # optimizer.load_state_dict 直接赋值 CPU 张量，需迁移至训练设备，否则首步 optimizer.step 崩溃
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            best_loss = ckpt.get('best_loss', float('inf'))
            start_epoch = resume_epoch + 1
            # 暂存 scaler state，待 scaler 创建后恢复（resume 块先于 scaler 创建）
            _resume_scaler_state = ckpt.get('scaler_state_dict', None)
            print(f"[Resume] best_loss={best_loss:.4f}, 从 epoch {start_epoch} 继续")
        else:
            print("[Resume] 未找到 checkpoint，从头开始训练")

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
    # 恢复 GradScaler state（fp16 resume 时 scaler scale 已调整，丢失会导致梯度溢出）
    if scaler is not None and _resume_scaler_state is not None:
        scaler.load_state_dict(_resume_scaler_state)
    
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

    # 交替/分段增强训练配置：
    #  - enhancement_schedule：分段掩码列表（dict），按 batch 循环切换；缺省键补 True（恒开）。
    #  - enhancement_off_prob：整体随机关闭概率（旧式交替，与 schedule 互斥，schedule 优先）。
    enhancement_schedule = config['training'].get('enhancement_schedule')
    enhancement_off_prob = config['training'].get('enhancement_off_prob', 0.0)
    if enhancement_schedule is not None:
        _full_keys = ["qk_norm", "attn_temp", "residual_gate", "hybrid_gate"]
        enhancement_schedule = [{**{k: True for k in _full_keys}, **m}
                                for m in enhancement_schedule]
        print(f"  Enhancement schedule: {len(enhancement_schedule)} 段分段（按开关粒度交替）")
    else:
        if enhancement_off_prob > 0:
            print(f"  Enhancement off-prob: {enhancement_off_prob}（整体随机交替）")
    # 课程式退火（阶段8.5）：替代固定 SEL。早期全增强，后期按进度随机关闭指定增强。
    curriculum_anneal = config['training'].get('curriculum_anneal')
    if curriculum_anneal is not None:
        print(f"  Curriculum anneal: {curriculum_anneal}（课程退火替代 SEL 交替）")
    # 互斥校验：enhancement_schedule / enhancement_off_prob / curriculum_anneal 三者只能其一，
    # 否则后者静默覆盖前者（train_epoch 内优先级 schedule > off_prob > curriculum）。主动告警避免误配。
    _n_set = sum([
        enhancement_schedule is not None,
        enhancement_off_prob > 0,
        curriculum_anneal is not None
    ])
    if _n_set > 1:
        print("[warn] 训练增强策略配置冲突：enhancement_schedule / enhancement_off_prob / "
              "curriculum_anneal 同时设置，仅按优先级（schedule > off_prob > curriculum）生效其一，"
              "其余被忽略。建议只保留一个。")
    total_steps_all = total_batches * config['training']['epochs']

    # Training loop
    print("\n[Training] Starting training...")
    history = {'train_loss': [], 'best_epoch': 0}
    no_improve_epochs = 0
    patience = config['training'].get('early_stop_patience', 5)
    global_step = 0  # 跨 epoch 累计步数，供课程退火计算训练进度

    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        train_loss = train_epoch(
            model, dataloader, optimizer, criterion, device, epoch,
            warmup_steps=config['training'].get('warmup_steps', 0),
            base_lr=opt_base_lr,
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
            enhancement_off_prob=config['training'].get('enhancement_off_prob', 0.0),
            enhancement_schedule=enhancement_schedule,
            complexity_lambda=float(config['training'].get('complexity_lambda', 0.0)),
            complexity_budget=config['training'].get('complexity_budget', None),
            curriculum_anneal=curriculum_anneal,
            igmcg_sel_prob=float(config['training'].get('igmcg_sel_prob', 0.0)),
            global_step=global_step,
            curriculum_total_steps=total_steps_all,
        )
        global_step += total_batches

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
                save_checkpoint(model, optimizer, epoch, best_loss, checkpoint_dir, len(vocab), config['model'], scaler=scaler)
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
    parser.add_argument('--resume', action='store_true',
                        help='从 checkpoint_dir 中最新的 checkpoint 续训（恢复模型/优化器/best_loss/epoch）')
    args = parser.parse_args()
    
    main(args.config, resume=args.resume)