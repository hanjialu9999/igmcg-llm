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
from models.config_loader import build_model, load_config
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


def find_step_checkpoint(checkpoint_dir: str):
    """找最新的 step 级 checkpoint（checkpoint_step*_pct.pt），返回 path 或 None。

    step checkpoint 用于全量训练中断续训：在 25%/50%/80% 进度时保存，
    比 epoch checkpoint 粒度更细，恢复时从对应 batch 续训而非整个 epoch 重来。
    """
    import glob as _glob
    pattern = os.path.join(checkpoint_dir, 'checkpoint_step*pct.pt')
    files = _glob.glob(pattern)
    if not files:
        return None
    # 按修改时间取最新
    files.sort(key=lambda f: os.path.getmtime(f))
    return files[-1]


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

    # R38: clamp progress ≤ 1.0——eff_step 超 total_eff（resume 后数据规模变化）时，
    # cosine/wsd 的 cos(π·progress) 周期回升，LR 从 eta_min 重新爬升（静默 bug）。
    progress = min(1.0, (eff_step - warmup_target) / max(1, total_eff - warmup_target))
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


def clip_grad_norm_dml(params, max_norm):
    """GPU 侧梯度裁剪（R38）：全程在设备上计算，无 .item() CPU 同步。

    torch.nn.utils.clip_grad_norm_ 内部 total_norm.item() 每优化步同步一次
    （DML 上 ~0.5-2ms/步）。本实现数学等价：
      scale = min(max_norm / (total_norm + 1e-6), 1)，梯度 *= scale
    与 clip_grad_norm_ 默认 (eps=1e-6, error_if_nonfinite=False) 语义一致：
    非有限 total_norm 时 scale 为 NaN，NaN < 1 为 False → 不缩放（同原版）。
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return None
    total_sq = torch.zeros((), device=grads[0].device, dtype=grads[0].dtype)
    for g in grads:
        total_sq = total_sq + g.pow(2).sum()
    total_norm = total_sq.sqrt()
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = clip_coef.clamp(max=1.0)
    if clip_coef < 1.0:
        torch._foreach_mul_(grads, clip_coef)
    return total_norm


def train_epoch(model, dataloader, optimizer, criterion, device, epoch,
                 warmup_steps=0, base_lr=0.0005, gradient_clip=1.0, scaler=None,
                 use_amp=True, autocast_dtype=torch.float32, grad_accum_steps=1,
                 lr_schedule='cosine', eta_min=0.0, wsd_decay_frac=0.1,
                 show_progress=True, amp_device=None, enhancement_off_prob=0.0,
                 enhancement_schedule=None, complexity_lambda=0.0,
                 complexity_budget=None,                  curriculum_anneal=None,
                 global_step=0, curriculum_total_steps=1,
                 igmcg_sel_prob=0.0,
                 skip_batches=0, checkpoint_dir=None, checkpoint_percents=(),
                 checkpoint_meta=None, initial_eff_step=0):
    """Train one epoch with warmup, gradient accumulation and mixed precision.

    - warmup_steps: 预热步数。若 <1 则按"占整个 epoch 有效步数的比例"解释（如 0.1=前 10% 步预热）。
    - grad_accum_steps: 梯度累积步数；有效 batch = batch_size * grad_accum_steps。
    - lr_schedule: cosine | constant | wsd（见 compute_lr）。
    - skip_batches: 跳过前 N 个 batch（step checkpoint resume 时续训用）。
    - checkpoint_dir: step checkpoint 保存目录。
    - checkpoint_percents: 在指定进度（如 (0.25, 0.5, 0.8)）保存 step checkpoint。
    - checkpoint_meta: dict 含 scaler 等，传给 step checkpoint 保存。
    - initial_eff_step: resume 续训时已完成的累计有效优化步（R38 修复：此前从 0 重计，
      resume 后 warmup/wsd LR 调度从头爬升，与"不中断训练"不等价）。
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

    # R38: set_to_none=True——梯度置 None 而非零填充（省 memset 拷贝，无数值影响）
    optimizer.zero_grad(set_to_none=True)
    accumulated = 0
    eff_step = initial_eff_step

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
            clip_grad_norm_dml(model.parameters(), max_norm=gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            clip_grad_norm_dml(model.parameters(), max_norm=gradient_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0

    progress = tqdm(dataloader, desc=f"Epoch {epoch}", total=total_steps,
                    initial=skip_batches, leave=True) if (HAS_TQDM and show_progress) else None

    # 创建 iterator，跳过已训 batch（step checkpoint resume 时续训）
    dataloader_iter = iter(progress) if progress is not None else iter(dataloader)
    if skip_batches > 0:
        print(f"  [Resume] 跳过前 {skip_batches} batch...")
        for _ in range(skip_batches):
            next(dataloader_iter, None)

    # 已保存的 step checkpoint 集合（避免重复保存）
    _saved_checkpoints = set()

    for batch_idx, batch in enumerate(dataloader_iter):
        actual_idx = skip_batches + batch_idx
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        target_ids = batch['target_ids'].to(device, non_blocking=True)

        # 交替/分段增强训练：
        #  - enhancement_schedule（分段，按开关粒度）：按 batch_idx 循环取各分段的增强掩码（dict）。
        #    每个掩码仅切换指定增强（如只动 residual_gate/hybrid_gate，qk_norm/attn_temp 恒开），
        #    关闭的增强本步不更新梯度。多段循环无额外开销（仅切几个布尔开关）。
        #  - enhancement_off_prob（整体随机）：以该概率跳过本批次全部增强。
        # 默认两者皆无 = 始终全开。
        if enhancement_schedule is not None:
            # R38: 用 actual_idx（含 skip_batches）而非 batch_idx——resume 续训时
            # batch_idx 从 0 重计，用它会相位错位（与不中断训练不一致）。
            model.set_enhancements_active(enhancement_schedule[actual_idx % len(enhancement_schedule)])
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
        # R36-5: 训练期传入 targets 用于出口层辅助 CE 损失（early_exit 开启时生效；
        #         关闭时 forward 内 targets 被忽略，无开销）
        if use_amp and amp_device is not None:
            with torch_autocast(amp_device, dtype=autocast_dtype):
                logits = model(input_ids, igmcg_force_off=_igmcg_off, targets=target_ids).view(-1, model.vocab_size)
                loss = criterion(logits, target_ids.view(-1))
        else:
            logits = model(input_ids, igmcg_force_off=_igmcg_off, targets=target_ids).view(-1, model.vocab_size)
            loss = criterion(logits, target_ids.view(-1))
        # 第十四轮：层间对比绑定辅助损失——训练期 model._contrastive_loss 累积相邻层 (1-cos_sim)，
        # 以小权重（0.01）加入主 loss，防止深层过度偏离浅层特征。eval 时不计算（_contrastive_loss=None）。
        _cl = getattr(model, '_contrastive_loss', None)
        if _cl is not None:
            loss = loss + 0.01 * _cl
        # R36-5: 提前退出辅助损失——训练期 model._early_exit_aux_loss 累积出口层加权 CE，
        # 以 early_exit_loss_weight 加入主 loss（浅层出口权重大，鼓励浅层学到判别性）。
        _ee = getattr(model, '_early_exit_aux_loss', None)
        if _ee is not None:
            loss = loss + model.early_exit_loss_weight * _ee
        # R36-6: MoE 辅助损失——负载均衡（Switch Transformer）+ router z-loss（ST-MoE）。
        # 训练期 model._moe_load_balance_loss / _moe_z_loss 跨层累积，按各自权重加入主 loss。
        _mlb = getattr(model, '_moe_load_balance_loss', None)
        if _mlb is not None:
            loss = loss + model.moe_load_balance_weight * _mlb
        _mzl = getattr(model, '_moe_z_loss', None)
        if _mzl is not None:
            loss = loss + model.moe_router_z_loss_weight * _mzl
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
        # R38: .float() 保证 fp32 累加（bf16/fp16 下长 epoch 尾数不漂移）
        loss_sum = loss_sum + loss.detach().float()
        loss_count += 1
        tokens_total += int(input_ids.numel())

        if (actual_idx + 1) % 10 == 0 or actual_idx + 1 == total_steps:
            avg = (loss_sum / loss_count).item()  # 仅此处同步 DML→CPU
            elapsed = time.time() - t_start
            tps = tokens_total / elapsed if elapsed > 0 else 0.0
            if progress is not None:
                progress.set_postfix(loss=f"{avg:.4f}",
                                     lr=f"{optimizer.param_groups[0]['lr']:.6f}",
                                     tok_s=f"{tps:.0f}")
            else:
                print(f"Epoch {epoch} | Batch {actual_idx + 1}/{total_steps} | "
                      f"Loss: {avg:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | "
                      f"Speed: {tps:.0f} tok/s ({elapsed:.0f}s)")

        # ---- Step 级 checkpoint：在指定进度（25%/50%/80%）保存，支持中断续训 ----
        # 全量训练动辄数十小时，仅 epoch 末保存风险过高（OOM/断电/Windows 更新会丢整个 epoch）
        if checkpoint_dir and checkpoint_percents:
            _progress_frac = (actual_idx + 1) / total_steps
            for _pct in checkpoint_percents:
                _threshold = int(_pct * total_steps)
                if actual_idx + 1 == _threshold and _pct not in _saved_checkpoints:
                    _saved_checkpoints.add(_pct)
                    _label = f"step{int(_pct * 100)}pct"
                    _ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{_label}.pt')
                    _save_dict = {
                        'epoch': epoch,
                        'batch_idx': actual_idx + 1,
                        'global_step': global_step + actual_idx + 1,
                        'model_state_dict': _cpu_offload(model.state_dict()),
                        'optimizer_state_dict': _cpu_offload(optimizer.state_dict()),
                        'progress_frac': _progress_frac,
                        # R38: 记录 best_loss，resume 时恢复（此前 resume 后 best_loss
                        # 被重置为 inf，训练完成后会丢弃一个更优的模型）
                        'best_loss': checkpoint_meta.get('best_loss', float('inf')) if checkpoint_meta else float('inf'),
                    }
                    if checkpoint_meta:
                        _scaler_ref = checkpoint_meta.get('scaler')
                        if _scaler_ref is not None:
                            _save_dict['scaler_state_dict'] = _scaler_ref.state_dict()
                    torch.save(_save_dict, _ckpt_path)
                    print(f"\n  [Checkpoint] {_label} saved ({_progress_frac:.1%}) at {_ckpt_path}")

    if progress is not None:
        progress.close()

    # Flush any leftover accumulated gradients (rescale if partial bucket)
    if accumulated % grad_accum_steps != 0:
        actual_accum = accumulated % grad_accum_steps
        if actual_accum != grad_accum_steps:
            # 梯度被除以了 grad_accum_steps，但只有 actual_accum 步贡献，需修正比例
            scale = grad_accum_steps / actual_accum
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
        step_optimizer()

    return (loss_sum / loss_count).item() if loss_count else 0.0


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    model.set_enhancements_active(True)  # 验证用增强开启模式，反映训练所得"开"行为
    # GPU 累加 loss，仅在末尾 .item() 同步一次（避免每 batch 的 DML→CPU 同步税）
    loss_sum = torch.zeros(1, device=device)
    loss_count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            target_ids = batch['target_ids'].to(device, non_blocking=True)

            # Forward pass
            logits = model(input_ids)

            # Reshape for loss calculation
            logits = logits.view(-1, logits.size(-1))
            target_ids = target_ids.view(-1)

            # Calculate loss
            loss = criterion(logits, target_ids)
            loss_sum = loss_sum + loss.detach()
            loss_count += 1

    return (loss_sum / max(1, loss_count)).item()





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

    # 词表一致性防护：模型输出词表必须覆盖分词器实际词表大小，否则 embedding / CE 会因
    # token id 越界崩溃或静默错乱（data.vocab_size 与 model.vocab_size 是两个独立配置键，
    # 从不交叉校验，过去仅靠约定一致，属潜伏陷阱）。
    _model_vocab = config['model']['vocab_size']
    if _model_vocab < len(vocab):
        raise ValueError(
            f"model.vocab_size={_model_vocab} 小于分词器词表大小 {len(vocab)}，token id 将越界。"
            f"请保证 data.vocab_size <= model.vocab_size"
            f"（当前 data.vocab_size={config['data']['vocab_size']}）。")
    
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
    # 构建逻辑统一收敛到 models.checkpoint.build_ngram_model，避免与 generate.py 重复实现。
    from models.checkpoint import build_ngram_model, safe_torch_load
    _ngram_model = build_ngram_model(vocab, config['model'])
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

    # 第十一轮：量化感知训练（QAT）——config.model.qat_bits>0 时启用伪量化，
    # 让模型在训练时学适应量化噪声，便于后续低比特部署。eval 时自动恒等。
    _qat_bits = int(config.get('model', {}).get('qat_bits', 0) or 0)
    if _qat_bits > 0:
        from models.qat import enable_qat
        enable_qat(model, bits=_qat_bits)
        print(f"[QAT] 已启用 {_qat_bits}bit 量化感知训练（权重+激活双量化，eval 恒等）")

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
    # DML 兼容：AdamW/Adam 内部用 torch._foreach_lerp_ 更新 exp_avg，
    # aten::lerp.Scalar_out 不支持 DML，每步回退 CPU + DML↔CPU 数据搬运（严重性能杀手，
    # 实测每步 ~206ms 额外开销，占总训练时间 ~23%）。
    # 数学等价替换：foreach_lerp_(self_list, end_list, w) =
    #   for each i: self_list[i] = (1-w)*self_list[i] + w*end_list[i]
    #   → foreach_mul_(self_list, 1-w) + foreach_add_(self_list, [g*w for g in end_list])
    # 注意：
    #   1) 必须同时 patch torch._foreach_lerp_（AdamW 实际调用）和 torch.Tensor.lerp_
    #      （单张量 fallback 路径），缺一不可。
    #   2) 不能用 add_(end, alpha=w)——DML 的 add_ alpha 参数有 bug（曾致 val_loss 16→7.5）。
    #   3) torch._foreach_mul_ / torch._foreach_add_ 在 DML 上原生支持（已验证）。
    #   4) 模型 forward 无 lerp_ 调用（已 grep 确认），全局 patch 仅影响优化器，安全。
    if device.type == 'privateuseone' and opt_name in ('adamw', 'adam'):
        def _dml_foreach_lerp(self_list, end_list, weight):
            # foreach 版：批量 mul_ + add_，避免 Python 循环开销
            torch._foreach_mul_(self_list, 1 - weight)
            torch._foreach_add_(self_list, [g * weight for g in end_list])
            return self_list
        _orig_foreach_lerp = torch._foreach_lerp_
        torch._foreach_lerp_ = _dml_foreach_lerp
        # 同时 patch 单张量版（部分代码路径可能用单张量 lerp_）
        _orig_lerp = torch.Tensor.lerp_
        def _dml_lerp(self, end, weight):
            return self.mul_(1 - weight).add_(end * weight)
        torch.Tensor.lerp_ = _dml_lerp
        print(f"[DML] 已 monkey-patch _foreach_lerp_ + lerp_ → mul_+add_（避免 AdamW/Adam CPU 回退税）")
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
    # 优先找 step checkpoint（更细粒度，支持 epoch 中间续训），没有再找 epoch checkpoint
    start_epoch = 1
    best_loss = float('inf')
    _resume_scaler_state = None
    resume_skip_batches = 0  # step checkpoint 恢复时跳过的 batch 数
    resume_global_step = 0   # R38: 恢复累计 global_step，保持课程退火/checkpoint 编号连续
    if resume:
        step_ckpt_path = find_step_checkpoint(checkpoint_dir)
        if step_ckpt_path is not None:
            # step checkpoint：从 epoch 中间续训（如 50% 处中断 → 从 50% 继续）
            print(f"\n[Resume] 从 step checkpoint {step_ckpt_path} 续训")
            ckpt = safe_torch_load(step_ckpt_path, map_location='cpu')
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            start_epoch = ckpt['epoch']  # 当前 epoch 还没训完
            resume_skip_batches = ckpt['batch_idx']
            # R38: 此前 best_loss 被重置为 inf，训练完成后把更优模型当"新 best"保存
            best_loss = ckpt.get('best_loss', float('inf'))
            resume_global_step = ckpt.get('global_step', 0)
            _resume_scaler_state = ckpt.get('scaler_state_dict', None)
            print(f"[Resume] epoch {start_epoch}, 跳过前 {resume_skip_batches} batch (进度 {ckpt.get('progress_frac', 0):.1%}), best_loss={best_loss:.4f}")
        else:
            # 回退到 epoch checkpoint（整个 epoch 已训完，从下一个 epoch 开始）
            resume_epoch, resume_path = find_latest_checkpoint(checkpoint_dir)
            if resume_path is not None:
                print(f"\n[Resume] 从 epoch checkpoint {resume_path} (epoch {resume_epoch}) 续训")
                ckpt = safe_torch_load(resume_path, map_location='cpu')
                model.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
                best_loss = ckpt.get('best_loss', float('inf'))
                resume_global_step = ckpt.get('global_step', 0)
                start_epoch = resume_epoch + 1
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
        _full_keys = list(TransformerModel.ENHANCEMENT_KEYS)
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
    # R38: resume 后 global_step 从恢复值继续（此前从 0 重计，课程退火相位错位）
    global_step = resume_global_step  # 跨 epoch 累计步数，供课程退火计算训练进度

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
            # Step 级 checkpoint：在指定进度保存，支持全量训练中断续训
            skip_batches=resume_skip_batches if epoch == start_epoch else 0,
            checkpoint_dir=checkpoint_dir,
            checkpoint_percents=tuple(config['training'].get('checkpoint_percents', [])),
            checkpoint_meta={'scaler': scaler, 'best_loss': best_loss},
            # R38: resume 后 LR 调度从已完成的累计步继续（此前 warmup/衰减从头爬升）
            initial_eff_step=resume_skip_batches // grad_accum_steps if epoch == start_epoch else 0,
        )
        global_step += (total_batches - resume_skip_batches) if epoch == start_epoch else total_batches
        # R39 修复：resume 轮实际只训了 (total_batches - skip_batches) 批却记满
        # total_batches → 课程退火 frac 超前 skip 批，重复 resume 累计漂移。

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
                save_checkpoint(model, optimizer, epoch, best_loss, checkpoint_dir, len(vocab), config['model'], scaler=scaler, global_step=global_step)
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epoch(s).")
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered after {no_improve_epochs} epochs without improvement.")
                break
        
        print("-" * 50)
    
    # R39 修复：清理挪到 save_final_model 之后（此前先删 step ckpt 再保存 final——
    # 若 final save 失败/崩溃，step ckpt 已删且无 final 模型，无可恢复点）。
    # Save final model and vocab
    # CPU-offload 后再保存，确保任意设备（含 DML/CUDA）都能用 weights_only=True 加载
    final_model_path, vocab_path = save_final_model(
        model, vocab, checkpoint_dir, config['model'])

    # Clean up old checkpoints after final save succeeded
    print("\n" + "="*50)
    print("Cleaning up old checkpoints...")
    print("="*50)
    cleanup_old_checkpoints(checkpoint_dir, keep_last_n=5)
    # R38 修复：训练已完成，删除 step 级 checkpoint。
    # 此前 step ckpt 残留，resume 时优先加载它（比 final 更旧）→ 已训完的模型被回滚。
    # 训练中断（OOM/断电）时 step ckpt 仍保留，可正常续训。
    import glob as _glob
    _step_ckpts = _glob.glob(os.path.join(checkpoint_dir, 'checkpoint_step*pct.pt'))
    for _p in _step_ckpts:
        os.remove(_p)
    if _step_ckpts:
        print(f"[Cleanup] 删除 {len(_step_ckpts)} 个已完成训练的 step checkpoint")

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