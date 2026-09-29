# -*- coding: utf-8 -*-
"""50 步训练剖析（本轮唯一允许的训练类运行）。

复用 `scripts/train.py:train_epoch` 的**真实训练循环**，用代理/猴子补丁做分段计时：
  - 不写任何 checkpoint（checkpoint_dir=None，checkpoint_percents=()）
  - 不改任何 config（只读 configs/config_train_8k_r42.yaml）
  - 不落盘任何模型权重

两个 pass：
  pass1 有仪器：分段计时（每段末尾强制同步，否则 DML 异步队列会把耗时挪到下一段）
  pass2 无仪器：只用 TimedLoader 量整步墙钟 → 真实吞吐（tok/s）
另加 4 项微基准，验证同步审查报告里"每步几毫秒"的具体断言。

用法（workdir=igmcg-llm）：
  python scripts/_profile_train_steps.py --steps 50 --controller on
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from models.checkpoint import build_ngram_model
from models.config_loader import build_model, load_config
from models.data_utils import load_data, create_dataloader, split_dataset
from models.device import get_device, apply_cpu_threads
from models.utils import _cpu_offload
from scripts import train as T

# ---------------------------------------------------------------- 计时器
C = defaultdict(float)          # phase -> 累计秒
LOSSES = []                     # 每步 loss（pass1）
_MODEL = [None]                 # 供同步点取参数/梯度


def _sync_tensor(t):
    """DML/CUDA 异步：读一个元素强制排空队列，否则本段耗时会被下一段吸收。"""
    if isinstance(t, torch.Tensor) and t.numel() > 0:
        t.detach().reshape(-1)[0].item()


def _sync_grads():
    for p in _MODEL[0].parameters():
        if p.grad is not None and p.grad.numel() > 0:
            p.grad.detach().reshape(-1)[0].item()
            return


def _sync_params():
    for p in _MODEL[0].parameters():
        if p.numel() > 0:
            p.detach().reshape(-1)[0].item()
            return


class TimedLoader:
    """量两段：data=取下一批的等待；loop_body=调用方处理本批的耗时（=每步墙钟）。"""

    def __init__(self, loader, limit=0):
        self.loader = loader
        self.limit = limit

    def __len__(self):
        # 必须返回完整长度：train_epoch 用它算 total_eff → LR 调度。
        # 若按 limit 截断，50 步会被当成整个 epoch，WSD 衰减在第 45 步就启动、
        # LR 掉到 0（首轮跑出现 Batch 50 LR=0.000000），偏离真实 epoch1 行为。
        return len(self.loader)

    def __iter__(self):
        it = iter(self.loader)
        done = 0
        while True:
            t_fetch = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                return
            C['data'] += time.perf_counter() - t_fetch
            t_yield = time.perf_counter()
            yield batch
            C['loop_body'] += time.perf_counter() - t_yield
            done += 1
            if self.limit and done >= self.limit:
                return


class TimedModel:
    def __init__(self, m):
        self._m = m

    def __getattr__(self, name):
        return getattr(self._m, name)

    def __call__(self, *a, **k):
        t0 = time.perf_counter()
        out = self._m(*a, **k)
        _sync_tensor(out)
        C['fwd'] += time.perf_counter() - t0
        return out


class TimedCriterion:
    def __init__(self, inner):
        self._c = inner

    def __getattr__(self, name):
        return getattr(self._c, name)

    def __call__(self, *a, **k):
        t0 = time.perf_counter()
        out = self._c(*a, **k)
        LOSSES.append(float(out))     # 记录 loss + 同步点
        C['ce'] += time.perf_counter() - t0
        return out


class TimedOptimizer:
    def __init__(self, o):
        self._o = o

    def __getattr__(self, name):
        return getattr(self._o, name)

    def step(self, *a, **k):
        t0 = time.perf_counter()
        r = self._o.step(*a, **k)
        _sync_params()
        C['opt'] += time.perf_counter() - t0
        return r

    def zero_grad(self, *a, **k):
        t0 = time.perf_counter()
        r = self._o.zero_grad(*a, **k)
        C['zero'] += time.perf_counter() - t0
        return r


_PATCHES = {}


def install_patches():
    _PATCHES['backward'] = torch.Tensor.backward

    def _timed_backward(self, *a, **k):
        t0 = time.perf_counter()
        r = _PATCHES['backward'](self, *a, **k)
        _sync_grads()
        C['bwd'] += time.perf_counter() - t0
        return r

    torch.Tensor.backward = _timed_backward

    _PATCHES['clip'] = T.clip_grad_norm_dml

    def _timed_clip(params, max_norm, **kw):
        t0 = time.perf_counter()
        r = _PATCHES['clip'](params, max_norm, **kw)
        _sync_grads()
        C['clip'] += time.perf_counter() - t0
        return r

    T.clip_grad_norm_dml = _timed_clip


def remove_patches():
    torch.Tensor.backward = _PATCHES.pop('backward')
    T.clip_grad_norm_dml = _PATCHES.pop('clip')


# ---------------------------------------------------------------- 准备
def setup(config, device, controller_on):
    """复刻 scripts/train.py:main() 的构建路径（不建目录、不备份、不 resume）。"""
    torch.manual_seed(config['seed'])
    device = get_device(config.get('device', 'auto') if device == 'auto' else device)
    apply_cpu_threads(config['training'].get('cpu_threads'))

    dataset, vocab = load_data(
        config['data']['train_file'],
        vocab_size=config['data']['vocab_size'],
        max_seq_length=config['data']['max_seq_length'])
    train_ds, _ = split_dataset(dataset, train_ratio=1.0 - config['data'].get('test_split', 0.0),
                                seed=config['seed'])
    loader = create_dataloader(train_ds, batch_size=config['training']['batch_size'],
                               shuffle=True, num_workers=config['data'].get('num_workers', 0))

    ngram = build_ngram_model(vocab, config['model'])
    model = build_model(config, device=device, ngram_model=ngram)
    model.set_enhancements_active(True)
    if getattr(model, 'controller_enabled', False):
        model._rt_controller = bool(controller_on)

    if device.type == 'privateuseone':          # 复刻 main() 的 DML lerp 补丁
        def _fl(self_list, end_list, weight):
            torch._foreach_mul_(self_list, 1 - weight)
            torch._foreach_add_(self_list, [g * weight for g in end_list])
            return self_list
        torch._foreach_lerp_ = _fl
        torch.Tensor.lerp_ = lambda self, end, weight: self.mul_(1 - weight).add_(end * weight)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    _fe = True if bool(config['training'].get('use_foreach_optimizer', False)) else None
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config['training']['learning_rate'],
                                  weight_decay=config['training']['weight_decay'],
                                  betas=(0.9, 0.999), eps=1e-8, foreach=_fe)
    return device, model, loader, criterion, optimizer, vocab


def run_epoch(model, loader, optimizer, criterion, device, config, instrumented, steps):
    C.clear()
    if instrumented:
        LOSSES.clear()
        install_patches()
        model = TimedModel(model)
        criterion = TimedCriterion(criterion)
        optimizer = TimedOptimizer(optimizer)
    try:
        t0 = time.perf_counter()
        T.train_epoch(
            model, loader, optimizer, criterion, device, epoch=1,
            warmup_steps=config['training'].get('warmup_steps', 0),
            base_lr=float(config['training']['learning_rate']),
            gradient_clip=config['training']['gradient_clip'],
            scaler=None, use_amp=False, autocast_dtype=torch.float32,
            grad_accum_steps=int(config['training'].get('grad_accum_steps', 1)),
            lr_schedule=str(config['training'].get('lr_schedule', 'cosine')).lower(),
            eta_min=float(config['training'].get('eta_min', 0.0)),
            wsd_decay_frac=float(config['training'].get('wsd_decay_frac', 0.1)),
            show_progress=False,
            complexity_lambda=float(config['training'].get('complexity_lambda', 0.0)),
            complexity_budget=config['training'].get('complexity_budget', None),
            igmcg_sel_prob=float(config['training'].get('igmcg_sel_prob', 0.0)),
            global_step=0, curriculum_total_steps=steps,
            skip_batches=0, checkpoint_dir=None, checkpoint_percents=(),
            checkpoint_meta=None, initial_eff_step=0,
            controller_active=True,
            use_foreach_norm_clip=bool(config['training'].get('use_foreach_norm_clip', False)),
        )
        wall = time.perf_counter() - t0
    finally:
        if instrumented:
            remove_patches()
    return wall


# ---------------------------------------------------------------- 微基准
def microbench(model, loader, criterion, device, vocab, n=20):
    """每项都带同样的同步点，保证 DML 异步下各项可比。"""
    out = {}
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        # train_epoch 每步末尾 zero_grad(set_to_none=True) → 梯度已被清空，
        # 这里补一次真实前向+反向把梯度造出来（只造梯度，不 step，不改权重）
        batch = next(iter(loader))
        x = batch['input_ids'].to(device)
        y = batch['target_ids'].to(device)
        model.zero_grad(set_to_none=True)
        logits = model(x, targets=y)
        criterion(logits.view(-1, logits.size(-1)), y.view(-1)).backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return {'error': 'no grads available for microbench'}

    def bench(fn, m):
        def _flush(r):
            if isinstance(r, torch.Tensor) and r.numel() > 0:
                r.detach().reshape(-1)[0].item()
            elif isinstance(r, (list, tuple)) and r and isinstance(r[0], torch.Tensor):
                r[0].detach().reshape(-1)[0].item()
            else:
                grads[0].detach().reshape(-1)[0].item()
        try:
            _flush(fn())                          # 预热 + 同步
            t0 = time.perf_counter()
            for _ in range(m):
                _flush(fn())                      # 与 fn 同步配对，DML 异步下才可比
            return round((time.perf_counter() - t0) / m * 1000.0, 4)
        except Exception as e:                    # 某算子在该后端不支持（如 DML foreach_norm）
            return f'ERR:{type(e).__name__}'

    # 1) checkpoint 用的 CPU 卸载（审查报告称 3-8ms/次）
    sd = model.state_dict()
    out['cpu_offload_state_dict_ms'] = bench(lambda: _cpu_offload(sd), max(1, n // 4))

    # 2) 梯度总范数两种算法（审查报告称 clip 的 Python 循环 3-7ms/步）
    def cur_loop():
        t = torch.zeros((), device=grads[0].device, dtype=grads[0].dtype)
        for g in grads:
            t = t + g.pow(2).sum()
        return t.sqrt()

    out['clip_pow_loop_ms'] = bench(cur_loop, n)
    out['clip_foreach_norm_ms'] = bench(lambda: torch._foreach_norm(grads, 2.0), n)

    # 3) foreach vs Python 循环（审查报告称 foreach_mul_ 有 DML 风险）
    out['foreach_mul_ms'] = bench(lambda: torch._foreach_mul_(grads, 0.5), n)
    out['python_mul_loop_ms'] = bench(lambda: [g.mul_(0.5) for g in grads], n)

    # 4) 单次 .item() 同步代价（R38/L8 关心的量）
    p0 = next(p for p in model.parameters() if p.numel() > 0)
    try:
        t0 = time.perf_counter()
        for _ in range(n * 5):
            p0.detach().reshape(-1)[0].item()
        out['sync_item_ms'] = round((time.perf_counter() - t0) / (n * 5) * 1000, 4)
    except Exception as e:
        out['sync_item_ms'] = f'ERR:{type(e).__name__}'
    out['n_grad_tensors'] = len(grads)
    return out


def main():
    ap = argparse.ArgumentParser(description='50 步训练剖析（不写 checkpoint / 不改 config）')
    ap.add_argument('--config', default='configs/config_train_8k_r42.yaml')
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--controller', choices=['on', 'off'], default='on',
                    help='on=稳态（r42 第 2/3 epoch 的形态）；off=复刻 r42 epoch1（warmup 期关 Controller）')
    ap.add_argument('--out', default='baselines/r42_profile_50steps.json')
    ap.add_argument('--foreach-clip', choices=['on', 'off'], default=None,
                    help='覆盖 config training.use_foreach_norm_clip（梯度总范数走 _foreach_norm）')
    ap.add_argument('--foreach-opt', choices=['on', 'off'], default=None,
                    help='覆盖 config training.use_foreach_optimizer（AdamW 显式 foreach=True）')
    args = ap.parse_args()

    config = load_config(args.config)
    if args.foreach_clip is not None:
        config['training']['use_foreach_norm_clip'] = (args.foreach_clip == 'on')
    if args.foreach_opt is not None:
        config['training']['use_foreach_optimizer'] = (args.foreach_opt == 'on')
    device, model, loader, criterion, optimizer, vocab = setup(
        config, args.device, args.controller == 'on')
    n_params = sum(p.numel() for p in model.parameters())
    print(f'device={device}  params={n_params:,}  controller={args.controller}')

    # 数据侧固定：只取前 steps 批
    cfg_loader = loader
    tok_per_step = config['training']['batch_size'] * config['data']['max_seq_length']

    # ---- pass1：有仪器 ----
    _MODEL[0] = model
    print(f'[pass1/2] 有仪器跑 {args.steps} 步（每段末尾强制同步）...')
    l1 = TimedLoader(cfg_loader, limit=args.steps)
    wall1 = run_epoch(model, l1, optimizer, criterion, device, config, True, args.steps)
    n1 = max(1, len(LOSSES))
    phase_ms = {k: round(v / n1 * 1000, 3) for k, v in sorted(C.items(), key=lambda kv: -kv[1])}
    print('  分段（ms/步，' + str(n1) + ' 步）：' +
          '  '.join(f'{k}={v}' for k, v in phase_ms.items()))
    print(f'  整步墙钟 {wall1 / n1 * 1000:.1f} ms/步   {tok_per_step * n1 / wall1:.0f} tok/s')
    print(f'  loss: 首={LOSSES[0]:.4f} 50步末={LOSSES[-1]:.4f}')

    # ---- pass2：无仪器（真实吞吐）----
    print(f'[pass2/2] 无仪器跑 {args.steps} 步...')
    l2 = TimedLoader(cfg_loader, limit=args.steps)
    wall2 = run_epoch(model, l2, optimizer, criterion, device, config, False, args.steps)
    n2 = args.steps
    # 仪器自身的开销 = (pass1 整步) - (pass2 整步)
    overhead_ms = wall1 / n1 * 1000 - wall2 / n2 * 1000
    print(f'  整步墙钟 {wall2 / n2 * 1000:.1f} ms/步   {tok_per_step * n2 / wall2:.0f} tok/s')
    print(f'  仪器开销 ≈ {overhead_ms:+.2f} ms/步（pass1 与 pass2 之差）')

    mb = microbench(model, cfg_loader, criterion, device, vocab)
    print('  微基准：' + '  '.join(f'{k}={v}' for k, v in mb.items()))

    result = {
        'config': args.config, 'device': str(device), 'n_params': n_params,
        'steps': args.steps, 'controller': args.controller,
        'use_foreach_norm_clip': bool(config['training'].get('use_foreach_norm_clip', False)),
        'use_foreach_optimizer': bool(config['training'].get('use_foreach_optimizer', False)),
        'batch_size': config['training']['batch_size'],
        'max_seq_length': config['data']['max_seq_length'],
        'tok_per_step': tok_per_step,
        'pass1_ms_per_step': phase_ms,
        'pass1_total_ms_per_step': round(wall1 / n1 * 1000, 3),
        'pass1_tok_s': round(tok_per_step * n1 / wall1, 1),
        'pass1_losses': [round(v, 6) for v in LOSSES],
        'pass2_total_ms_per_step': round(wall2 / n2 * 1000, 3),
        'pass2_tok_s': round(tok_per_step * n2 / wall2, 1),
        'instrument_overhead_ms_per_step': round(overhead_ms, 3),
        'microbench': mb,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'已写入 {out}')


if __name__ == '__main__':
    main()
