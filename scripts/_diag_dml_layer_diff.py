"""第六批 C：定位「DML 整段前向 ≠ CPU 整段前向」的第一处偏离模块/算子（只读诊断）。

已知事实（本轮 B 已复现，ctrl=off 口径 24 序列）：
  CPU : tf = prefix = incremental = 6.7058（cmb off）/ 6.7058（cmb on）—— 整段与前缀**自洽**
  DML : tf = 6.4490 ≠ prefix = 6.4954（ctrl off）—— 整段与前缀**不自洽**，且 tf 更"好"
        （= 整段前向像是偷看了未来信息，但只在 DML 上出现）

诊断口径（关键）：**不比 CPU vs DML 的绝对数值**（两侧 reduction 顺序不同，处处有 ~1e-5 噪声，
无法区分"正常数值差"与"真偏离"）。改成**同设备内**比：
  同一设备、同一权重、同一序列，分别做
    (a) 整段前向 x[:64]          → 记每个模块在位置 t 的输出
    (b) 前缀前向 x[:t+1]         → 记同一个模块在**最后一个位置**的输出
  因果模型下两者**必须逐位相等**（输入在 t 之前完全相同）。
  对 CPU 与 DML 各算一遍每模块的 max|diff|，**DML 显著偏离而 CPU ≈0 的第一个模块 = 出口**。

用法：
  python scripts/_diag_dml_layer_diff.py --device cpu
  python scripts/_diag_dml_layer_diff.py --device privateuseone:0 --label dml
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.checkpoint import load_model          # noqa: E402
from models.device import get_device, apply_cpu_threads  # noqa: E402
from models.config_loader import load_config      # noqa: E402
from scripts.baseline_eval import load_val_set    # noqa: E402

# 取几个代表位置（含 t=0 只看 BOS、以及靠后的位置），不必 64 个全跑
DEFAULT_TS = '0,1,7,15,31,47,63'


def _to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().to('cpu')
    if isinstance(x, (list, tuple)) and x:
        return _to_cpu(x[0])
    return None


def _slice_last(out, t):
    """把模块输出切到"位置 t"那一路；2D(B,D) 说明该模块自己聚合了时间维 → 原样返回。"""
    if out is None:
        return None
    if out.dim() == 2:            # (B, D) —— 时间维已被该模块吃掉（潜在的非因果聚合点）
        return out
    if out.dim() == 3:            # (B, L, D)
        return out[:, t, :]
    if out.dim() == 4:            # (B, H, L, D)
        return out[:, :, t, :]
    return out.reshape(out.size(0), -1)


class Recorder:
    def __init__(self, model):
        self.buf = {}
        self.handles = []
        for name, mod in model.named_modules():
            if name == '':
                continue
            self.handles.append(mod.register_forward_hook(self._mk(name)))

    def _mk(self, name):
        def hook(mod, inp, out):
            self.buf[name] = _to_cpu(out)
        return hook

    def clear(self):
        self.buf = {}

    def close(self):
        for h in self.handles:
            h.remove()


def run_pair(model, x, t, ignore_index):
    """返回 {module: tensor(位置 t 的输出)} 的两个字典：整段 / 前缀。"""
    rec = Recorder(model)
    try:
        with torch.no_grad():
            rec.clear()
            logits_full = model(x)                     # (B, L, V)
            full = {k: _slice_last(v, t) for k, v in rec.buf.items()}
            full['_logits_'] = _slice_last(logits_full.detach().float().cpu(), t)

            rec.clear()
            logits_pre = model(x[:, :t + 1])           # (B, t+1, V)
            pre = {k: _slice_last(v, t) for k, v in rec.buf.items()}
            pre['_logits_'] = _slice_last(logits_pre.detach().float().cpu(), t)
    finally:
        rec.close()

    tgt = x[:, t + 1] if t + 1 < x.size(1) else None
    return full, pre, tgt


def nll_from(logits_bt_v, tgt, ignore_index):
    if tgt is None:
        return None
    # DML 上 cross_entropy 直接报 'devices' argument must be DML，故统一挪回 CPU 算
    return float(torch.nn.functional.cross_entropy(
        logits_bt_v.cpu(), tgt.cpu(), ignore_index=ignore_index,
        reduction='mean').item())


def main():
    ap = argparse.ArgumentParser(description='DML vs CPU 整段前向逐模块偏离诊断（只读）')
    ap.add_argument('--config', default='configs/config_train_8k_r42.yaml')
    ap.add_argument('--model', default='checkpoints_train_8k_r42/final_model.pt')
    ap.add_argument('--vocab', default='checkpoints_train_8k_r42/vocab.json')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--label', default='')
    ap.add_argument('--max-seqs', type=int, default=4, help='用前 N 条 val 序列')
    ap.add_argument('--ts', default=DEFAULT_TS)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tol', type=float, default=1e-4, help='判定"显著偏离"的阈值')
    ap.add_argument('--controller', choices=['on', 'off'], default='off')
    ap.add_argument('--direction', choices=['on', 'off'], default='on')
    ap.add_argument('--char-merge-buffer', choices=['on', 'off'], default='off')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    ts = [int(v) for v in args.ts.split(',') if v.strip() != '']
    cfg = load_config(args.config)
    device = get_device(args.device)
    apply_cpu_threads(int(cfg.get('training', {}).get('cpu_threads', 0) or 0))
    torch.manual_seed(args.seed)

    label = args.label or str(device)
    print(f'[device] {label}  ({device})')

    model, vocab = load_model(args.model, args.vocab, device=device)
    model.eval()
    model.set_enhancements_active(True)
    if getattr(model, 'controller_enabled', False):
        model._rt_controller = (args.controller == 'on')
    if getattr(model, 'controller', None) is not None:
        model.controller.use_direction = (args.direction == 'on')
    if getattr(model, 'char_merge_enabled', False):
        model.char_merge.incremental_buffer = (args.char_merge_buffer == 'on')
        model.char_merge.reset_buffer()
    ignore = vocab.pad_idx
    print(f'开关：controller={args.controller} direction={args.direction} '
          f'cmb={args.char_merge_buffer}')

    val_dataset, _ = load_val_set(cfg, args.max_seqs)
    bs = int(cfg['training']['batch_size'])
    x = torch.stack([val_dataset[i]['input_ids'] for i in range(len(val_dataset))]).to(device)
    print(f'输入: {tuple(x.shape)}  (max_seqs={args.max_seqs}, batch_size={bs})')
    T = x.size(1)
    ts = [t for t in ts if t < T - 1]

    # 先打印模块清单，便于把"第一个偏离模块"落到源码
    names = [n for n, _ in model.named_modules() if n]
    print(f'\n[模块清单] 共 {len(names)} 个')
    for n in names:
        print('  ' + n)

    per_t = {}
    for t in ts:
        full, pre, tgt = run_pair(model, x, t, ignore)
        assert set(full) == set(pre), '模块集合不一致（不可能）'
        row = {}
        for k in full:
            a, b = full[k], pre[k]
            if a is None or b is None:
                row[k] = float('nan')
                continue
            row[k] = float((a - b).abs().max().item())
        row['_nll_full_'] = nll_from(full['_logits_'], tgt, ignore)
        row['_nll_prefix_'] = nll_from(pre['_logits_'], tgt, ignore)
        per_t[t] = (row, full['_logits_'], pre['_logits_'])

    # 逐模块跨 t 的最大偏离
    agg = {}
    for t, (row, _, _) in per_t.items():
        for k, v in row.items():
            if k.startswith('_nll'):
                continue
            if v != v:      # nan
                continue
            agg[k] = max(agg.get(k, 0.0), v)

    print(f'\n[同设备内] 整段前向 vs 前缀前向 —— 每模块跨 t={ts} 的 max|diff|'
          f'（tol={args.tol}）')
    print('%-58s %14s' % ('module', 'max|diff|'))
    order = sorted(agg.items(), key=lambda kv: -kv[1])
    first_bad = None
    for k, v in order:
        flag = ''
        if first_bad is None and v > args.tol:
            flag = '   <== 首个超阈值（按偏离大小排序）'
            first_bad = (k, v)
        print('%-58s %14.3e%s' % (k, v, flag))

    # 按**模块层级顺序**（named_modules 顺序）找第一个超阈值的模块 = 沿数据流的第一个出口
    seq_bad = [(k, agg[k]) for k in agg if agg[k] > args.tol]
    if seq_bad:
        first_by_order = seq_bad[0]
    else:
        first_by_order = None

    print('\n[nll 自洽性]（同一位置 t，整段 vs 前缀）')
    print('%4s %16s %16s %14s' % ('t', 'nll_full', 'nll_prefix', 'diff'))
    for t in ts:
        row, _, _ = per_t[t]
        a, b = row['_nll_full_'], row['_nll_prefix_']
        if a is None or b is None:
            print('%4d %16s %16s %14s' % (t, 'n/a', 'n/a', 'n/a'))
            continue
        print('%4d %16.6f %16.6f %+14.6f' % (t, a, b, a - b))

    print(f'\n结论[{label}]：按 named_modules 顺序，第一个超阈值模块 = '
          f'{first_by_order if first_by_order else "无（本设备整段与前缀自洽）"}')
    print(f'按偏离幅度最大 = {first_bad if first_bad else "无"}')

    if args.out:
        import json, time
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'meta': {'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                                'device': label, 'device_str': str(device),
                                'ts': ts, 'tol': args.tol, 'max_seqs': args.max_seqs,
                                'controller': args.controller,
                                'direction': args.direction,
                                'char_merge_buffer': args.char_merge_buffer,
                                'first_by_order': first_by_order,
                                'first_by_magnitude': first_bad},
                       'per_module_max_abs_diff': agg,
                       'per_t_nll': {str(t): {'full': per_t[t][0]['_nll_full_'],
                                              'prefix': per_t[t][0]['_nll_prefix_']}
                                     for t in ts}}, f, ensure_ascii=False, indent=2)
        print(f'已写入 {args.out}')


if __name__ == '__main__':
    main()
