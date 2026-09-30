"""第六批（B 加项）：逐位置 prefix loss —— 看 Controller 的作用是否随位置越往后越大。

只推理、只读（CPU），不训练不改模型代码。
两组配置（各 24 条 val 序列，`prefix` 口径 = 逐 t 只喂前缀 x[:t+1]，无未来信息）：
  A = 真·关 direction + 缓冲开   （controller=on, controller_direction=off, cmb=on）
  B = Controller 全关            （controller=off，三信号全无）
输出 64 行：位置 / A / B / 差(A−B) / 该位置有效 token 数。

用法：
  python scripts/_prefix_loss_by_pos.py --device cpu --max-seqs 24
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.checkpoint import load_model                      # noqa: E402
from models.device import get_device, apply_cpu_threads       # noqa: E402
from models.config_loader import load_config                  # noqa: E402
from scripts.baseline_eval import load_val_set                # noqa: E402

CONFIGS = [
    ('A_dir_off_cmb_on', {'controller': True, 'direction': False, 'cmb': True}),
    ('B_ctrl_off',       {'controller': False, 'direction': True, 'cmb': False}),
]


def _apply(model, sw):
    """按开关组设运行时状态（必须在 set_enhancements_active 之后）。"""
    model.eval()
    model.set_enhancements_active(True)
    if getattr(model, 'controller_enabled', False):
        model._rt_controller = sw['controller']
    if getattr(model, 'controller', None) is not None:
        model.controller.use_direction = sw['direction']
    if getattr(model, 'char_merge_enabled', False):
        model.char_merge.incremental_buffer = sw['cmb']
        model.char_merge.reset_buffer()


def eval_prefix_by_pos(model, loader, device, ignore_index):
    """逐位置累加 prefix 口径的 NLL → 返回 (nll_sum[64], valid[64])。"""
    nll = None
    cnt = None
    with torch.no_grad():
        for batch in loader:
            x = batch['input_ids'].to(device)
            y = batch['target_ids'].to(device)
            T = x.size(1)
            if nll is None:
                nll = np.zeros(T, dtype=np.float64)
                cnt = np.zeros(T, dtype=np.int64)
            for t in range(T):
                logits = model(x[:, :t + 1])
                lg = logits[:, t, :].reshape(-1, logits.size(-1))
                tgt = y[:, t]
                n = int((tgt != ignore_index).sum().item())
                if n == 0:
                    continue
                s = torch.nn.functional.cross_entropy(
                    lg, tgt, ignore_index=ignore_index, reduction='sum')
                nll[t] += float(s.item())
                cnt[t] += n
    return nll, cnt


def main():
    ap = argparse.ArgumentParser(description='逐位置 prefix loss（只读，CPU）')
    ap.add_argument('--config', default='configs/config_train_8k_r42.yaml')
    ap.add_argument('--model', default='checkpoints_train_8k_r42/final_model.pt')
    ap.add_argument('--vocab', default='checkpoints_train_8k_r42/vocab.json')
    ap.add_argument('--device', default='cpu', help='固定 cpu：三口径只在 CPU 上可比')
    ap.add_argument('--max-seqs', type=int, default=24)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device)
    apply_cpu_threads(int(cfg.get('training', {}).get('cpu_threads', 0) or 0))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f'[1/3] 加载模型 {args.model} @ {device}')
    model, vocab = load_model(args.model, args.vocab, device=device)
    model.eval()
    ignore = vocab.pad_idx

    print(f'[2/3] 加载 val 集（前 {args.max_seqs} 条）')
    val_dataset, _ = load_val_set(cfg, args.max_seqs)
    bs = int(cfg['training']['batch_size'])
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=False)
    print(f'      val 序列数={len(val_dataset)} batch_size={bs}')

    print('[3/3] 逐位置 prefix loss')
    rows = []
    for name, sw in CONFIGS:
        _apply(model, sw)
        nll, cnt = eval_prefix_by_pos(model, loader, device, ignore)
        rows.append((name, sw, nll, cnt))
        print(f'      {name}: 总 loss={nll.sum() / max(1, cnt.sum()):.6f} '
              f'(tokens={int(cnt.sum())})')

    (nA, nA_sw, a, ca), (nB, nB_sw, b, cb) = rows
    T = len(a)
    print()
    print(f'开关口径 A = {nA}: controller={nA_sw["controller"]} '
          f'controller_direction={nA_sw["direction"]} '
          f'char_merge_incremental_buffer={nA_sw["cmb"]}')
    print(f'开关口径 B = {nB}: controller={nB_sw["controller"]} '
          f'controller_direction={nB_sw["direction"]} '
          f'char_merge_incremental_buffer={nB_sw["cmb"]}')
    print(f'device={args.device}  max_seqs={args.max_seqs}  seed={args.seed}  '
          f'batch_size={bs}  mode=prefix')
    print()
    print('| 位置 | A (dir off+cmb on) | B (Controller 全关) | 差 A−B | 有效 token 数 |')
    print('|---:|---:|---:|---:|---:|')
    out_lines = []
    for i in range(T):
        la = a[i] / ca[i] if ca[i] else float('nan')
        lb = b[i] / cb[i] if cb[i] else float('nan')
        d = la - lb
        print('| %d | %.6f | %.6f | %+.6f | %d |' % (i + 1, la, lb, d, ca[i]))
        out_lines.append((i + 1, la, lb, d, int(ca[i])))
    print()
    half = np.nanmean([out_lines[i][3] for i in range(0, 32)])
    h2 = np.nanmean([out_lines[i][3] for i in range(32, 64)])
    print(f'差值均值：位置 1-32 = {half:+.6f}，位置 33-64 = {h2:+.6f}')
    print(f'差值 |diff| 最大 = {max(abs(x[3]) for x in out_lines):.6f}')

    if args.out:
        import json, time
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({
                'meta': {'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                         'device': args.device, 'max_seqs': args.max_seqs,
                         'seed': args.seed, 'batch_size': bs, 'mode': 'prefix',
                         'A': nA_sw, 'B': nB_sw},
                'rows': out_lines,
            }, f, ensure_ascii=False, indent=2)
        print(f'已写入 {args.out}')


if __name__ == '__main__':
    main()
