"""第六批 C（算子级）：把「DML 整段前向 ≠ 前缀前向」的偏离钉死到 SDPA 这一个算子上。

上一步（_diag_dml_layer_diff.py）已定位：DML 上偏离**只出现在位置 t=0**，
且沿模块链 embedding→char_merge→blocks.0.attn.{qkv,rope,qk_norm,output_gate} 全 ≈0，
**第一个跳变点是 blocks.0.attn.proj 的输入 = scaled_dot_product_attention 的输出**。

本脚本做法：运行时包装 `scaled_dot_product_attention`（不改模型源码），把真实前向里
SDPA 收到的 q/k/v/attn_mask 和产出的 out 原样录下来，然后
  (1) 比整段(T=64) vs 前缀(T=1) 在**位置 0** 的 q/k/v/mask/out；
  (2) 把录到的张量搬到 CPU 再算一遍 SDPA，判断偏离是「掩码长度不同」还是「DML 算子本身」。

用法：
  python scripts/_diag_sdpa_dml.py --device dml
  python scripts/_diag_sdpa_dml.py --device cpu
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import models.mixers as mixers_mod                       # noqa: E402
import models.transformer as transformer_mod             # noqa: E402
from models.checkpoint import load_model                 # noqa: E402
from models.device import get_device, apply_cpu_threads  # noqa: E402
from models.config_loader import load_config             # noqa: E402
from scripts.baseline_eval import load_val_set           # noqa: E402

_orig = None
_rec = []


def _wrap(q, k, v, attn_mask=None, is_causal=False, **kw):
    out = _orig(q, k, v, attn_mask=attn_mask, is_causal=is_causal, **kw)
    _rec.append({
        'q': q.detach().float().cpu(),
        'k': k.detach().float().cpu(),
        'v': v.detach().float().cpu(),
        'mask': attn_mask.detach().float().cpu() if attn_mask is not None else None,
        'out': out.detach().float().cpu(),
        'is_causal': is_causal,
    })
    return out


def _patch():
    global _orig
    _orig = mixers_mod.scaled_dot_product_attention
    mixers_mod.scaled_dot_product_attention = _wrap
    # transformer.py 也单独 import 了一份，一并覆盖
    if hasattr(transformer_mod, 'scaled_dot_product_attention'):
        transformer_mod.scaled_dot_product_attention = _wrap


def _unpatch():
    mixers_mod.scaled_dot_product_attention = _orig
    if hasattr(transformer_mod, 'scaled_dot_product_attention'):
        transformer_mod.scaled_dot_product_attention = _orig


def _f(x):
    return f'{x:.6e}'


def main():
    ap = argparse.ArgumentParser(description='SDPA 算子级 DML 偏离定位（只读）')
    ap.add_argument('--config', default='configs/config_train_8k_r42.yaml')
    ap.add_argument('--model', default='checkpoints_train_8k_r42/final_model.pt')
    ap.add_argument('--vocab', default='checkpoints_train_8k_r42/vocab.json')
    ap.add_argument('--device', default='dml')
    ap.add_argument('--label', default='')
    ap.add_argument('--max-seqs', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--controller', choices=['on', 'off'], default='off')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device)
    apply_cpu_threads(int(cfg.get('training', {}).get('cpu_threads', 0) or 0))
    torch.manual_seed(args.seed)
    label = args.label or str(device)

    model, vocab = load_model(args.model, args.vocab, device=device)
    model.eval()
    model.set_enhancements_active(True)
    if getattr(model, 'controller_enabled', False):
        model._rt_controller = (args.controller == 'on')

    val_dataset, _ = load_val_set(cfg, args.max_seqs)
    x = torch.stack([val_dataset[i]['input_ids'] for i in range(len(val_dataset))]).to(device)
    print(f'[device] {label} ({device})  x={tuple(x.shape)}  controller={args.controller}')

    _patch()
    try:
        with torch.no_grad():
            _rec.clear()
            model(x)
            full = list(_rec)
            _rec.clear()
            model(x[:, :1])
            pre = list(_rec)
    finally:
        _unpatch()

    print(f'SDPA 调用次数：整段 {len(full)} 次（4 层），前缀 {len(pre)} 次\n')
    n = min(len(full), len(pre))
    report = []
    print('=' * 100)
    print('【(1) 位置 0：整段(T=64) vs 前缀(T=1) —— SDPA 的输入与输出】')
    print('=' * 100)
    hdr = '%-8s %-10s %14s %14s %14s'
    for i in range(n):
        f_, p_ = full[i], pre[i]
        print(f'\n--- blocks.{i}.attn  (mask_fill=-1e9, float mask '
              f'{tuple(f_["mask"].shape)}) ---')
        rows = []
        for key in ('q', 'k', 'v'):
            a = f_[key][:, :, 0, :]
            b = p_[key][:, :, 0, :]
            d = float((a - b).abs().max().item())
            rows.append((key, d))
        # out：形状 (B,H,T,D)
        a = f_['out'][:, :, 0, :]
        b = p_['out'][:, :, 0, :]
        d_out = float((a - b).abs().max().item())
        rows.append(('out', d_out))
        # mask 第 0 行（位置 0 允许看的列）
        m64 = f_['mask'][0, 0, 0, :]
        m1 = p_['mask'][0, 0, 0, :]
        n_unmasked_64 = int((m64 > -1e8).sum().item())
        m_diff = float((m64[:1] - m1).abs().max().item())
        print(hdr % ('tensor', 'max|diff|', '', '', ''))
        for k_, d in rows:
            flag = '   <== 跳变' if k_ == 'out' and d > 1e-4 else ''
            print('%-8s %10s %14s %14s %14s%s' % (k_, '', _f(d), '', '', flag))
        print(f'  mask: T=64 时位置0未遮蔽列数={n_unmasked_64}，'
              f'T=1 时位置0未遮蔽列数={int((m1 > -1e8).sum().item())}，'
              f'第0列值之差={_f(m_diff)}')
        report.append({'block': i, 'q': rows[0][1], 'k': rows[1][1],
                       'v': rows[2][1], 'out': d_out,
                       'unmasked_cols_T64': n_unmasked_64})

    print('\n' + '=' * 100)
    print('【(2) 录到的张量（已在 CPU 上）再算一遍 SDPA —— 拆开"掩码长度"与"设备算子"两个变量】')
    print('=' * 100)
    print('%-8s %22s %22s %22s %22s'
          % ('block', 'dev(T=64) vs dev(T=1)', 'dev(T=64) vs cpu(T=64)',
             'dev(T=1) vs cpu(T=1)', 'cpu(T=64) vs cpu(T=1)'))
    for i in range(n):
        f_, p_ = full[i], pre[i]
        dev_full0 = f_['out'][:, :, 0, :]
        dev_pre0 = p_['out'][:, :, 0, :]
        # _rec 里存的 q/k/v/mask 已经 .cpu()，故 _orig(...) 即 CPU 参照
        cpu_full = _orig(f_['q'], f_['k'], f_['v'],
                         attn_mask=f_['mask'])[:, :, 0, :]
        cpu_pre = _orig(p_['q'], p_['k'], p_['v'],
                        attn_mask=p_['mask'])[:, :, 0, :]
        d1 = float((dev_full0 - dev_pre0).abs().max().item())
        d2 = float((dev_full0 - cpu_full).abs().max().item())
        d3 = float((dev_pre0 - cpu_pre).abs().max().item())
        d4 = float((cpu_full - cpu_pre).abs().max().item())
        print('%-8s %22s %22s %22s %22s'
              % (i, _f(d1), _f(d2), _f(d3), _f(d4)))

    print('\n' + '=' * 100)
    print('【(3) 掩码消融：同一份 q/k/v(64)，只换 mask 形态，看位置0输出】')
    print('=' * 100)
    f_ = full[0]
    q, k, v, m = f_['q'].to(device), f_['k'].to(device), f_['v'].to(device), f_['mask'].to(device)
    ref_dml = f_['out'][:, :, 0, :].cpu()
    variants = {}
    variants['mask(1,1,T,T) float -1e9'] = m
    variants['mask expand 到 (B,H,T,T)'] = m.expand(q.size(0), -1, -1, -1).contiguous()
    neg_inf = torch.where(m > -1e8, torch.zeros_like(m), torch.full_like(m, float('-inf')))
    variants['mask float -inf'] = neg_inf
    for name, mm in variants.items():
        try:
            o = _orig(q, k, v, attn_mask=mm)[:, :, 0, :].cpu()
            print('%-34s max|diff vs 录到的整段结果| = %s' % (name, _f(float((o - ref_dml).abs().max().item()))))
        except Exception as e:
            print('%-34s 报错: %s: %s' % (name, type(e).__name__, e))
    try:
        o = _orig(q, k, v, is_causal=True)[:, :, 0, :].cpu()
        print('%-34s max|diff vs 录到的整段结果| = %s'
              % ('is_causal=True（无 mask）', _f(float((o - ref_dml).abs().max().item()))))
    except Exception as e:
        print('is_causal=True 报错: %s' % e)

    if args.out:
        import json, time
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'meta': {'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                                'device': label, 'controller': args.controller,
                                'max_seqs': args.max_seqs, 'seed': args.seed},
                       'per_block': report}, f, ensure_ascii=False, indent=2)
        print(f'\n已写入 {args.out}')


if __name__ == '__main__':
    main()
