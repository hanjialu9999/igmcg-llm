"""GDN 循环外提的逐位对拍：某个提交里的 GatedDeltaNet vs 工作区的 models/mixers.py。

用法（先 git apply gdn_loop_v3.patch，再跑）：
    python experiments/claude_review_20261003/gdn_loop_v3_check.py                       # v2(HEAD) 对 v3，CPU
    python experiments/claude_review_20261003/gdn_loop_v3_check.py --device dml
    python experiments/claude_review_20261003/gdn_loop_v3_check.py --base 82c44f8 --device dml   # v1 对 v3
--base 指定改前代码所在提交（默认 HEAD），用 `git show <base>:models/mixers.py` 取出，和工作区版本
同权重同输入比较：前向、输入梯度、全部参数梯度 torch.equal 和最大差；CPU 下另报前向+反向算子数。
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)


def load_base_mixers(base):
    src = subprocess.check_output(['git', 'show', f'{base}:models/mixers.py'], cwd=ROOT)
    fd, path = tempfile.mkstemp(suffix='_mixers_head.py')
    with os.fdopen(fd, 'wb') as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location('mixers_head', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def count_ops(fn):
    from torch.utils._python_dispatch import TorchDispatchMode
    n = [0]

    class Count(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            n[0] += 1
            return func(*args, **(kwargs or {}))
    with Count():
        fn()
    return n[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu', choices=['cpu', 'dml'])
    ap.add_argument('--base', default='HEAD', help='改前代码所在提交，默认 HEAD')
    args = ap.parse_args()
    if args.device == 'dml':
        import torch_directml
        dev = torch_directml.device()
    else:
        dev = torch.device('cpu')
    from models.mixers import GatedDeltaNet as New
    Old = load_base_mixers(args.base).GatedDeltaNet
    torch.manual_seed(0)
    kw = dict(dim=256, num_heads=8, qk_norm=True, attn_temp=True, max_seq_length=64,
              alpha_init=-2.0, beta_init=2.0)
    m_old, m_new = Old(**kw), New(**kw)
    with torch.no_grad():
        m_old.alpha_beta_proj.weight.normal_(0, 0.05)   # 让门随输入变化，避免测试过于平凡
    m_new.load_state_dict(m_old.state_dict())
    m_old.to(dev); m_new.to(dev)
    x = torch.randn(24, 64, 256)

    def run(m):
        xx = x.to(dev).requires_grad_(True)
        y, _ = m(xx, use_cache=True)
        y.pow(2).sum().backward()
        return y.detach().cpu(), xx.grad.cpu(), {n: p.grad.cpu() for n, p in m.named_parameters() if p.grad is not None}

    y1, gx1, gp1 = run(m_old)
    y2, gx2, gp2 = run(m_new)
    print('device', dev, ' base', args.base)
    print('前向 torch.equal:', torch.equal(y1, y2), ' max|Δ|=', (y1 - y2).abs().max().item())
    print('输入梯度 torch.equal:', torch.equal(gx1, gx2))
    print('参数梯度全部 torch.equal:', all(torch.equal(gp1[n], gp2[n]) for n in gp1),
          ' 最大差', max((gp1[n] - gp2[n]).abs().max().item() for n in gp1))
    if args.device == 'cpu':
        n1 = count_ops(lambda: run(m_old)); n2 = count_ops(lambda: run(m_new))
        print(f'前向+反向算子数 改前 {n1} → 改后 {n2}（{(n2 - n1) / n1:+.1%}）')
    for label, m in (('改前', m_old), ('改后', m_new)):
        ts = []
        for _ in range(4):
            t0 = time.perf_counter(); run(m); ts.append(time.perf_counter() - t0)
        ts = sorted(ts[1:])
        print(f'{label} 前向+反向 {ts[1] * 1e3:.1f} ms（{ts[0] * 1e3:.1f}–{ts[2] * 1e3:.1f}）')


if __name__ == '__main__':
    main()
