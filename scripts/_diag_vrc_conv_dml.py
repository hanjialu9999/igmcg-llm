"""第六批 C（算子级·第二跳）：验证 value_relative_coding 的 conv1d 因果卷积在 DML 上是否算错。

上一步结论（_diag_sdpa_dml.py）：
  · 同一输入下 SDPA 在 DML 与 CPU 上**逐位相同**（dev(T=64) vs cpu(T=64) = 0.0）
  · CPU：整段(T=64) 与 前缀(T=1) 在位置 0 的 q/k/v/**out 全部 = 0.0**
  · DML：q/k ≈1e-6，但 **v 与 out = 6.3677e-01**（且 out 差 = v 差，因掩码使 out[0]=v[0]）
  · 唯一只作用在 v 上、且只在 T>1 时生效的变换 = VRC 递推（mixers.py:617-661），
    其全量路径用 F.conv1d(groups=C, padding=T-1) 实现因果卷积；T=1 时该分支不进 → v 原样。
  ⇒ 嫌疑锁定 mixers.py:658-661 的 conv1d。

第七轮（2026-09-30）更正：conv1d 无罪。真正根因是本脚本 `vrc_conv` 里构造卷积核的
`(lam ** torch.arange(...))` —— lam 是 `reshape(1,1,1,1)` 的 rank-4 标量，在 DML 上
命中 broadcast bug（见 docs/ARCHITECTURE.md 附录 B.3 **N6**）。conv1d 本身 DML/CPU 一致。
已改用 `models.mixers._vrc_decay_kernel`。

本脚本直接对该 conv1d 做独立复现（不碰模型），比 DML vs CPU vs 手写递推三者。

用法：
  python scripts/_diag_vrc_conv_dml.py --device dml
  python scripts/_diag_vrc_conv_dml.py --device cpu
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.device import get_device  # noqa: E402
from models.mixers import _vrc_decay_kernel  # noqa: E402


def vrc_conv(v, lam):
    """复刻 mixers.py:643-661（T>1 分支）。v: (B,H,T,D)"""
    T = v.size(2)
    B_, H_, _, D_ = v.shape
    v_2d = v.permute(0, 2, 1, 3).reshape(B_, T, H_ * D_, 1)
    _klen = T
    # 第七轮更正：原来写 `(lam ** torch.arange(...))`，而 lam 是 reshape(1,1,1,1)
    # 的 rank-4 标量 → 在 DML 上正是 N6 broadcast bug（只取 base[0]**exp[0] 广播全张量、
    # 0-dim `0**0` 返 nan）。conv1d 本身没问题，锅在核构造。改用 models/mixers.py
    # 的 _vrc_decay_kernel（纯连乘，DML/CPU 逐位一致）。
    _ker = _vrc_decay_kernel(lam.reshape(()), _klen).reshape(1, 1, _klen)
    _ker_full = _ker.repeat(H_ * D_, 1, 1)
    _v_c = F.conv1d(v_2d.squeeze(-1).permute(0, 2, 1), _ker_full,
                    groups=H_ * D_, padding=_klen - 1)
    return _v_c[..., :_klen].permute(0, 2, 1).reshape(B_, T, H_, D_).permute(0, 2, 1, 3)


def vrc_ref(v, lam):
    """手写逐 token 递推（真值）。v: (B,H,T,D) → v_enc[t] = v[t] + lam*v_enc[t-1]"""
    out = torch.zeros_like(v)
    h = torch.zeros(v.size(0), v.size(1), v.size(3), device=v.device, dtype=v.dtype)
    for t in range(v.size(2)):
        h = v[:, :, t, :] + lam * h
        out[:, :, t, :] = h
    return out


def main():
    ap = argparse.ArgumentParser(description='VRC conv1d 因果卷积 DML 正确性验证（只读）')
    ap.add_argument('--device', default='dml')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = get_device(args.device)
    torch.manual_seed(args.seed)
    print(f'[device] {device}\n')

    for T in (2, 4, 16, 64):
        for lam_val in (0.0, 0.3, -0.7, 0.99):
            B, H, D, C = 2, 8, 32, 8 * 32
            v = torch.randn(B, H, T, D, device=device)
            lam = torch.tensor(lam_val, device=device).reshape(1, 1, 1, 1)
            got = vrc_conv(v, lam)
            ref = vrc_ref(v, torch.tensor(lam_val, device=device))
            d0 = float((got[:, :, 0, :] - v[:, :, 0, :]).abs().max().item())
            dall = float((got - ref).abs().max().item())
            d0ref = float((got[:, :, 0, :] - ref[:, :, 0, :]).abs().max().item())
            # 与 CPU 同输入对照（把输入搬到 cpu 再算）
            got_cpu = vrc_conv(v.cpu(), lam.cpu())
            ddev = float((got.cpu() - got_cpu).abs().max().item())
            flag = '   <== 位置0 应为 v[0]，偏离即错' if d0 > 1e-5 else ''
            print('T=%-3d lam=%-5.2f | pos0(conv vs v[0])=%9.2e | '
                  'conv vs 手写递推 max=%9.2e | pos0 vs 手写=%9.2e | '
                  'dev vs cpu=%9.2e%s' % (T, lam_val, d0, dall, d0ref, ddev, flag))

    # 单独看 conv1d padding 语义：DML 是否把 padding=T-1 处理对
    print('\n--- 单测 conv1d padding（C=4, T=8, kernel=[λ^7..1]）---')
    T = 8
    x = torch.arange(1, T + 1, dtype=torch.float32, device=device).reshape(1, 1, T)
    lam = 0.5
    ker = torch.tensor([lam ** (T - 1 - i) for i in range(T)],
                       dtype=torch.float32, device=device).reshape(1, 1, T)
    out = F.conv1d(x, ker, padding=T - 1)[0, 0, :T]
    # 直接用递推真值
    h = 0.0
    ref2 = []
    for t in range(T):
        h = float(x[0, 0, t]) + lam * h
        ref2.append(h)
    ref2 = torch.tensor(ref2, device=device)
    print('conv1d 输出 :', [round(float(t), 6) for t in out])
    print('递推真值    :', [round(float(t), 6) for t in ref2])
    print('max|conv - 递推| = %.3e' % float((out - ref2).abs().max().item()))
    print('conv[0] 应= x[0] = %.6f，实际 = %.6f'
          % (float(x[0, 0, 0]), float(out[0])))


if __name__ == '__main__':
    main()
