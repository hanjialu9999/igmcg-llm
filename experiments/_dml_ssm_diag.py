import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, 'scripts') not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import sys, torch
sys.path.insert(0, '.')
from models.device import get_device
from models.transformer import MambaSSM
import torch.nn.functional as F

dev = get_device('auto')
print('device', dev, flush=True)
open('logs/ssm_diag.txt', 'w', encoding='utf-8', errors='replace').close()

def tryback(name, fn):
    try:
        fn()
        print(name, 'OK', flush=True)
    except Exception as e:
        print(name, 'CRASH', flush=True)
        with open('logs/ssm_diag.txt', 'a', encoding='utf-8', errors='replace') as f:
            f.write(f'=== {name}: {type(e).__name__} ===\n')
            for a in e.args:
                if isinstance(a, bytes):
                    f.write('  bytes: ' + a.decode('latin-1', 'replace') + '\n')
                else:
                    try:
                        f.write('  ' + str(a) + '\n')
                    except Exception:
                        f.write('  ' + repr(a) + '\n')

# 1) softplus backward
def t_softplus():
    x = torch.randn(2, 16, 32, device=dev, requires_grad=True)
    y = F.softplus(x); y.sum().backward()
tryback('softplus', t_softplus)

# 2) conv1d (depthwise) backward
def t_conv():
    conv = torch.nn.Conv1d(32, 32, 3, padding=1, groups=32, bias=False).to(dev)
    x = torch.randn(2, 32, 16, device=dev, requires_grad=True)
    y = conv(x); y.sum().backward()
tryback('conv1d', t_conv)

# 3) manual softplus (log1p/exp) backward
def t_manual_sp():
    x = torch.randn(2, 16, 32, device=dev, requires_grad=True)
    y = x.clamp(min=0) + torch.log1p(torch.exp(-x.abs())); y.sum().backward()
tryback('manual_softplus', t_manual_sp)

# 4) full MambaSSM backward
def t_ssm():
    m = MambaSSM(32, d_state=8, d_inner_factor=1).to(dev)
    x = torch.randn(2, 16, 32, device=dev, requires_grad=True)
    out = m(x); out.sum().backward()
tryback('MambaSSM', t_ssm)

print('DIAG DONE', flush=True)
