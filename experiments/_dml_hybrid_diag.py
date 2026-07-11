import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, 'scripts') not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import sys, torch
sys.path.insert(0, '.')
from models.config_loader import build_model
from models.device import get_device
from models.transformer import MambaSSM

dev = get_device('auto')
print('device', dev, flush=True)
open('logs/ssm_diag.txt', 'w', encoding='utf-8', errors='replace').close()

def log(name, e):
    with open('logs/ssm_diag.txt', 'a', encoding='utf-8', errors='replace') as f:
        f.write(f'=== {name}: {type(e).__name__} ===\n')
        for a in e.args:
            if isinstance(a, bytes):
                f.write('  ' + a.decode('latin-1', 'replace') + '\n')
            else:
                try:
                    f.write('  ' + str(a) + '\n')
                except Exception:
                    f.write('  ' + repr(a) + '\n')

# A) MambaSSM at TRAINING size, no checkpoint
def t_ssm_size():
    m = MambaSSM(512, d_state=16, d_inner_factor=1).to(dev)
    x = torch.randn(32, 64, 512, device=dev, requires_grad=True)
    out = m(x); out.sum().backward()
try:
    t_ssm_size(); print('A ssm_size OK', flush=True)
except Exception as e:
    print('A ssm_size CRASH', flush=True); log('A ssm_size', e)

# B) full hybrid model WITH gradient_checkpointing
def run_hybrid(ckpt):
    cfg = {'model': {'vocab_size': 12000, 'embedding_dim': 512, 'num_heads': 8, 'num_layers': 6,
        'hidden_dim': 1024, 'max_seq_length': 64, 'dropout': 0.0, 'tie_weights': True,
        'gradient_checkpointing': ckpt,
        'layer_plan': 'attn,ssm,attn,ssm,attn,ssm', 'ssm_d_state': 16,
        'ssm_d_inner_factor': 1, 'attn_window': 0, 'attn_rel_bias': False}}
    m = build_model(cfg, device=dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, foreach=False)
    for _ in range(3):
        x = torch.randint(0, 12000, (32, 64), device=dev)
        out = m(x); loss = out.float().sum()
        opt.zero_grad(); loss.backward(); opt.step()
    print(f'B hybrid ckpt={ckpt} OK', flush=True)

for ckpt in (True, False):
    try:
        run_hybrid(ckpt)
    except Exception as e:
        print(f'B hybrid ckpt={ckpt} CRASH', flush=True); log(f'B hybrid ckpt={ckpt}', e)

print('DONE', flush=True)
