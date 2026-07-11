import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, 'scripts') not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import sys, torch
sys.path.insert(0, '.')
from models.config_loader import build_model
from models.data_utils import load_data
from models.device import get_device
from torch.utils.data import DataLoader

V = 12000
dataset, vocab = load_data('data/pretrain_corpus/merged_sample.txt', vocab_size=V, max_seq_length=64)
dl = DataLoader(dataset, batch_size=32, shuffle=True)

cfg = {'model': {
    'vocab_size': V, 'embedding_dim': 512, 'num_heads': 8, 'num_layers': 6,
    'hidden_dim': 1024, 'max_seq_length': 64, 'dropout': 0.0, 'tie_weights': True,
    'gradient_checkpointing': True,
    'layer_plan': 'attn,ssm,attn,ssm,attn,ssm',
    'ssm_d_state': 16, 'ssm_d_inner_factor': 1, 'attn_window': 0, 'attn_rel_bias': False,
}}
dev = get_device('cpu')
model = build_model(cfg, device=dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, foreach=False)
crit = torch.nn.functional.cross_entropy
model.train()
losses = []
for i, batch in enumerate(dl):
    if i >= 40:
        break
    x = batch['input_ids'].to(dev); y = batch['target_ids'].to(dev)
    logits = model(x)
    loss = crit(logits.view(-1, V), y.view(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())
    if i % 8 == 0:
        print(f'step {i} loss={loss.item():.3f}', flush=True)
print(f'FIRST={losses[0]:.3f} LAST={losses[-1]:.3f}', flush=True)

model.eval()
ids = vocab.encode('今天天气')
gen = model.generate(ids, max_length=30, temperature=0.8, top_k=30, device=dev)
out = vocab.decode(gen, skip_special=True)
with open('logs/hybrid_smoke.txt', 'w', encoding='utf-8') as f:
    f.write(f'FIRST={losses[0]:.3f} LAST={losses[-1]:.3f}\nGEN: {out}\n')
print('REALDATA SMOKE OK', flush=True)
