import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, 'scripts') not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import sys, torch
sys.path.insert(0, '.')
from models.config_loader import build_model, load_vocab
from models.device import get_device
from scripts.generate import generate_igmcg

cfg = {
  'model': {
    'vocab_size': 12000, 'embedding_dim': 512, 'num_heads': 8, 'num_layers': 6,
    'hidden_dim': 1024, 'max_seq_length': 64, 'dropout': 0.0, 'tie_weights': True,
    'gradient_checkpointing': True,
    'layer_plan': 'attn,ssm,attn,ssm,attn,ssm',
    'ssm_d_state': 16, 'ssm_d_inner_factor': 1, 'attn_window': 0, 'attn_rel_bias': False,
  }
}
dev = get_device('cpu')
model = build_model(cfg, device=dev)
print('params =', sum(p.numel() for p in model.parameters()), flush=True)
print('layer_plan =', model.layer_plan, flush=True)

opt = torch.optim.Adam(model.parameters(), lr=1e-3, foreach=False)
crit = torch.nn.functional.cross_entropy
model.train()
for step in range(5):
    x = torch.randint(0, 12000, (4, 64), device=dev)
    logits = model(x)
    loss = crit(logits.view(-1, 12000), x.view(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    print(f'step {step} loss={loss.item():.3f}', flush=True)

model.eval()
gen = model.generate([2, 100, 200, 300], max_length=20, temperature=0.8, top_k=30, device=dev)
print('generate ids =', gen[:15], flush=True)

vocab = load_vocab('checkpoints/vocab.json')
text, cands = generate_igmcg(model, vocab, '今天天气',
                             intuition=[0.8, 0.5, 0.5, 0.8, 0.7, 0.5, 0.5],
                             num_candidates=3, max_length=30, device=dev)
print('IGMCG best =', text[:60], flush=True)
for c in cands:
    print(f"  temp={c['temp']} score={c['score']:.2f} flu={c['flu']:.2f} rep={c['rep']:.2f} -> {c['text'][:40]}", flush=True)
print('HYBRID SMOKE OK', flush=True)
