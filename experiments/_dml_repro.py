import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, 'scripts') not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import sys
import torch
from models.device import get_device
from models.transformer import TransformerModel

dev = get_device('auto')
print("device:", dev, flush=True)
model = TransformerModel(
    vocab_size=12000, embedding_dim=512, num_heads=8, num_layers=6,
    hidden_dim=1024, max_seq_length=64, dropout=0.0,
    tie_weights=True, gradient_checkpointing=False).to(dev)
print("model built, params:", sum(p.numel() for p in model.parameters()), flush=True)

use_foreach = (len(sys.argv) > 1 and sys.argv[1] == 'foreach')
opt = torch.optim.Adam(model.parameters(), lr=1e-3, foreach=use_foreach)
print("optimizer foreach =", use_foreach, flush=True)

model.train()
for step in range(8):
    x = torch.randint(0, 12000, (2, 64), device=dev)
    logits = model(x)
    loss = logits.float().sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"step {step} ok, loss={loss.item():.3f}", flush=True)
print("REPRO DONE", flush=True)
