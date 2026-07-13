"""对照基准：逐个开启架构增强开关，测量 TransformerModel 单步 forward+backward 耗时（DML）。

目标：定位开启增强后训练吞吐下降的主要来源（qk_norm / attn_temp / residual_gate）。
用法：python experiments/_bench_enh.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import torch.nn as nn

try:
    import torch_directml
    device = torch_directml.device(0)
    dev_name = "dml"
except Exception:
    device = torch.device("cpu")
    dev_name = "cpu"

print(f"[device] {device}")

from models.transformer import TransformerModel

B, T, V, D, H, L, HD = 32, 64, 2000, 256, 4, 4, 512

combos = {
    "none":        dict(qk_norm=False, attn_temp=False, residual_gate=False),
    "qk_norm":     dict(qk_norm=True,  attn_temp=False, residual_gate=False),
    "attn_temp":   dict(qk_norm=False, attn_temp=True,  residual_gate=False),
    "residual_gate": dict(qk_norm=False, attn_temp=False, residual_gate=True),
    "all":         dict(qk_norm=True,  attn_temp=True,  residual_gate=True),
    "all_nockpt":  dict(qk_norm=True,  attn_temp=True,  residual_gate=True, _nockpt=True),
    "none_nockpt": dict(qk_norm=False, attn_temp=False, residual_gate=False, _nockpt=True),
}

def build(flags):
    nockpt = flags.pop("_nockpt", False)
    m = TransformerModel(vocab_size=V, embedding_dim=D, num_heads=H, num_layers=L,
                         hidden_dim=HD, max_seq_length=T, dropout=0.0, tie_weights=True,
                         gradient_checkpointing=(not nockpt), **flags).to(device)
    return m

def step(m):
    x = torch.randint(0, V, (B, T), device=device)
    t0 = time.perf_counter()
    m.zero_grad(set_to_none=True)
    logits = m(x)
    loss = nn.functional.cross_entropy(logits.reshape(-1, V), x.reshape(-1))
    loss.backward()
    _ = loss.item()  # 触发 DML 流同步，保证计时准确
    t1 = time.perf_counter()
    return t1 - t0

results = {}
for name, flags in combos.items():
    m = build(flags)
    nparams = sum(p.numel() for p in m.parameters())
    # warmup
    for _ in range(3):
        step(m)
    ts = []
    for _ in range(8):
        ts.append(step(m))
    avg = sum(ts) / len(ts)
    results[name] = avg
    print(f"[{name:14s}] {avg*1000:8.1f} ms/step   params={nparams}")

base = results["none"]
print("\n相对 none 的耗时倍数：")
for name, t in results.items():
    mult = t / base
    tok_s = (B * T) / t
    print(f"  {name:14s} {mult:5.2f}x   ~{tok_s:6.0f} tok/s")
