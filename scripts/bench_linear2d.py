"""全链路线性化验证：benchmark 各 mixer 的推理速度 + 冒烟训练收敛性。

用法:
    python scripts/bench_linear2d.py          # benchmark only
    python scripts/bench_linear2d.py --train  # benchmark + smoke train
"""
import argparse, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from models.transformer import TransformerModel

VOCAB, DIM, HEADS, NLAYERS, HIDDEN = 200, 64, 4, 2, 128


def _bench_one(mixer, layer_plan, seq_len, steps=20, device='cpu'):
    m = TransformerModel(
        vocab_size=VOCAB, embedding_dim=DIM, num_heads=HEADS, num_layers=NLAYERS,
        hidden_dim=HIDDEN, max_seq_length=seq_len + 4, mixer=mixer,
        layer_plan=layer_plan, ssm_d_state=16, ssm_d_inner_factor=1)
    m.eval()
    x = torch.randint(0, VOCAB, (1, seq_len), device=device)
    with torch.no_grad():
        for _ in range(3):
            m(x)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            m(x)
    elapsed = (time.perf_counter() - t0) / steps
    params = sum(p.numel() for p in m.parameters())
    return elapsed, params


def _bench_incremental(mixer, layer_plan, seq_len, device='cpu'):
    m = TransformerModel(
        vocab_size=VOCAB, embedding_dim=DIM, num_heads=HEADS, num_layers=NLAYERS,
        hidden_dim=HIDDEN, max_seq_length=seq_len + 4, mixer=mixer,
        layer_plan=layer_plan, ssm_d_state=16, ssm_d_inner_factor=1)
    m.eval()
    prefill_len = min(seq_len // 2, 8)
    incr_steps = min(10, seq_len - prefill_len)
    x = torch.randint(0, VOCAB, (1, prefill_len + incr_steps), device=device)
    with torch.no_grad():
        _, past = m(x[:, :prefill_len], use_cache=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(incr_steps):
            tok = x[:, prefill_len + i:prefill_len + i + 1]
            _, past = m(tok, past_key_values=past, use_cache=True)
    elapsed = (time.perf_counter() - t0) / max(incr_steps, 1)
    return elapsed


def smoke_train(mixer, layer_plan, seq_len=16, steps=50, lr=3e-4):
    torch.manual_seed(0)
    m = TransformerModel(
        vocab_size=VOCAB, embedding_dim=DIM, num_heads=HEADS, num_layers=NLAYERS,
        hidden_dim=HIDDEN, max_seq_length=seq_len + 4, mixer=mixer,
        layer_plan=layer_plan, ssm_d_state=16, ssm_d_inner_factor=1)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    x = torch.randint(0, VOCAB, (4, seq_len))
    target = torch.roll(x, -1, dims=1)
    losses = []
    for i in range(steps):
        opt.zero_grad()
        logits = m(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    first5 = sum(losses[:5]) / 5
    last5 = sum(losses[-5:]) / 5
    return first5, last5, last5 < first5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Run smoke training')
    args = parser.parse_args()

    configs = [
        ('attn',      'attn',     'attn'),
        ('linear',    'attn',     'linear'),
        ('linear2d',  'attn',     'linear2d'),
        ('hybrid_ssm','hybrid',   'attn'),
        ('hyb_lin2d', 'hybrid',   'hybrid_linear2d'),
    ]

    seq_lens = [16, 36, 64]
    print(f"{'Config':<14} {'T':>4}  {'Fwd(ms)':>8}  {'Incr(ms)':>8}  {'Params':>8}")
    print("-" * 56)
    for name, lp, mix in configs:
        for sl in seq_lens:
            fwd_t, params = _bench_one(mix, lp, sl)
            inc_t = _bench_incremental(mix, lp, sl)
            print(f"{name:<14} {sl:>4}  {fwd_t*1000:>8.1f}  {inc_t*1000:>8.1f}  {params:>8}")
        print()

    if args.train:
        print("=== Smoke Training (50 steps, loss must decrease) ===")
        for name, lp, mix in configs:
            first5, last5, ok = smoke_train(mix, lp)
            status = "PASS" if ok else "FAIL"
            print(f"  {name:<14} loss: {first5:.3f} -> {last5:.3f}  [{status}]")


if __name__ == '__main__':
    main()
