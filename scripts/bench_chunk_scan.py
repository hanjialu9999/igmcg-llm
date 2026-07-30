"""R36-4b chunk_scan 性能基准：对比 chunk_scan ON vs OFF 的前向+反向时间。

验证 _matrix_prefix_scan（O(T log T)）vs for 循环（O(T²)）在 DML 后端的实际加速比。
理论：T=64 时 for 循环 64 步串行 dispatch vs chunk_scan ~6 轮并行 bmm。
"""
import torch
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mixers import GatedDeltaNet


def bench_gdn(gdn, x, iters=50, warmup=10):
    """计前向+反向平均时间（μs/iter）。"""
    gdn.train()
    # warmup
    for _ in range(warmup):
        out, _ = gdn(x, use_cache=False)
        loss = out.sum()
        loss.backward()
        gdn.zero_grad()
    # DML 无 cuda.synchronize，用空跑一次同步
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(iters):
        out, _ = gdn(x, use_cache=False)
        loss = out.sum()
        loss.backward()
        gdn.zero_grad()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return (time.perf_counter() - t0) / iters * 1e6  # μs/iter


def main():
    try:
        from models.device import get_device
        device = get_device()
    except Exception:
        device = torch.device('cpu')
    print(f"Device: {device}")
    print(f"torch: {torch.__version__}\n")

    torch.manual_seed(42)
    dim, heads = 128, 4
    configs = [
        ("T=16 (1 chunk)", 16),
        ("T=32 (2 chunks)", 32),
        ("T=64 (4 chunks)", 64),
    ]

    B = 2
    print(f"{'Config':<22} {'OFF (μs)':>12} {'ON (μs)':>12} {'Speedup':>10}")
    print("-" * 58)
    for label, T in configs:
        x = torch.randn(B, T, dim, device=device)
        # OFF
        gdn_off = GatedDeltaNet(dim, heads, qk_norm=True, attn_temp=True,
                                max_seq_length=64, alpha_init=-2.0, beta_init=2.0,
                                chunk_scan=False).to(device)
        t_off = bench_gdn(gdn_off, x)
        # ON
        gdn_on = GatedDeltaNet(dim, heads, qk_norm=True, attn_temp=True,
                              max_seq_length=64, alpha_init=-2.0, beta_init=2.0,
                              chunk_scan=True, chunk_size=16).to(device)
        t_on = bench_gdn(gdn_on, x)
        speedup = t_off / t_on if t_on > 0 else float('inf')
        print(f"{label:<22} {t_off:>12.0f} {t_on:>12.0f} {speedup:>9.2f}x")

    # 增量解码 benchmark（T=1 单步）
    print(f"\n{'增量解码 T=1':<22} {'OFF (μs)':>12} {'ON (μs)':>12} {'Speedup':>10}")
    print("-" * 58)
    x1 = torch.randn(1, 1, dim, device=device)
    gdn_off = GatedDeltaNet(dim, heads, qk_norm=True, attn_temp=True,
                            max_seq_length=64, alpha_init=-2.0, beta_init=2.0,
                            chunk_scan=False).to(device)
    gdn_off.eval()
    gdn_on = GatedDeltaNet(dim, heads, qk_norm=True, attn_temp=True,
                           max_seq_length=64, alpha_init=-2.0, beta_init=2.0,
                           chunk_scan=True, chunk_size=16).to(device)
    gdn_on.eval()
    # 构造初始 cache（4 元组）
    H, D = heads, dim // heads
    past = (torch.randn(1, H, 1, D, device=device), torch.randn(1, H, 1, D, device=device),
            torch.randn(1, H, D, D, device=device), torch.randn(1, H, D, device=device))
    t_off = bench_gdn_inc(gdn_off, x1, past)
    t_on = bench_gdn_inc(gdn_on, x1, past)
    speedup = t_off / t_on if t_on > 0 else float('inf')
    print(f"{'inc decode':<22} {t_off:>12.0f} {t_on:>12.0f} {speedup:>9.2f}x")


def bench_gdn_inc(gdn, x, past, iters=50, warmup=10):
    """增量解码单步计时。"""
    gdn.eval()
    for _ in range(warmup):
        o, p = gdn(x, past_kv=past, use_cache=True, start_pos=0)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(iters):
        o, p = gdn(x, past_kv=past, use_cache=True, start_pos=0)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return (time.perf_counter() - t0) / iters * 1e6


if __name__ == '__main__':
    main()
