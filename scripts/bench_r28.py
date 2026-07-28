"""R28-2 RoPE 向量化优化基准测试。

对比向量化 RoPE（out = x_rot*cos + x_swap*(sin*sign)）与原方案（x1*cos-x2*sin, x1*sin+x2*cos）。
在 DML 上跑 1000 次前向+反向，计时对比。
"""
import torch
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.rope import RotaryEmbedding


def bench_original_rope(x, cos_half, sin_half, d, iters=1000):
    """原方案：4 mul + 1 sub + 1 add + 1 cat = 7 算子"""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(iters):
        x1, x2 = x[..., :d], x[..., d:]
        out = torch.cat([
            x1 * cos_half - x2 * sin_half,
            x1 * sin_half + x2 * cos_half,
        ], dim=-1)
        loss = out.sum()
        loss.backward()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return (time.perf_counter() - t0) / iters * 1e6  # μs/iter


def bench_vectorized_rope(x, cos, sin, sign, d, iters=1000):
    """R28-2 向量化：cat + mul + mul + mul + add = 5 算子"""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(iters):
        x_swap = torch.cat([x[..., d:], x[..., :d]], dim=-1)
        sin_signed = sin * sign
        out = x * cos + x_swap * sin_signed
        loss = out.sum()
        loss.backward()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return (time.perf_counter() - t0) / iters * 1e6  # μs/iter


def main():
    try:
        from models.device import get_device
        device = get_device()
    except Exception:
        device = torch.device('cpu')
    print(f"Device: {device}")

    torch.manual_seed(42)
    B, H, T, D = 2, 4, 64, 64  # 模拟训练 batch
    rot_dim = D
    d = rot_dim // 2

    x = torch.randn(B, H, T, rot_dim, device=device, requires_grad=True)
    cos_half = torch.randn(1, 1, T, d, device=device)
    sin_half = torch.randn(1, 1, T, d, device=device)
    cos = torch.cat([cos_half, cos_half], dim=-1).expand(1, 1, T, rot_dim)
    sin = torch.cat([sin_half, sin_half], dim=-1).expand(1, 1, T, rot_dim)
    sign = torch.cat([-torch.ones(d, device=device), torch.ones(d, device=device)])

    iters = 500

    # warmup
    for _ in range(10):
        bench_original_rope(x, cos_half, sin_half, d, iters=1)
        bench_vectorized_rope(x, cos, sin, sign, d, iters=1)

    # 原方案
    x.grad = None
    t_orig = bench_original_rope(x, cos_half, sin_half, d, iters)
    print(f"原方案:   {t_orig:.1f} μs/iter (7 算子: 4 mul + 1 sub + 1 add + 1 cat)")

    # 向量化方案
    x.grad = None
    t_vec = bench_vectorized_rope(x, cos, sin, sign, d, iters)
    print(f"向量化:   {t_vec:.1f} μs/iter (5 算子: 1 cat + 3 mul + 1 add)")

    speedup = t_orig / t_vec if t_vec > 0 else 0
    print(f"加速比:   {speedup:.2f}x")
    print(f"省时:     {t_orig - t_vec:.1f} μs/iter")

    # 数值等价验证
    with torch.no_grad():
        x1, x2 = x[..., :d], x[..., d:]
        out_orig = torch.cat([
            x1 * cos_half - x2 * sin_half,
            x1 * sin_half + x2 * cos_half,
        ], dim=-1)
        x_swap = torch.cat([x[..., d:], x[..., :d]], dim=-1)
        sin_signed = sin * sign
        out_vec = x * cos + x_swap * sin_signed
        max_diff = (out_orig - out_vec).abs().max().item()
        print(f"\n数值等价: max_diff = {max_diff:.2e} ({'PASS' if max_diff < 1e-6 else 'FAIL'})")


if __name__ == '__main__':
    main()
