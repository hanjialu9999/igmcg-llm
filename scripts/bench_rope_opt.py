"""第二十六轮性能基准：rope.py 缓存命中路径删冗余 .to(dtype) 优化效果验证。

测试方法：
1. 构建 4 层模型，前向 100 步精确计时
2. A/B 对比：
   - A: 优化后（当前代码）
   - B: 模拟优化前（手动加回 .to(dtype)）
3. 控制变量：相同模型权重、相同输入、相同 batch_size

注意：DML 上每个 aten::to 算子有 ~50μs dispatch tax
4 层模型每步 8 次 RoPE 调用（每层 q+k 各一次）× 2 次 .to = 16 次冗余算子/step
预期节省：16 × 50μs = 0.8ms/step（理论值，实际可能受其他因素影响）
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models.rope import RotaryEmbedding
from models.transformer import TransformerModel
from models.device import get_device


def bench_rope_standalone():
    """独立测试 RoPE 缓存命中路径：优化前后算子数对比"""
    print("=" * 70)
    print("Test 1: Standalone RoPE cache hit benchmark")
    print("=" * 70)
    device = get_device()
    rope = RotaryEmbedding(dim=32).to(device)
    q = torch.randn(2, 8, 64, 32, device=device)
    # 预热填充缓存
    rope._get_cos_sin(0, 64, device, q.dtype)

    # 优化后：cache hit 直接切片
    n_iter = 1000
    start = time.perf_counter()
    for _ in range(n_iter):
        rope._get_cos_sin(0, 64, device, q.dtype)
    elapsed_opt = time.perf_counter() - start
    print(f"Optimized: {n_iter} iters in {elapsed_opt*1000:.2f}ms "
          f"= {elapsed_opt*1000/n_iter*1000:.1f}μs/iter")

    # 模拟优化前：每次都 .to(dtype)
    start = time.perf_counter()
    for _ in range(n_iter):
        cos_full, sin_full = rope._cache[(str(device), str(q.dtype), rope.inv_freq.shape[0])]
        # 模拟旧路径：每次 .to(dtype)
        cos = cos_full[:, :, 0:64, :].to(q.dtype)
        sin = sin_full[:, :, 0:64, :].to(q.dtype)
    elapsed_old = time.perf_counter() - start
    print(f"Old (with .to): {n_iter} iters in {elapsed_old*1000:.2f}ms "
          f"= {elapsed_old*1000/n_iter*1000:.1f}μs/iter")
    speedup = elapsed_old / elapsed_opt
    saved_us = (elapsed_old - elapsed_opt) * 1000 / n_iter * 1000
    print(f"Speedup: {speedup:.2f}x, Saved: {saved_us:.1f}μs/iter")
    print()


def bench_full_model():
    """端到端模型前向 100 步计时"""
    print("=" * 70)
    print("Test 2: Full model forward 100 steps (DML, 4 layers, 7M params)")
    print("=" * 70)
    device = get_device()
    torch.manual_seed(42)

    # 与 config_train_8k.yaml 一致的配置
    model = TransformerModel(
        vocab_size=2000,
        embedding_dim=256,
        num_heads=8,
        num_layers=4,
        hidden_dim=512,
        max_seq_length=64,
        dropout=0,
        tie_weights=True,
        gradient_checkpointing=False,
        mixer='attn_linear',
        alibi=True,
        char_merge=True,
        char_merge_kernel=3,
        rope_dim_fraction=0.5,
        output_gate=True,
        zero_centered_norm=True,
        fuse_swiglu=True,
        yarn_scale=2.0,
        intra_hybrid_rope=True,
        intra_hybrid_ratio=0.5,
        head_temp=True,
        value_relative_coding=True,
        gpas=True,
        alibi_learnable=True,
    ).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params, device={device}")

    batch_size = 24
    seq_len = 64
    # 模拟训练数据
    src = torch.randint(0, 2000, (batch_size, seq_len), device=device)
    tgt = torch.randint(0, 2000, (batch_size, seq_len), device=device)

    # 预热 10 步
    print("Warming up 10 steps...")
    for _ in range(10):
        optimizer.zero_grad()
        logits = model(src)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1))
        loss.backward()
        optimizer.step()

    # 正式计时 100 步
    print("Benchmarking 100 steps...")
    torch.synchronize() if hasattr(torch, 'synchronize') else None
    start = time.perf_counter()
    total_tokens = 0
    for _ in range(100):
        optimizer.zero_grad()
        logits = model(src)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1))
        loss.backward()
        optimizer.step()
        total_tokens += batch_size * seq_len
    torch.synchronize() if hasattr(torch, 'synchronize') else None
    elapsed = time.perf_counter() - start

    ms_per_step = elapsed * 1000 / 100
    tokens_per_sec = total_tokens / elapsed
    print(f"\nResult:")
    print(f"  Total: {elapsed:.2f}s for 100 steps")
    print(f"  Per step: {ms_per_step:.2f}ms")
    print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")
    print(f"  Loss: {loss.item():.4f}")
    print()


if __name__ == '__main__':
    bench_rope_standalone()
    bench_full_model()
