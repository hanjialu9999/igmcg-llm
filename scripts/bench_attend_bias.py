"""第二十七轮：attend bias 合并优化基准测试。

验证 broadcast_tensors+stack+sum 是否比链式 add 更快。
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F


def bench_bias_merge_chain(base, biases):
    """链式 add：attn_mask = ((base + b1) + b2) + b3"""
    for _ in range(100):
        attn_mask = base
        for b in biases:
            attn_mask = attn_mask + b
    return attn_mask


def bench_bias_merge_stack(base, biases):
    """broadcast+stack+sum：1 次 stack + 1 次 sum"""
    for _ in range(100):
        addends = [base] + list(biases)
        broadcasted = torch.broadcast_tensors(*addends)
        attn_mask = torch.stack(broadcasted, dim=0).sum(dim=0)
    return attn_mask


def bench_bias_merge_add2(base, biases):
    """2 bias 路径：直接 base + b1（不进 stack+sum）"""
    for _ in range(100):
        attn_mask = base + biases[0]
    return attn_mask


def main():
    from models.device import get_device
    device = get_device()
    print(f"Device: {device}")

    # 模拟 attend 中的 bias 场景
    # base_mask: (1,1,T,Tkv)，其他 bias: (1,H,T,Tkv) 或 (B,H,T,Tkv)
    B, H, T, Tkv = 2, 4, 64, 64

    scenarios = [
        ("2 biases (base+alibi)", [
            torch.randn(1, 1, T, Tkv, device=device),
            torch.randn(1, H, T, Tkv, device=device),
        ]),
        ("3 biases (base+alibi+mem)", [
            torch.randn(1, 1, T, Tkv, device=device),
            torch.randn(1, H, T, Tkv, device=device),
            torch.randn(B, H, T, Tkv, device=device),
        ]),
        ("4 biases (base+rel+alibi+mem)", [
            torch.randn(1, 1, T, Tkv, device=device),
            torch.randn(1, H, T, Tkv, device=device),
            torch.randn(1, H, T, Tkv, device=device),
            torch.randn(B, H, T, Tkv, device=device),
        ]),
    ]

    n_iter = 1000

    for name, biases in scenarios:
        base = biases[0]
        rest = biases[1:]

        # 数值等价验证
        chain_result = base
        for b in rest:
            chain_result = chain_result + b
        broadcasted = torch.broadcast_tensors(*([base] + rest))
        stack_result = torch.stack(broadcasted, dim=0).sum(dim=0)
        max_diff = (chain_result - stack_result).abs().max().item()
        assert max_diff < 1e-5, f"数值不一致: {max_diff}"

        # 预热
        for _ in range(10):
            bench_bias_merge_chain(base, rest)
            if len(rest) >= 2:
                bench_bias_merge_stack(base, rest)
            else:
                bench_bias_merge_add2(base, rest)

        # 基准：链式 add
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.perf_counter()
        for _ in range(n_iter):
            bench_bias_merge_chain(base, rest)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        chain_time = (time.perf_counter() - start) / n_iter * 1000

        # 优化：stack+sum（仅当 >= 2 bias）
        if len(rest) >= 2:
            start = time.perf_counter()
            for _ in range(n_iter):
                bench_bias_merge_stack(base, rest)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            stack_time = (time.perf_counter() - start) / n_iter * 1000
            speedup = chain_time / stack_time
            saved = (chain_time - stack_time) * 1000  # μs
            print(f"{name}: chain={chain_time*1000:.1f}μs stack={stack_time*1000:.1f}μs "
                  f"speedup={speedup:.2f}x saved={saved:.1f}μs/step")
        else:
            start = time.perf_counter()
            for _ in range(n_iter):
                bench_bias_merge_add2(base, rest)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            add2_time = (time.perf_counter() - start) / n_iter * 1000
            print(f"{name}: chain={chain_time*1000:.1f}μs add2={add2_time*1000:.1f}μs")

    print("\n--- 端到端训练速度基准（100 步平均）---")
    # 构建小模型测试端到端速度
    from models.transformer import TransformerModel
    from models.device import get_device

    device = get_device()
    model = TransformerModel(
        vocab_size=5000, embedding_dim=256, num_heads=8, num_layers=4,
        hidden_dim=512, max_seq_length=64, mixer='attn',
        alibi=True,  # 开启 alibi 触发 bias 合并路径
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randint(0, 5000, (4, 64), device=device)
    y = torch.randint(0, 5000, (4, 64), device=device)

    # 预热
    for _ in range(5):
        out = model(x)
        loss = F.cross_entropy(out.view(-1, 5000), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 基准
    n_steps = 100
    start = time.perf_counter()
    for _ in range(n_steps):
        out = model(x)
        loss = F.cross_entropy(out.view(-1, 5000), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    elapsed = time.perf_counter() - start
    ms_per_step = elapsed / n_steps * 1000
    tokens_per_sec = (4 * 64) / (ms_per_step / 1000)
    print(f"4 层 attn+alibi 模型: {ms_per_step:.1f}ms/step, {tokens_per_sec:.0f} tok/s")


if __name__ == '__main__':
    main()
