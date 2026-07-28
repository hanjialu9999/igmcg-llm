"""R28 端到端训练基准：对比 R27 vs R28 的训练步骤速度。

用纯 attn 模型跑 50 步训练，计时每步 forward+backward。
"""
import torch
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer import TransformerModel
from models.device import get_device


def main():
    device = get_device()
    print(f"Device: {device}")

    # 构建小模型（与训练配置一致：7M 参数级别）
    model = TransformerModel(
        vocab_size=200,
        embedding_dim=128,
        num_heads=4,
        num_layers=4,
        hidden_dim=256,
        max_seq_length=64,
        mixer='attn',
        alibi=True,
    ).to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 模拟训练数据
    B, T = 2, 64
    x = torch.randint(0, 200, (B, T), device=device)
    y = torch.randint(0, 200, (B, T), device=device)

    # warmup
    for _ in range(3):
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out.reshape(-1, 200), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 基准计时
    iters = 30
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(iters):
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out.reshape(-1, 200), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - t0
    ms_per_step = elapsed / iters * 1000
    tok_per_sec = (B * T) / (elapsed / iters)

    print(f"模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"每步耗时: {ms_per_step:.1f} ms")
    print(f"吞吐量:   {tok_per_sec:.0f} tok/s")
    print(f"({iters} 步总耗时 {elapsed:.1f}s)")


if __name__ == '__main__':
    main()
