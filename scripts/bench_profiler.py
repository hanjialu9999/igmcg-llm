"""第二十六轮 profiler 分析：找出优化后的新瓶颈"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models.transformer import TransformerModel
from models.device import get_device


def profile_model():
    device = get_device()
    torch.manual_seed(42)

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

    batch_size = 24
    seq_len = 64
    src = torch.randint(0, 2000, (batch_size, seq_len), device=device)
    tgt = torch.randint(0, 2000, (batch_size, seq_len), device=device)

    # 预热
    for _ in range(5):
        optimizer.zero_grad()
        logits = model(src)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1))
        loss.backward()
        optimizer.step()

    # 用 profiler 跑 10 步
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,
    ) as prof:
        for _ in range(10):
            optimizer.zero_grad()
            logits = model(src)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1))
            loss.backward()
            optimizer.step()

    print("\n=== Top 20 operators by CPU time (10 steps) ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    print("\n=== Top 20 operators by call count (10 steps) ===")
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=20))

    # 提取关键算子统计
    print("\n=== 关键算子统计 (10 steps) ===")
    events = prof.key_averages()
    key_ops = ['aten::to', 'aten::copy_', 'aten::mul', 'aten::add',
               'aten::cat', 'aten::where', 'aten::reshape', 'aten::_to_copy',
               'aten::clone', 'aten::empty', 'aten::arange', 'aten::unsqueeze']
    for ev in events:
        for k in key_ops:
            if k in str(ev.key):
                print(f"  {ev.key}: count={ev.count}, cpu_time={ev.cpu_time_total/1000:.2f}ms")
                break


if __name__ == '__main__':
    profile_model()
