"""H2 修复（10-05）：controller_causal_memory=True 时 Controller mem_kv 逐位置 (B,T,M,·)，位置 t 只读 S_t。

验收对应 docs/H2_CAUSALIZATION_PLAN.md §4：对抗测试（改未来 token，前缀 logits 不变）+ 全量/增量 parity；
注入侧（逐位置 K 置 0 + V 指示通道）与共享记忆同一 softmax 的等价性单独对拍。
"""

import pytest
import torch
import torch.nn as nn

try:
    import torch_directml  # noqa: F401  与 test_round42 同：提前注册 DML 后端
except Exception:
    pass

from models.mixers import GatedDeltaNet, SlidingWindowCausalSelfAttention
from models.model_config import ModelConfig
from models.transformer import TransformerModel


@pytest.mark.parametrize('meta', [None, {'retrieval_gate': torch.tensor(0.3)}])
@pytest.mark.parametrize('use_cache', [False, True])
def test_positional_inject_equals_shared(meta, use_cache):
    torch.manual_seed(0)
    attn = SlidingWindowCausalSelfAttention(64, 4, max_seq_length=32).eval()
    x = torch.randn(2, 10, 64)
    mk, mv = torch.randn(2, 3, 16), torch.randn(2, 3, 16)
    pos = lambda t: t.unsqueeze(1).expand(-1, 10, -1, -1)
    with torch.no_grad():
        a, pa = attn(x, use_cache=use_cache, memory_kv=(mk, mv, meta))
        b, pb = attn(x, use_cache=use_cache, memory_kv=(pos(mk), pos(mv), meta))
    assert (a - b).abs().max().item() < 1e-5
    if use_cache:  # present 不含记忆列，也不带 V 指示通道
        assert pa[1].shape == pb[1].shape and (pa[1] - pb[1]).abs().max().item() < 1e-6


def test_positional_inject_reads_own_slot_only():
    torch.manual_seed(0)
    attn = SlidingWindowCausalSelfAttention(64, 4, max_seq_length=32).eval()
    x = torch.randn(2, 10, 64)
    mk, mv = torch.randn(2, 10, 3, 16), torch.randn(2, 10, 3, 16)
    mk2, mv2 = mk.clone(), mv.clone()
    mk2[:, 7] += 1.0
    mv2[:, 7] += 1.0
    with torch.no_grad():
        a = attn(x, memory_kv=(mk, mv, None))[0]
        b = attn(x, memory_kv=(mk2, mv2, None))[0]
    d = (a - b).abs().amax(dim=(0, 2))
    assert d[7] > 1e-3 and torch.cat([d[:7], d[8:]]).max() < 1e-6


@pytest.mark.parametrize('mode', ['wy', 'loop', 'scan'])
def test_gdn_reads_full_vs_incremental(mode):
    torch.manual_seed(0)
    m = GatedDeltaNet(dim=64, num_heads=4, max_seq_length=64,
                      chunk_wy=(mode == 'wy'), chunk_scan=(mode == 'scan')).eval()
    with torch.no_grad():
        m.alpha_beta_proj.weight.normal_(0, 0.3)
    rq = torch.randn(3, 4, 16)
    x = torch.randn(2, 37, 64)
    with torch.no_grad():
        _, _, full = m.forward_with_reads(x, use_cache=True, read_q=rq)
        _, past, r = m.forward_with_reads(x[:, :1], use_cache=True, read_q=rq)
        rs = [r]
        for t in range(1, 37):
            _, past, r = m.forward_with_reads(x[:, t:t + 1], past_kv=past, use_cache=True, start_pos=t, read_q=rq)
            rs.append(r)
        out_plain, _ = m(x)
        out_reads, _, _ = m.forward_with_reads(x, read_q=rq)
    assert full.shape == (2, 37, 3, 16)
    assert (full - torch.cat(rs, dim=1)).abs().max().item() < 1e-4
    assert torch.equal(out_plain, out_reads)  # 读出不改主输出


def _build(causal_memory=True, chunk_scan=True):
    cfg = ModelConfig(vocab_size=50, embedding_dim=64, num_heads=4, num_layers=3, hidden_dim=128,
                      max_seq_length=32, controller=True, gradient_checkpointing=False,
                      controller_direction_causal=True, controller_chunk_scan=chunk_scan,
                      controller_causal_memory=causal_memory)
    m = TransformerModel.from_config(cfg).eval()
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():  # 放开零初始化的 Controller 输出投影（否则测不出记忆分量）
        c = m.controller
        c.mem_proj.weight.copy_(torch.randn(c.mem_proj.weight.shape, generator=g) * 0.3)
        c.direction_proj.weight.copy_(torch.randn(c.direction_proj.weight.shape, generator=g) * 0.1)
        for proj in c.film_projs:
            if isinstance(proj, nn.Linear):
                proj.weight.copy_(torch.randn(proj.weight.shape, generator=g) * 0.05)
    return m


def _future_leak(m):
    x = torch.randint(0, 50, (2, 12), generator=torch.Generator().manual_seed(0))
    x2 = x.clone()
    x2[:, 6:] = (x2[:, 6:] + 7) % 50  # 只改位置 6 之后
    with torch.no_grad():
        return (m(x)[:, :6] - m(x2)[:, :6]).abs().max().item()


@pytest.mark.parametrize('chunk_scan', [True, False])
def test_causal_memory_no_future_leak(chunk_scan):
    assert _future_leak(_build(chunk_scan=chunk_scan)) < 1e-5


def test_old_memory_leaks_future():
    """对照：H1 已修、只关 causal_memory 时仍漏未来（证明上面的测试测得出记忆分量）。"""
    assert _future_leak(_build(causal_memory=False)) > 1e-4


@pytest.mark.parametrize('chunk_scan', [True, False])
def test_causal_memory_full_vs_incremental_parity(chunk_scan):
    m = _build(chunk_scan=chunk_scan)
    x = torch.randint(0, 50, (2, 20), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        full = m(x)
        y, past = m(x[:, :3], use_cache=True)
        ys = [y]
        for t in range(3, 20):
            y, past = m(x[:, t:t + 1], past_key_values=past, use_cache=True)
            ys.append(y)
    assert (full - torch.cat(ys, dim=1)).abs().max().item() < 1e-4
