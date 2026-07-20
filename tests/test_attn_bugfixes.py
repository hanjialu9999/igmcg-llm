import torch

from models.transformer import TransformerModel
from models.memory import MemoryBank


def _run_full(m, ids, **kw):
    m.eval()
    with torch.no_grad():
        return m(torch.tensor([ids], dtype=torch.long), use_cache=False, **kw)[0, -1]


def _run_incremental(m, ids, **kw):
    m.eval()
    m.reset_ngram_state()
    with torch.no_grad():
        past = None
        for t in ids:
            logits, past = m(torch.tensor([[t]], dtype=torch.long),
                             past_key_values=past, use_cache=True, **kw)
    return logits[0, -1]


def test_rel_bias_generate_no_nameerror():
    """Bug A 回归：attn_rel_bias=True 时增量解码（generate/use_cache）不得 NameError。"""
    m = TransformerModel(vocab_size=100, embedding_dim=32, hidden_dim=64,
                         num_heads=4, num_layers=2, max_seq_length=32,
                         attn_rel_bias=True)
    out = _run_incremental(m, [1, 2, 3, 4])
    assert out.shape[0] == 100


def test_rel_bias_cache_matches_full():
    """Bug A 回归：attn_rel_bias=True（无记忆）时，增量解码与全量前向最后位置输出一致。"""
    m = TransformerModel(vocab_size=100, embedding_dim=32, hidden_dim=64,
                         num_heads=4, num_layers=2, max_seq_length=32,
                         attn_rel_bias=True)
    ids = [5, 3, 8, 1, 9, 2]
    full, incr = _run_full(m, ids), _run_incremental(m, ids)
    assert torch.allclose(full, incr, atol=1e-4), \
        f"rel_bias 增量/全量不一致 max_diff={(full-incr).abs().max():.3e}"


def test_rel_bias_memory_train_no_shape_error():
    """Bug B 回归：attn_rel_bias=True + 记忆开启时训练前向不得因 self._mask 形状不符崩溃。"""
    m = TransformerModel(vocab_size=100, embedding_dim=32, hidden_dim=64,
                         num_heads=4, num_layers=2, max_seq_length=32,
                         attn_rel_bias=True, memory_size=16, memory_comp_dim=16,
                         memory_retrieval=True)
    m.train()
    out = m(torch.randint(0, 100, (2, 6)))          # 不得 RuntimeError: size mismatch
    assert out.shape[:2] == (2, 6)


def test_rel_bias_memory_incremental_no_crash():
    """Bug A+B 回归：rel_bias + 记忆组合下增量解码不得 NameError/崩溃，且输出有限。"""
    m = TransformerModel(vocab_size=100, embedding_dim=32, hidden_dim=64,
                         num_heads=4, num_layers=2, max_seq_length=32,
                         attn_rel_bias=True, memory_size=16, memory_comp_dim=16,
                         memory_retrieval=True)
    out = _run_incremental(m, [5, 3, 8, 1, 9, 2])
    assert torch.isfinite(out).all(), "rel_bias+记忆 增量解码输出含非有限值"


def test_retrieval_gate_sigmoid_applied():
    """Bug C 回归：_full_retrieval_bias 对 retrieval_gate 须与 inject_memory 一致做 sigmoid。

    关键判别：gate=10 时 sigmoid(10)≈1，输出应≈ gate=None（不缩放）；若错误地用 raw gate
    直接乘，则会被放大约 10 倍，与 gate=None 明显不同。由此证明函数内部对 gate 做了 sigmoid。
    """
    from models.mixers import SlidingWindowCausalSelfAttention
    attn = SlidingWindowCausalSelfAttention(dim=16, num_heads=2, window=4,
                                            retrieval_full=True, max_seq_length=32)
    attn.retrieval_topk = 4
    attn.rel_bias = False
    attn.eval()
    dev = next(attn.parameters()).device
    B, H, Tq, Treal, D = 1, 2, 3, 6, 16
    q = torch.randn(B, H, Tq, D)
    k = torch.randn(B, H, Treal, D)
    with torch.no_grad():
        rbias_none = attn._full_retrieval_bias(q, k, Treal, 0, None, dev)
        attn.retrieval_gate = torch.nn.Parameter(torch.full((1,), 10.0))
        rbias_s = attn._full_retrieval_bias(q, k, Treal, 0, attn.retrieval_gate, dev)
    assert rbias_none is not None and rbias_s is not None
    # sigmoid(10)≈0.99995，故 rbias_s ≈ rbias_none（仅结构掩码 + 极小区分）；
    # 若用 raw gate 乘则会放大约 10 倍，allclose 必失败。
    assert torch.allclose(rbias_none, rbias_s, atol=1e-2, rtol=1e-2), \
        "retrieval_gate 未做 sigmoid（gate=10 应≈gate=None；raw 乘会放大约10倍）"


def test_alibi_memory_columns_zero():
    """Bug D 回归：alibi=True 时 _alibi_bias 须在记忆列（前 mem_cols 列）清零，
    且主序列段 ALiBi 不变；增量解码（start_pos 大）也须如此。"""
    from models.mixers import SlidingWindowCausalSelfAttention
    attn = SlidingWindowCausalSelfAttention(dim=16, num_heads=2, alibi=True,
                                            max_seq_length=32)
    attn.eval()
    dev = next(attn.parameters()).device
    mem_cols = 16
    with torch.no_grad():
        full = attn._alibi_bias(6, 6 + mem_cols, dev, start_pos=0, mem_cols=mem_cols)
        cache = attn._alibi_bias(1, 1 + mem_cols, dev, start_pos=20, mem_cols=mem_cols)
    assert full is not None and cache is not None
    assert torch.allclose(full[..., :mem_cols], torch.zeros_like(full[..., :mem_cols])), \
        "全量路径记忆列 ALiBi 偏置未清零"
    assert torch.allclose(cache[..., :mem_cols], torch.zeros_like(cache[..., :mem_cols])), \
        "增量路径记忆列 ALiBi 偏置未清零（生成时记忆被位置偏置压制）"
    # 主序列段（记忆列之后）ALiBi 应非全 0（确有位置偏置）
    assert not torch.allclose(full[..., mem_cols:],
                              torch.zeros_like(full[..., mem_cols:])), \
        "主序列段 ALiBi 被错误清零"


def test_alibi_cache_matches_full():
    """Bug D 回归：alibi=True（无记忆）时增量解码与全量前向输出一致。"""
    m = TransformerModel(vocab_size=100, embedding_dim=32, hidden_dim=64,
                         num_heads=4, num_layers=2, max_seq_length=32, alibi=True)
    ids = [5, 3, 8, 1, 9, 2]
    full, incr = _run_full(m, ids), _run_incremental(m, ids)
    assert torch.allclose(full, incr, atol=1e-4), \
        f"alibi 增量/全量不一致 max_diff={(full-incr).abs().max():.3e}"


def test_memory_write_granularity_independent():
    """记忆归一化回归：_recompute_kv_cache 统一归一化后，全量 write 与逐 token write
    产生的记忆 K/V 必须逐位一致（修复前逐写归一化导致 norm(a+b)≠norm(norm(a)+b) 的
    训练-推理记忆 divergence）。"""
    mb = MemoryBank(dim=32, num_slots=16, comp_dim=16, retrieval=True)
    mb.eval()
    dev = next(mb.parameters()).device
    B, T, D = 2, 6, 32
    x = torch.randn(B, T, D)
    mb.reset(B, dev, torch.float)
    with torch.no_grad():
        mb.write(x)
        kf, vf, _ = mb.get_kv()
    mb2 = MemoryBank(dim=32, num_slots=16, comp_dim=16, retrieval=True)
    mb2.eval()
    mb2.load_state_dict(mb.state_dict())
    mb2.reset(B, dev, torch.float)
    with torch.no_grad():
        for t in range(T):
            mb2.write(x[:, t:t + 1])
        ki, vi, _ = mb2.get_kv()
    assert torch.allclose(kf, ki, atol=1e-5), \
        f"全量/逐token 记忆 K 不一致 max_diff={(kf-ki).abs().max():.3e}"
    assert torch.allclose(vf, vi, atol=1e-5), \
        f"全量/逐token 记忆 V 不一致 max_diff={(vf-vi).abs().max():.3e}"


def test_mamba_selective_scan_roll_parity():
    """#4 回归：roll 版 _selective_scan 与朴素顺序前缀扫描逐位一致（半群结合律不变）。"""
    from models.mixers import MambaSSM
    Bs, L, d_in, d_st = 2, 12, 8, 5
    mamba = MambaSSM(dim=8, d_state=d_st)
    mamba.eval()
    torch.manual_seed(0)
    a = torch.randn(Bs, L, d_in, d_st)
    b = torch.randn(Bs, L, d_in, d_st)
    with torch.no_grad():
        h_scan = mamba._selective_scan(a, b)
        h_ref = torch.zeros(Bs, d_in, d_st)
        h_list = []
        for t in range(L):
            h_ref = a[:, t] * h_ref + b[:, t]
            h_list.append(h_ref.clone())
        h_seq = torch.stack(h_list, dim=1)
    assert torch.allclose(h_scan, h_seq, atol=1e-5), \
        f"roll 版选择性扫描与顺序前缀不一致 max_diff={(h_scan-h_seq).abs().max():.3e}"
    past = h_scan[:, -1]
    a2 = torch.randn(Bs, 3, d_in, d_st)
    b2 = torch.randn(Bs, 3, d_in, d_st)
    with torch.no_grad():
        h2 = mamba._selective_scan(a2, b2, past_state=past)
    h_ref2 = past.clone()
    h2_list = []
    for t in range(3):
        h_ref2 = a2[:, t] * h_ref2 + b2[:, t]
        h2_list.append(h_ref2.clone())
    h2_seq = torch.stack(h2_list, dim=1)
    assert torch.allclose(h2, h2_seq, atol=1e-5), \
        f"带 past_state 的选择性扫描不一致 max_diff={(h2-h2_seq).abs().max():.3e}"
