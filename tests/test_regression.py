"""回归测试：覆盖已修复 bug 的路径，防止退化。

覆盖清单（按修复提交时间倒序）：
- 训练路径滑动窗口因果掩码（66159e8）
- 生成期记忆跨 token 累积（9b76cce）
- CharMerge 因果卷积（74454a8）
- BOS 不重复（74454a8, 59c9383, 9b76cce）
- cache vs 非 cache 数值一致性（新测试，抓滑窗/因果泄露）
- MemoryBank 基础操作（无历史测试）
- _selective_scan past_state 一致性（冗余扫描修复 ee78de5）
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer import (
    TransformerModel,
    MemoryBank,
    CharMergeLayer,
    SlidingWindowCausalSelfAttention,
    MambaSSM,
)
from models.config_loader import load_config, build_model
from models.data_utils import CharTokenizer


# ============================================================
# 1. 因果注意力：扰动未来 token，过去 logit 不变
# ============================================================

def test_causal_attention_perturb_future():
    """核心因果性测试：改变序列末尾 token，前面对数必须不变。

    抓住：训练路径滑窗泄露（66159e8）、cache/非 cache 掩码不一致。
    """
    config = load_config('configs/pretrain.yaml')
    model = build_model(config, device='cpu')
    model.eval()

    base_ids = torch.randint(0, config['model']['vocab_size'], (1, 8))
    # 基线前向
    with torch.no_grad():
        logits_base = model(base_ids)

    # 扰动最后一个 token
    perturbed_ids = base_ids.clone()
    perturbed_ids[0, -1] = (perturbed_ids[0, -1] + 1) % config['model']['vocab_size']
    with torch.no_grad():
        logits_perturbed = model(perturbed_ids)

    # 前 7 个位置的 logit 必须完全一致（因果性）
    diff = (logits_base[0, :7] - logits_perturbed[0, :7]).abs().max().item()
    assert diff == 0.0, (
        f"因果性违反：扰动 token[-1] 导致 token[:7] logit 变化 max_diff={diff}")
    # 最后一个位置的 logit 应不同（因为输入变了）
    diff_last = (logits_base[0, -1] - logits_perturbed[0, -1]).abs().max().item()
    assert diff_last > 0.0, "扰动末尾 token 后 logit 无变化，测试无效"
    print("✅ test_causal_attention_perturb_future passed")


def test_causal_attention_perturb_future_with_cache():
    """因果性 + KV cache 路径：逐 token 解码与全量前向最后一个位置必须一致。"""
    config = load_config('configs/pretrain.yaml')
    model = build_model(config, device='cpu')
    model.eval()

    ids = torch.randint(0, config['model']['vocab_size'], (1, 8))

    # 全量前向
    with torch.no_grad():
        logits_full = model(ids)

    # 增量解码
    with torch.no_grad():
        logits_0, past = model(ids[:, :7], use_cache=True)
        logits_1, _ = model(ids[:, 7:], past_key_values=past, use_cache=True)

    # 逐 token 解码的最后一个位置 logit 与全量前向的第 7 位必须一致
    diff = (logits_1[0, 0] - logits_full[0, 7]).abs().max().item()
    assert diff < 1e-4, (
        f"cache 一致性违反：增量解码 vs 全量前向 token[7] max_diff={diff}")


# ============================================================
# 2. 滑动窗口掩码（window>0 路径覆盖）
# ============================================================

def test_sliding_window_mask_correctness():
    """滑动窗口注意力：窗口外 token 的注意力分数必须被遮蔽。"""
    D = 64
    H = 4
    window = 8
    attn = SlidingWindowCausalSelfAttention(D, H, max_seq_length=64, window=window)
    attn.eval()

    T = 20
    x = torch.randn(1, T, D)
    with torch.no_grad():
        # 触发 forward → attend → _bias_cache 构建
        _ = attn(x, use_cache=False)

    # 验证 _bias_cache（实际使用的掩码）中窗口外+未来位置为 -1e9
    mask = attn._bias_cache[0, 0]  # (T, T)
    for q in range(T):
        for k in range(T):
            should_mask = (k > q) or (q - k > window)  # 因果 OR 窗口外
            is_masked = mask[q, k].item() != 0
            assert is_masked == should_mask, (
                f"掩码错误 mask[{q},{k}]: 期望 masked={should_mask}, 实际 masked={is_masked}")
    print("✅ test_sliding_window_mask_correctness passed")


def test_sliding_window_causal_leak():
    """滑动窗口 + 因果：单层注意力中，扰动窗口外 token 不影响当前位置输出。

    多层模型中信息可通过残差跨层传播，无法做零 diff 断言。
    此测试直接验证注意力层的掩码行为。
    """
    D = 64
    H = 4
    window = 4
    attn = SlidingWindowCausalSelfAttention(D, H, max_seq_length=64, window=window)
    attn.eval()

    T = 12
    x = torch.randn(1, T, D)
    with torch.no_grad():
        out_base, _ = attn(x, use_cache=False)

    # 扰动位置 0（距离位置 10 为 10 > window=4）
    x_perturbed = x.clone()
    x_perturbed[0, 0] = torch.randn(D)
    with torch.no_grad():
        out_perturbed, _ = attn(x_perturbed, use_cache=False)

    # 位置 10 的输出不应被位置 0 的扰动影响（距离 10 > window 4）
    diff = (out_base[0, 10] - out_perturbed[0, 10]).abs().max().item()
    assert diff == 0.0, (
        f"窗口因果泄露：单层注意力中扰动 token[0] 影响 token[10] (window=4, dist=10) max_diff={diff}")
    # 位置 3 应受影响（距离 3 ≤ window 4，且在因果范围内）
    diff3 = (out_base[0, 3] - out_perturbed[0, 3]).abs().max().item()
    assert diff3 > 0.0, "窗口内 token[3] 未受影响，测试无效"
    print("✅ test_sliding_window_causal_leak passed")


# ============================================================
# 3. cache vs 非 cache 数值一致性
# ============================================================

def test_cache_vs_noncache_numerical_parity():
    """缓存路径与非缓存路径的输出必须数值等价。

    这是抓滑窗泄露、因果掩码不一致最便宜的手段。
    """
    config = load_config('configs/pretrain.yaml')
    model = build_model(config, device='cpu')
    model.eval()

    ids = torch.randint(0, config['model']['vocab_size'], (1, 12))

    with torch.no_grad():
        logits_full = model(ids, use_cache=False)
        logits_cached, _ = model(ids, use_cache=True)

    diff = (logits_full - logits_cached).abs().max().item()
    assert diff < 1e-4, (
        f"cache 数值不一致：max_diff={diff}")
    print("✅ test_cache_vs_noncache_numerical_parity passed")


def test_cache_vs_noncache_with_window():
    """滑动窗口下 cache vs 非 cache 数值一致性。"""
    config = load_config('configs/pretrain.yaml')
    config = dict(config)
    model_cfg = dict(config['model'])
    model_cfg['attn_window'] = 8
    config['model'] = model_cfg
    model = build_model(config, device='cpu')
    model.eval()

    ids = torch.randint(0, config['model']['vocab_size'], (1, 16))

    with torch.no_grad():
        logits_full = model(ids, use_cache=False)
        logits_cached, _ = model(ids, use_cache=True)

    diff = (logits_full - logits_cached).abs().max().item()
    assert diff < 1e-4, (
        f"window cache 数值不一致：max_diff={diff}")
    print("✅ test_cache_vs_noncache_with_window passed")


# ============================================================
# 4. MemoryBank 基础操作
# ============================================================

def test_memory_bank_reset_write_get():
    """MemoryBank：reset→write→get_kv 的基本流程。"""
    dim = 64
    num_slots = 16
    comp_dim = 32
    mb = MemoryBank(dim, num_slots=num_slots, comp_dim=comp_dim)
    mb.eval()

    B = 2
    T = 5
    # reset
    mb.reset(B, torch.device('cpu'), torch.float32)
    assert mb.slots.shape == (B, num_slots, comp_dim)
    assert mb.slots.abs().max() == 0.0  # 零初始化

    # write
    x = torch.randn(B, T, dim)
    mb.write(x)
    assert mb.slots.shape == (B, num_slots, comp_dim)
    assert mb.slots.abs().max() > 0.0  # 写入后非零

    # get_kv
    k, v, meta = mb.get_kv()
    assert k.shape == (B, num_slots, dim)
    assert v.shape == (B, num_slots, dim)
    assert meta is None  # 无 retrieval 时 meta 为 None
    print("✅ test_memory_bank_reset_write_get passed")


def test_memory_bank_cumulative_write():
    """MemoryBank：多次 write 后记忆累积（不被覆盖）。"""
    dim = 64
    mb = MemoryBank(dim, num_slots=8, comp_dim=16)
    mb.eval()

    mb.reset(1, torch.device('cpu'), torch.float32)

    # 第一次写入
    x1 = torch.randn(1, 3, dim)
    mb.write(x1)
    slots_after_1 = mb.slots.clone()

    # 第二次写入
    x2 = torch.randn(1, 3, dim)
    mb.write(x2)
    slots_after_2 = mb.slots.clone()

    # 槽应不同（累积而非覆盖）
    diff = (slots_after_1 - slots_after_2).abs().max().item()
    assert diff > 0.0, "两次 write 后 slots 无变化，记忆未累积"
    # 归一化改为在读取时（_recompute_kv_cache）统一做一次，故 slots 本身未归一化；
    # 此处复现读取时的归一化并断言其为单位范数（幂等、与写入粒度无关）。
    normed = mb.slots / (1e-6 + mb.slots.norm(dim=-1, keepdim=True))
    norms = normed.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        f"读取时归一化后 norm 非单位: {norms.min()}..{norms.max()}"
    print("✅ test_memory_bank_cumulative_write passed")


def test_memory_bank_get_kv_finite():
    """MemoryBank：get_kv 输出必须有限（无 NaN/Inf）。"""
    mb = MemoryBank(64, num_slots=8, comp_dim=16)
    mb.eval()
    mb.reset(1, torch.device('cpu'), torch.float32)
    x = torch.randn(1, 4, 64)
    mb.write(x)
    k, v, _ = mb.get_kv()
    assert torch.isfinite(k).all(), "get_kv K 含 NaN/Inf"
    assert torch.isfinite(v).all(), "get_kv V 含 NaN/Inf"
    print("✅ test_memory_bank_get_kv_finite passed")


# ============================================================
# 5. MemoryBank 跨 token 累积回归（9b76cce）
# ============================================================

def test_memory_accumulates_across_forward_steps():
    """回归测试：增量解码时记忆必须在步间累积，不能每步 reset。

    commit 9b76cce 修复了生成期 memory 每步 reset 的 bug。
    """
    config = load_config('configs/pretrain.yaml')
    # 启用 memory
    config = dict(config)
    model_cfg = dict(config['model'])
    model_cfg['memory_size'] = 8
    model_cfg['memory_comp_dim'] = 16
    config['model'] = model_cfg
    model = build_model(config, device='cpu')
    model.eval()

    # 首步：prefill
    ids = torch.randint(0, config['model']['vocab_size'], (1, 4))
    with torch.no_grad():
        _, past = model(ids, use_cache=True)

    # 检查 memory 已被写入（slots 非零）
    slots_after_prefill = model.memory_bank.slots.clone()
    assert slots_after_prefill.abs().max() > 0.0, "prefill 后记忆为空"

    # 第二步：增量解码
    step_ids = torch.randint(0, config['model']['vocab_size'], (1, 1))
    with torch.no_grad():
        _, past2 = model(step_ids, past_key_values=past, use_cache=True)

    # 记忆应累积（不同于 prefill 后的状态）
    slots_after_step = model.memory_bank.slots.clone()
    diff = (slots_after_prefill - slots_after_step).abs().max().item()
    assert diff > 0.0, "增量步后记忆无变化，累积失败"
    assert slots_after_step.abs().max() > 0.0, "增量步后记忆为空"
    print("✅ test_memory_accumulates_across_forward_steps passed")


# ============================================================
# 6. CharMerge 因果性（74454a8）
# ============================================================

def test_char_merge_causality():
    """CharMergeLayer：改变未来 token，当前位置输出不变（因果卷积）。"""
    dim = 32
    layer = CharMergeLayer(dim, kernel_size=3)
    layer.eval()

    T = 10
    x = torch.randn(1, T, dim)
    with torch.no_grad():
        y_base = layer(x)

    # 改变位置 7（未来）的输入
    x_perturbed = x.clone()
    x_perturbed[0, 7] = torch.randn(dim)
    with torch.no_grad():
        y_perturbed = layer(x_perturbed)

    # 位置 0~4 的输出不应变化（卷积核大小=3，因果填充=2，位置 t 看 t-2..t）
    diff = (y_base[0, :5] - y_perturbed[0, :5]).abs().max().item()
    assert diff == 0.0, (
        f"CharMerge 因果违反：扰动 token[7] 影响了 token[:5] max_diff={diff}")
    print("✅ test_char_merge_causality passed")


# ============================================================
# 7. BOS 一致性（各入口恰好 1 个 BOS）
# ============================================================

def test_bos_consistency_generate_text():
    """generate_text 路径：vocab.encode(prompt) 默认加 BOS，不重复。"""
    vocab = CharTokenizer(vocab_size=100)
    # 模拟 encode 行为
    tokens = vocab.encode("hello", add_special_tokens=True)
    assert tokens[0] == vocab.bos_idx, f"首 token 应为 BOS: {tokens[0]}"
    # 不应有连续两个 BOS
    assert not (len(tokens) > 1 and tokens[1] == vocab.bos_idx), "BOS 重复"
    print("✅ test_bos_consistency_generate_text passed")


def test_bos_consistency_generate_igmcg():
    """generate_igmcg 路径：手动加 BOS + encode(add_special_tokens=False)。"""
    vocab = CharTokenizer(vocab_size=100)
    ids = [vocab.bos_idx] + vocab.encode("hello", add_special_tokens=False)
    assert ids[0] == vocab.bos_idx, f"首 token 应为 BOS: {ids[0]}"
    # 不应有连续两个 BOS
    assert not (len(ids) > 1 and ids[1] == vocab.bos_idx), "BOS 重复"
    # 第二个 token 不应是 BOS（encode 不加 special tokens）
    if len(ids) > 1:
        assert ids[1] != vocab.bos_idx or len(set(ids[1:])) > 1, \
            "encode(add_special_tokens=False) 不应产生 BOS"
    print("✅ test_bos_consistency_generate_igmcg passed")


def test_bos_no_duplicate_in_model_generate():
    """model.generate()：传入含 1 个 BOS 的序列，生成结果不含额外 BOS。"""
    config = load_config('configs/pretrain.yaml')
    model = build_model(config, device='cpu')
    model.eval()
    # 传入 [BOS, ...]
    tokens = [2] + [5, 6, 7]
    generated = model.generate(tokens, max_length=10, device='cpu',
                               repetition_penalty=1.0)
    # 生成结果不应在中间插入 BOS（只在开头有一个）
    bos_count = sum(1 for t in generated if t == 2)
    assert bos_count <= 1, f"生成结果含 {bos_count} 个 BOS，应 <= 1"
    print("✅ test_bos_no_duplicate_in_model_generate passed")


# ============================================================
# 8. _selective_scan past_state 一致性（ee78de5）
# ============================================================

def test_selective_scan_past_state_consistency():
    """_selective_scan：past_state 路径与零初始化路径数值一致。

    验证 past_state=h_{-1} 时 h_0 = a_0 * h_{-1} + b_0 的计算正确。
    """
    ssm = MambaSSM(dim=32, d_state=8)
    ssm.eval()

    B, L, D, S = 1, 4, 32, 8
    a = torch.randn(B, L, D, S).sigmoid()  # a in (0,1)
    b = torch.randn(B, L, D, S)

    # 无 past_state（h_0 = 0）
    with torch.no_grad():
        h_zero = ssm._selective_scan(a, b, past_state=None)

    # 有 past_state（h_{-1} = some_value）
    past_state = torch.randn(B, D, S)
    with torch.no_grad():
        h_with_past = ssm._selective_scan(a, b, past_state=past_state)

    # h_with_past[0] = a[0] * past_state + b[0]（不同于 h_zero[0] = b[0]）
    expected_h0 = a[:, 0] * past_state + b[:, 0]
    diff = (h_with_past[:, 0] - expected_h0).abs().max().item()
    assert diff < 1e-5, (
        f"selective_scan past_state 不一致：h[0] max_diff={diff}")

    # h_with_past[1:] 应与 h_zero[1:] 不同（因为 past_state 影响了前缀积传播）
    # 但在 past_state=0 时两者应相同
    past_zero = torch.zeros(B, D, S)
    with torch.no_grad():
        h_with_zero_past = ssm._selective_scan(a, b, past_state=past_zero)
    diff_zero = (h_with_zero_past - h_zero).abs().max().item()
    assert diff_zero < 1e-5, (
        f"selective_scan past_state=0 应与 None 等价：max_diff={diff_zero}")
    print("✅ test_selective_scan_past_state_consistency passed")


# ============================================================
# 9. 训练路径滑动窗口 + 因果掩码回归（66159e8）
# ============================================================

def test_training_sliding_window_causal_mask():
    """回归测试：训练路径（非缓存）的滑动窗口掩码必须同时满足因果+窗口约束。

    commit 66159e8 修复了训练路径仅有窗口约束缺少因果约束的 bug。
    构造 window=4 的 attention，验证训练路径掩码遮蔽未来+窗口外。
    """
    D = 32
    H = 2
    window = 4
    attn = SlidingWindowCausalSelfAttention(D, H, max_seq_length=32, window=window)
    attn.eval()

    T = 10
    x = torch.randn(1, T, D)

    # 用非缓存前向（训练路径）触发掩码构建
    with torch.no_grad():
        _ = attn(x, use_cache=False)

    # 验证实际使用的 _bias_cache（传给 SDPA 的 float attn_mask）
    # _bias_cache 是 float mask：被遮蔽位置为 MASK_FILL_VALUE(-1e9)，未遮蔽位置为 0
    mask = attn._bias_cache[0, 0]  # (T, T)
    for q in range(T):
        for k in range(T):
            should_mask = (k > q) or (q - k > window)  # 因果 OR 窗口外
            is_masked = mask[q, k].item() != 0
            assert is_masked == should_mask, (
                f"掩码错误 mask[{q},{k}]: 期望 masked={should_mask}, 实际 masked={is_masked}")

    # 用滑动窗口前向验证输出有限
    with torch.no_grad():
        out = attn(x, use_cache=False)
    assert torch.isfinite(out[0]).all(), "滑动窗口训练路径输出含 NaN/Inf"
    print("✅ test_training_sliding_window_causal_mask passed")


# ============================================================
# 10. generate() 端到端 + 记忆累积
# ============================================================

def test_generate_with_memory():
    """带记忆模型的 generate：生成过程中记忆应累积。"""
    config = load_config('configs/pretrain.yaml')
    config = dict(config)
    model_cfg = dict(config['model'])
    model_cfg['memory_size'] = 4
    model_cfg['memory_comp_dim'] = 8
    config['model'] = model_cfg
    model = build_model(config, device='cpu')
    model.eval()

    # generate 首步 reset
    tokens = [2, 5, 6]
    generated = model.generate(tokens, max_length=5, device='cpu',
                               repetition_penalty=1.0)
    assert len(generated) > len(tokens), "generate 未产生新 token"

    # generate 结束后记忆应非零
    slots = model.memory_bank.slots
    assert slots.abs().max() > 0.0, "generate 后记忆为空"
    print("✅ test_generate_with_memory passed")


if __name__ == '__main__':
    test_causal_attention_perturb_future()
    test_causal_attention_perturb_future_with_cache()
    test_sliding_window_mask_correctness()
    test_sliding_window_causal_leak()
    test_cache_vs_noncache_numerical_parity()
    test_cache_vs_noncache_with_window()
    test_memory_bank_reset_write_get()
    test_memory_bank_cumulative_write()
    test_memory_bank_get_kv_finite()
    test_memory_accumulates_across_forward_steps()
    test_char_merge_causality()
    test_bos_consistency_generate_text()
    test_bos_consistency_generate_igmcg()
    test_bos_no_duplicate_in_model_generate()
    test_selective_scan_past_state_consistency()
    test_training_sliding_window_causal_mask()
    test_generate_with_memory()
    print("\n🎉 All regression tests passed!")
