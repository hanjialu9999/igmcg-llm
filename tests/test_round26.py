"""第二十六轮回归测试：审查修复（B1/B3/B4 + 性能批量化）。

覆盖：
- B3: DifferentialAttention `_causal_mask` persistent=False（不进 state_dict）
- B4: DifferentialAttention T > max_seq_length 动态扩展（不静默缩小）
- B1: IGMCG + ngram_fusion (N,) 张量温度不崩溃
- 性能: _ngram_coherence 批量化与逐 token 数值等价
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer import TransformerModel


def _small(**over):
    kw = dict(vocab_size=200, embedding_dim=64, num_heads=4, num_layers=2,
              hidden_dim=128, max_seq_length=32)
    kw.update(over)
    return TransformerModel(**kw)


# ─── B3: DiffAttn _causal_mask 不进 state_dict ─────────────────────────────

def test_diff_attn_causal_mask_not_in_state_dict():
    """B3 回归：DifferentialAttention 的 `_causal_mask` buffer 应被 persistent=False
    标记，不进入 state_dict，避免 train/infer max_seq_length 不一致时
    load_state_dict 报 shape mismatch（strict=False 只忽略 missing/extra keys，
    不忽略 shape mismatch）。"""
    m = _small(mixer='diff', max_seq_length=32)
    sd = m.state_dict()
    # 不应包含任何 _causal_mask 条目
    causal_keys = [k for k in sd.keys() if '_causal_mask' in k]
    assert not causal_keys, f"_causal_mask 不应在 state_dict 中，但找到：{causal_keys}"
    # 但普通参数（如 qkv.weight）应正常存在
    assert any('qkv' in k for k in sd.keys()), "普通 qkv 参数缺失，state_dict 异常"


def test_diff_attn_causal_mask_load_with_different_max_seq_length():
    """B3 端到端：训练时 max_seq_length=32 保存的 state_dict，加载到 max_seq_length=16
    的模型不应崩溃（_causal_mask 不在 state_dict 中，无 shape 冲突）。"""
    m_src = _small(mixer='diff', max_seq_length=32)
    m_dst = _small(mixer='diff', max_seq_length=16)
    sd = m_src.state_dict()
    # 加载到不同 max_seq_length 的模型（strict=False 允许 missing/extra）
    m_dst.load_state_dict(sd, strict=False)
    # 前向验证
    m_dst.eval()
    x = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        out = m_dst(x)
    assert out.shape == (1, 8, 200)
    assert torch.isfinite(out).all()


# ─── B4: DiffAttn T > max_seq_length 动态扩展 ───────────────────────────────

def test_diff_attn_T_exceeds_max_seq_length():
    """B4 回归：DifferentialAttention 在 T > max_seq_length 时应动态扩展因果掩码，
    而非切片静默缩小（原 bug：causal = self._causal_mask[:, :, :T, :T] 当 T>size
    时返回 (1,1,max_seq_length,max_seq_length) 与 scores (B,H,T,T) 广播失败）。"""
    m = _small(mixer='diff', max_seq_length=8)
    m.eval()
    # T=16 > max_seq_length=8
    x = torch.randint(0, 200, (1, 16))
    with torch.no_grad():
        out = m(x)
    assert out.shape == (1, 16, 200), f"输出形状错误：{out.shape}"
    assert torch.isfinite(out).all(), "T>max_seq_length 路径输出含 nan/inf"


def test_diff_attn_T_equals_max_seq_length_boundary():
    """B4 边界：T == max_seq_length 时走 buffer 切片路径（不触发动态扩展）。"""
    m = _small(mixer='diff', max_seq_length=8)
    m.eval()
    x = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        out = m(x)
    assert out.shape == (1, 8, 200)
    assert torch.isfinite(out).all()


# ─── B1: IGMCG + ngram_fusion 张量温度 ──────────────────────────────────────

def _build_ngram_fusion_model():
    """构造启用 ngram_fusion 的小模型（复用 test_new_mechanisms._small_ngram 模式）。"""
    from tests.test_new_mechanisms import _small_ngram
    v, ng = _small_ngram()
    m = _small(vocab_size=len(v), ngram_fusion=True, ngram_model=ng)
    return m, v


def test_ngram_fusion_tensor_temperature_not_crash():
    """B1 回归：IGMCG 批量候选路径传 (N,) 张量温度时，_apply_ngram_fusion 中
    float(temperature) 不应崩溃（原 bug：对 (N,) 且 N>1 的张量调 float() 抛
    ValueError: only one element tensors can be converted to Python scalars）。

    触发条件：ngram_fusion=True + forward(temperature=(N,) 张量)。
    修复：isinstance(temperature, torch.Tensor) and numel()>1 分支用 clamp+view。"""
    m, v = _build_ngram_fusion_model()
    m.eval()
    N, T = 3, 6
    inp = torch.randint(0, len(v), (N, T))
    temps = torch.tensor([0.8, 1.0, 1.2], dtype=torch.float32)
    with torch.no_grad():
        out = m(inp, temperature=temps)
    assert out.shape == (N, T, len(v)), f"输出形状错误：{out.shape}"
    assert torch.isfinite(out).all(), "张量温度路径输出含 nan/inf"


def test_ngram_fusion_scalar_temperature_still_works():
    """B1 兼容性：标量温度路径不受影响（向后兼容）。"""
    m, v = _build_ngram_fusion_model()
    m.eval()
    inp = torch.randint(0, len(v), (1, 6))
    with torch.no_grad():
        out_scalar = m(inp, temperature=1.0)
        out_tensor1 = m(inp, temperature=torch.tensor(1.0))
    assert out_scalar.shape == (1, 6, len(v))
    assert out_tensor1.shape == (1, 6, len(v))
    # 标量与单元素张量应数值一致
    assert torch.allclose(out_scalar, out_tensor1, atol=1e-6), \
        "标量温度与单元素张量温度输出不一致"


def test_ngram_fusion_tensor_temperature_clamp():
    """B1 边界：(N,) 张量温度含越界值时被 clamp 到 [0.01, 10]。"""
    m, v = _build_ngram_fusion_model()
    m.eval()
    N, T = 2, 4
    inp = torch.randint(0, len(v), (N, T))
    # 含越界值（0.001 < 0.01，20 > 10）
    temps = torch.tensor([0.001, 20.0], dtype=torch.float32)
    with torch.no_grad():
        out = m(inp, temperature=temps)
    assert out.shape == (N, T, len(v))
    assert torch.isfinite(out).all(), "越界温度 clamp 后仍含 nan/inf"


# ─── 性能: _ngram_coherence 批量化数值等价 ──────────────────────────────────

def test_ngram_coherence_batch_equivalence():
    """性能回归：_ngram_coherence 批量化（torch.stack + 一次 .item()）
    与逐 token .item() 累加数值等价（atol=1e-6）。

    修复背景：原实现每 token 调 .item() 产生 DML→CPU 同步税（~100-200μs/次），
    60 token 序列产生 60 次同步；批量化后仅 1 次。数值应完全等价。"""
    from tests.test_new_mechanisms import _small_ngram
    from scripts.generate import _ngram_coherence
    v, ng = _small_ngram()
    ngram_fn = ng.logprob_vector
    # 构造 8 token 序列
    ids = [10, 20, 30, 40, 50, 60, 70, 80]
    # 批量化版本（当前实现）
    batched = _ngram_coherence(ngram_fn, ids, 'cpu')
    # 逐 token .item() 基线（原实现）
    tot, n = 0.0, 0
    for i in range(1, len(ids)):
        lp = ngram_fn(ids[:i], 'cpu')
        tot += lp[ids[i]].item()
        n += 1
    baseline = tot / max(1, n)
    assert abs(batched - baseline) < 1e-6, \
        f"批量化与逐 token 不一致：batched={batched}, baseline={baseline}"


def test_ngram_coherence_empty_and_short():
    """性能回归：_ngram_coherence 边界——ngram_fn=None 或 len(ids)<2 返回 0。"""
    from scripts.generate import _ngram_coherence
    assert _ngram_coherence(None, [1, 2, 3], 'cpu') == 0.0
    assert _ngram_coherence(None, [], 'cpu') == 0.0
    # 单 token 序列
    assert _ngram_coherence(None, [1], 'cpu') == 0.0


# ─── B2: MLA + VRC 增量解码快速分支 ─────────────────────────────────────────

def test_mla_vrc_cache_parity_nonzero_lambda():
    """B2 回归：MLA + VRC 在 λ≠0 时 train/infer cache parity。

    修复前：MLA 增量解码每步解压全部历史 token（v.size(2)=T_total>1），
    落入 elif v.size(2)>1 分支做 prefix scan；但全量前向也做 prefix scan，
    理论上应一致——然而 MLA present=(c_kv_full, None) 不缓存编码 V，
    每步重算 prefix scan 致 O(N² log N)，且浮点舍入累积可能 >1e-4。

    修复后：present=(c_kv_full, v_encoded)，增量步只编码新 token（O(1)），
    与全量 prefix scan 数学等价（atol=1e-4）。
    """
    torch.manual_seed(42)
    m = _small(use_mla_kv=True, kv_latent_dim=32,
               value_relative_coding=True, alibi=True, num_layers=2)
    m.eval()
    # 设置非零 λ 以暴露递推路径差异
    with torch.no_grad():
        m.blocks[0].attn.value_rel_lambda.fill_(0.5)
        m.blocks[1].attn.value_rel_lambda.fill_(0.5)
    x = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        full = m(x, use_cache=False)
        out, past = m(x[:, :1], past_key_values=None, use_cache=True)
        for t in range(1, 8):
            out, past = m(x[:, t:t + 1], past_key_values=past, use_cache=True)
    diff = (full[:, -1, :] - out[:, -1, :]).abs().max().item()
    assert diff < 1e-4, f"MLA+VRC λ=0.5 cache parity max_diff={diff:.2e} 超过 1e-4"


def test_mla_vrc_present_stores_encoded_v():
    """B2 回归：MLA+VRC 时 present[1] 应存编码后 V（非 None），供下一步快速分支复用。

    修复前：MLA present=(c_kv_full, None)，VRC 快速分支条件 past_kv[1] is not None 永不命中。
    修复后：MLA+VRC present=(c_kv_full, v_encoded)，快速分支命中。

    past 结构：past[layer] = (attn_kv, ssm_hidden, ssm_conv)
    attn_kv = (c_kv_full, v_encoded) for MLA+VRC
    """
    m = _small(use_mla_kv=True, kv_latent_dim=32,
               value_relative_coding=True, alibi=True, num_layers=2)
    m.eval()
    x = torch.randint(0, 200, (1, 4))
    with torch.no_grad():
        # 第一步（prefill 2 token）
        out, past = m(x[:, :2], past_key_values=None, use_cache=True)
    # past[0] = (attn_kv, ssm_hidden, ssm_conv)；attn_kv = (c_kv_full, v_encoded)
    attn_kv = past[0][0]
    assert attn_kv[1] is not None, \
        "MLA+VRC attn_kv[1] 应存编码 V（非 None），供快速分支复用"
    # v_encoded 形状应为 (B, H, T, hd) = (1, 4, 2, 16)
    assert attn_kv[1].shape == (1, 4, 2, 16), \
        f"v_encoded 形状错误：{attn_kv[1].shape}，期望 (1, 4, 2, 16)"


def test_mla_without_vrc_present_none():
    """B2 兼容性：MLA 无 VRC 时 attn_kv[1] 仍为 None（向后兼容，省内存）。"""
    m = _small(use_mla_kv=True, kv_latent_dim=32, alibi=True, num_layers=2)
    m.eval()
    x = torch.randint(0, 200, (1, 4))
    with torch.no_grad():
        out, past = m(x[:, :2], past_key_values=None, use_cache=True)
    attn_kv = past[0][0]
    assert attn_kv[1] is None, \
        "MLA 无 VRC 时 attn_kv[1] 应为 None（向后兼容），实际非 None"


def test_mla_vrc_fast_path_matches_prefix_scan():
    """B2 数值等价：MLA+VRC 增量快速分支（O(1) per step）与全量 prefix scan 数值一致。

    构造 8 token 序列，分别用：
    - 全量前向（prefix scan over T=8）
    - 增量解码（快速分支：第 1 步 prefix scan T=1，第 2..8 步 O(1) 快速分支）
    比较末位 logits。
    """
    torch.manual_seed(123)
    m = _small(use_mla_kv=True, kv_latent_dim=32,
               value_relative_coding=True, alibi=True, num_layers=1)
    m.eval()
    with torch.no_grad():
        m.blocks[0].attn.value_rel_lambda.fill_(0.3)
    x = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        full = m(x, use_cache=False)
        out, past = m(x[:, :1], past_key_values=None, use_cache=True)
        for t in range(1, 8):
            out, past = m(x[:, t:t + 1], past_key_values=past, use_cache=True)
    diff = (full[:, -1, :] - out[:, -1, :]).abs().max().item()
    assert diff < 1e-4, \
        f"MLA+VRC 快速分支 vs prefix scan max_diff={diff:.2e} 超过 1e-4"


# ─── 批次3-6: 新特性组合测试（R21-R25 交互验证）─────────────────────────────

def test_r21_r25_all_features_forward():
    """批次3-6: R21-R25 全特性组合 forward 不崩溃 + 输出有限。

    验证以下特性同时开启时无冲突：
    - R21: head_temp / value_relative_coding / rwkv7
    - R22: intra_hybrid_rope
    - R23: gpas
    - R25: alibi_learnable
    - R17: use_mla_kv（与 VRC 交互经 B2 修复）
    - R15: output_gate / zero_centered_norm（经 ModelConfig）
    """
    m = _small(
        num_layers=3,
        mixer='gated_delta',
        alibi=True, alibi_learnable=True,
        head_temp=True, value_relative_coding=True,
        rwkv7=True, gated_delta_channel_wise=True,
        intra_hybrid_rope=True,
        gpas=True, zero_centered_norm=True,
        output_gate=True,
        use_mla_kv=True, kv_latent_dim=32,
        dim_wise_rope=True, rope_dim_fraction=0.5,
    )
    m.eval()
    x = torch.randint(0, 200, (2, 8))
    with torch.no_grad():
        out = m(x)
    assert out.shape == (2, 8, 200), f"输出形状错误: {out.shape}"
    assert torch.isfinite(out).all(), "全特性组合输出含 nan/inf"


def test_alibi_learnable_with_shared_alibi_and_pe_gate():
    """批次3-6: alibi_learnable + shared_alibi + pe_gate 三特性组合。

    验证位置编码三特性正交兼容：
    - alibi_learnable: 斜率可学（Parameter）
    - shared_alibi: 所有层共享同一 Parameter 对象
    - pe_gate: per-head 位置信号强度门控
    """
    m = _small(
        num_layers=3,
        alibi=True, alibi_learnable=True,
        shared_alibi=True, pe_gate=True,
    )
    m.eval()
    # shared_alibi: 所有层 attn.alibi_slopes 应是同一对象
    slopes0 = m.blocks[0].attn.alibi_slopes
    slopes1 = m.blocks[1].attn.alibi_slopes
    assert slopes0 is slopes1, "shared_alibi 时 alibi_slopes 应共享同一 Parameter 对象"
    # alibi_learnable: 应是 Parameter（requires_grad=True）
    assert isinstance(slopes0, torch.nn.Parameter), "alibi_learnable 时 alibi_slopes 应为 Parameter"
    # pe_gate: log_pe_gate 应存在
    assert hasattr(m.blocks[0].attn, 'log_pe_gate'), "pe_gate 时 log_pe_gate 应存在"
    # forward 不崩溃
    x = torch.randint(0, 200, (1, 6))
    with torch.no_grad():
        out = m(x)
    assert out.shape == (1, 6, 200)
    assert torch.isfinite(out).all()


def test_mla_vrc_output_gate_combo():
    """批次3-6: MLA + VRC + output_gate 三特性组合 + cache parity。

    这三个特性都修改 attend 路径，验证组合后 train/infer 一致。
    """
    torch.manual_seed(77)
    m = _small(
        num_layers=2,
        use_mla_kv=True, kv_latent_dim=32,
        value_relative_coding=True, alibi=True,
        output_gate=True,
    )
    m.eval()
    with torch.no_grad():
        m.blocks[0].attn.value_rel_lambda.fill_(0.4)
        m.blocks[1].attn.value_rel_lambda.fill_(0.4)
    x = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        full = m(x, use_cache=False)
        out, past = m(x[:, :1], past_key_values=None, use_cache=True)
        for t in range(1, 8):
            out, past = m(x[:, t:t + 1], past_key_values=past, use_cache=True)
    diff = (full[:, -1, :] - out[:, -1, :]).abs().max().item()
    assert diff < 1e-4, f"MLA+VRC+output_gate cache parity diff={diff:.2e}"
