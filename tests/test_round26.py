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
