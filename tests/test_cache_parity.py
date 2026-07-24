"""增量解码 vs 全量前向 logits 一致性回归测试。

覆盖第十七轮审查修复的 cache 协议场景：
- MLA + attn_linear hybrid（MEDIUM #3：_accum_kv 修复维度不匹配）
- GatedDeltaNet（CRITICAL #1：start_pos 累积 + 输出布局修复）
- LinearAttention（CRITICAL #1：start_pos 累积）
- AxialLinearAttention（2D 训练 vs 1D 推理，已知架构取舍，标 xfail）
- Standard attn / MLA alone / attn_linear hybrid 基线

阈值 1e-4：fused SDPA + float32 下数值误差量级。
"""
import pytest
import torch

from models.transformer import TransformerModel


def _get_logits(out):
    if isinstance(out, dict):
        return out["logits"]
    if isinstance(out, tuple):
        return out[0]
    return out


def _check_parity(m, seq_len=8):
    m.eval()
    torch.manual_seed(42)
    ids = torch.randint(0, 200, (1, seq_len))
    with torch.no_grad():
        full = _get_logits(m(ids, use_cache=False))
        out, past = m(ids[:, :1], past_key_values=None, use_cache=True)
        for t in range(1, seq_len):
            out, past = m(ids[:, t:t + 1], past_key_values=past, use_cache=True)
        incr = _get_logits(out)
    return (full[:, -1, :] - incr[:, -1, :]).abs().max().item()


def _build(**kw):
    return TransformerModel(vocab_size=200, embedding_dim=64, num_heads=4, num_layers=2,
                            hidden_dim=128, max_seq_length=32, **kw)


@pytest.mark.parametrize("name,kw", [
    ("standard_attn", {}),
    ("mla_alone", {"use_mla_kv": True, "kv_latent_dim": 16}),
    ("attn_linear_hybrid", {"mixer": "attn_linear", "layer_plan": "attn,hybrid"}),
    ("mla_attn_linear_hybrid",
     {"mixer": "attn_linear", "layer_plan": "attn,hybrid",
      "use_mla_kv": True, "kv_latent_dim": 16}),
    ("gated_delta", {"mixer": "gated_delta"}),
    ("linear", {"mixer": "linear"}),
    ("dim_wise_rope", {"dim_wise_rope": True}),
])
def test_cache_parity(name, kw):
    """增量解码与全量前向末位 logits 一致（max_diff < 1e-4）。"""
    m = _build(**kw)
    diff = _check_parity(m)
    assert diff < 1e-4, f"{name}: max_diff={diff:.2e} 超过 1e-4"


@pytest.mark.xfail(reason="AxialLinearAttention 训练用 2D 轴向注意力，推理退化为 1D，"
                          "属已知架构取舍（非 bug）；2D 状态无法增量递推故推理路径统一为 1D",
                   strict=True)
def test_axial_linear2d_parity_known_divergence():
    """2D→1D 退化产生 ~5e-2 差异，记录为已知 trade-off。

    彻底修复需 2D 状态的增量递推（per-row/per-col 独立状态维护），
    当前为性能与一致性的权衡。
    """
    m = _build(mixer="linear2d")
    diff = _check_parity(m)
    assert diff < 1e-4, f"max_diff={diff:.2e}"
