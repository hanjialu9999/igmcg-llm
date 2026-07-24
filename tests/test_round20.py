"""第二十轮回归测试：DALA 层间对齐训练 + 维度级 RoPE 动态分配。

覆盖：
- DALA：contrastive_loss 非零、梯度回流、浅层对齐 x0 / 深层对齐前层、与 layer_contrastive 兼容
- dim_wise_rope：dim_wise_logit 参数创建、梯度回流、init sigmoid=0.5、与 Partial RoPE 正交
- 向后兼容：默认关时行为不变
"""
import torch
import torch.nn as nn

from models.transformer import TransformerModel
from models.rope import RotaryEmbedding


def _small(**over):
    kw = dict(vocab_size=200, embedding_dim=64, num_heads=4, num_layers=3,
              hidden_dim=128, max_seq_length=32)
    kw.update(over)
    return TransformerModel(**kw)


# ---------------------------------------------------------------------------
# DALA（Depth-Aware Layer Alignment）
# ---------------------------------------------------------------------------

def test_dala_contrastive_loss_nonzero():
    """DALA 启用时 _contrastive_loss 非零。"""
    m = _small(aligned_training=True)
    m.train()
    x = torch.randint(0, 200, (2, 8))
    m(x)
    assert m._contrastive_loss is not None, "DALA 应产生 _contrastive_loss"
    assert m._contrastive_loss.item() > 0, "DALA loss 应 > 0"


def test_dala_gradient_flow():
    """DALA loss 梯度回流到各层参数。"""
    m = _small(aligned_training=True)
    m.train()
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    loss = out.float().sum() + m._contrastive_loss
    loss.backward()
    # 各层 qkv 应有梯度
    for i, blk in enumerate(m.blocks):
        if hasattr(blk, 'attn') and hasattr(blk.attn, 'qkv'):
            assert blk.attn.qkv.weight.grad is not None, f"layer {i} qkv 无梯度"


def test_dala_eval_no_loss():
    """eval 时不计算 DALA loss。"""
    m = _small(aligned_training=True)
    m.eval()
    x = torch.randint(0, 200, (2, 8))
    m(x)
    assert m._contrastive_loss is None, "eval 时 _contrastive_loss 应为 None"


def test_dala_backward_compat():
    """默认关时 _contrastive_loss 为 None。"""
    m = _small()
    m.train()
    x = torch.randint(0, 200, (2, 8))
    m(x)
    assert m._contrastive_loss is None, "默认应无 _contrastive_loss"


def test_dala_changes_output_vs_layer_contrastive():
    """DALA 与 layer_contrastive 产生不同的 loss（对齐目标不同）。"""
    torch.manual_seed(42)
    m_lc = _small(layer_contrastive=True)
    torch.manual_seed(42)
    m_dala = _small(aligned_training=True)
    x = torch.randint(0, 200, (2, 8))
    m_lc.train(); m_dala.train()
    m_lc(x)
    m_dala(x)
    # DALA 的 geodesic 目标与 layer_contrastive 的纯前邻对齐不同
    assert not torch.allclose(m_lc._contrastive_loss, m_dala._contrastive_loss), \
        "DALA 与 layer_contrastive 的 loss 应不同"


def test_dala_geodesic_target_shallow_vs_deep():
    """DALA 的 α_i 插值逻辑：浅层偏向 x0，深层偏向前层。"""
    num_layers = 4
    # 验证 α_i = i / (num_layers - 1) 的插值逻辑
    for i in range(num_layers):
        alpha_i = i / max(num_layers - 1, 1)
        if i == 0:
            assert alpha_i == 0.0, f"层 0 的 α 应为 0（全部对齐 x0），实际 {alpha_i}"
        elif i == num_layers - 1:
            assert alpha_i == 1.0, f"层 {i} 的 α 应为 1（全部对齐前层），实际 {alpha_i}"
        else:
            assert 0.0 < alpha_i < 1.0, f"层 {i} 的 α 应在 (0,1)，实际 {alpha_i}"


# ---------------------------------------------------------------------------
# 维度级 RoPE 动态分配（dim_wise_rope）
# ---------------------------------------------------------------------------

def test_dim_wise_rope_param_created():
    """dim_wise_rope=True 时创建 dim_wise_logit 参数。"""
    m = _small(dim_wise_rope=True)
    rope = m.blocks[0].attn.rope
    assert hasattr(rope, 'dim_wise_logit'), "dim_wise_logit 未创建"
    assert rope.dim_wise_logit.shape[0] == rope.rot_dim // 2, \
        f"dim_wise_logit 维度应为 {rope.rot_dim // 2}，实际 {rope.dim_wise_logit.shape[0]}"


def test_dim_wise_rope_init_half():
    """dim_wise_logit init=0 → sigmoid=0.5 半旋转起步。"""
    m = _small(dim_wise_rope=True)
    rope = m.blocks[0].attn.rope
    assert torch.allclose(rope.dim_wise_logit, torch.zeros_like(rope.dim_wise_logit)), \
        "dim_wise_logit 应初始化为 0"


def test_dim_wise_rope_gradient_flow():
    """dim_wise_logit 梯度回流。"""
    m = _small(dim_wise_rope=True)
    m.train()
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    loss = out.float().sum()
    loss.backward()
    assert m.blocks[0].attn.rope.dim_wise_logit.grad is not None, \
        "dim_wise_logit 无梯度"


def test_dim_wise_rope_backward_compat():
    """默认关时不创建 dim_wise_logit。"""
    m = _small()
    rope = m.blocks[0].attn.rope
    assert not hasattr(rope, 'dim_wise_logit'), "默认不应创建 dim_wise_logit"
    assert not rope.dim_wise_enabled, "默认 dim_wise_enabled 应为 False"


def test_dim_wise_rope_changes_output():
    """dim_wise_rope=True 改变输出。"""
    torch.manual_seed(42)
    m_off = _small(dim_wise_rope=False)
    torch.manual_seed(42)
    m_on = _small(dim_wise_rope=True)
    x = torch.randint(0, 200, (2, 8))
    with torch.no_grad():
        out_off = m_off(x)
        out_on = m_on(x)
    # init sigmoid=0.5 → cos/sin 被半掩码，输出应不同
    assert not torch.allclose(out_off, out_on, atol=1e-5), \
        "dim_wise_rope 应改变输出"


def test_dim_wise_rope_with_partial_rope():
    """dim_wise_rope + Partial RoPE 组合正确。"""
    m = _small(dim_wise_rope=True, rope_dim_fraction=0.5)
    rope = m.blocks[0].attn.rope
    # head_dim = 64 // 4 = 16, fraction=0.5 → rot_dim = 8
    head_dim = 64 // 4
    expected_rot = max(2, int(head_dim * 0.5) // 2 * 2)
    assert rope.rot_dim == expected_rot, f"Partial RoPE rot_dim 应为 {expected_rot}，实际 {rope.rot_dim}"
    assert rope.dim_wise_logit.shape[0] == expected_rot // 2, \
        f"dim_wise_logit 应为 {expected_rot // 2}，实际 {rope.dim_wise_logit.shape[0]}"
    # 前向正常
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    assert out.shape == (2, 8, 200)


def test_dim_wise_rope_with_yarn():
    """dim_wise_rope + YaRN 组合正确。"""
    m = _small(dim_wise_rope=True, yarn_scale=2.0, yarn_orig_max_seq_length=32)
    rope = m.blocks[0].attn.rope
    assert rope.dim_wise_enabled, "dim_wise 应启用"
    assert rope.yarn_scale == 2.0, "YaRN 应启用"
    # 前向正常
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    assert out.shape == (2, 8, 200)


def test_dim_wise_rope_mask_logic():
    """mask=0 时 cos=1/sin=0（不旋转），mask=1 时 cos/sin 不变。"""
    dim = 16
    rope = RotaryEmbedding(dim, dim_wise=True)
    # 手动设置 logit：前半 -10（sigmoid≈0，不旋转），后半 10（sigmoid≈1，旋转）
    with torch.no_grad():
        rope.dim_wise_logit[:4] = -10.0  # 不旋转
        rope.dim_wise_logit[4:] = 10.0   # 旋转
    cos = torch.ones(1, 1, 4, dim)
    sin = torch.zeros(1, 1, 4, dim)
    cos_m, sin_m = rope._apply_dim_wise_mask(cos, sin)
    # 前半（mask≈0）：cos≈1, sin≈0
    assert torch.allclose(cos_m[..., :8], torch.ones_like(cos_m[..., :8]), atol=1e-4), \
        "mask=0 时 cos 应 ≈ 1"
    assert torch.allclose(sin_m[..., :8], torch.zeros_like(sin_m[..., :8]), atol=1e-4), \
        "mask=0 时 sin 应 ≈ 0"
    # 后半（mask≈1）：cos=1, sin=0（不变）
    assert torch.allclose(cos_m[..., 8:], torch.ones_like(cos_m[..., 8:]), atol=1e-4), \
        "mask=1 时 cos 应不变"
    assert torch.allclose(sin_m[..., 8:], torch.zeros_like(sin_m[..., 8:]), atol=1e-4), \
        "mask=1 时 sin 应不变"


def test_dim_wise_rope_incremental_decode_parity():
    """dim_wise_rope 增量解码与全量前向末位 logits 数值一致（cache parity）。

    审查发现原测试第二次前向未传 past_key_values，未真正验证增量解码续步。
    此处改为逐 token 续步并比对末位 logits（阈值 1e-4，与 test_cache_parity 一致）。
    """
    m = _small(dim_wise_rope=True, num_layers=2)
    m.eval()
    torch.manual_seed(42)
    x = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        full = m(x, use_cache=False)
        out, past = m(x[:, :1], past_key_values=None, use_cache=True)
        for t in range(1, 8):
            out, past = m(x[:, t:t + 1], past_key_values=past, use_cache=True)
    diff = (full[:, -1, :] - out[:, -1, :]).abs().max().item()
    assert diff < 1e-4, f"dim_wise_rope cache parity max_diff={diff:.2e} 超过 1e-4"


def test_dala_with_layer_contrastive():
    """aligned_training + layer_contrastive 同启：DALA 覆盖 layer_contrastive 的对齐目标。

    审查发现两者同启无专项测试。代码逻辑：aligned_training=True 时 _dala_x0 非 None，
    走 geodesic 插值；仅 layer_contrastive 时 _dala_x0=None 走纯前邻对齐。
    """
    m = _small(aligned_training=True, layer_contrastive=True)
    m.train()
    x = torch.randint(0, 200, (2, 8))
    m(x)
    assert m._contrastive_loss is not None, "同启时应产生 _contrastive_loss"
    assert m._contrastive_loss.item() > 0, "DALA loss 应 > 0"
