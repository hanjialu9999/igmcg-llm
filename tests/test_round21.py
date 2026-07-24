"""第二十一轮回归测试：NoPE 长度外推增强 + RWKV-7 广义 Delta Rule。

覆盖：
- head_temp：log_temp 从 (1,) 升级为 (num_heads,)、梯度回流、向后兼容
- value_relative_coding：value_rel_lambda 参数创建、梯度回流、cache parity
- NoPE 增强 + nope_layers 同启
- rwkv7：z_proj/b_proj 创建、专用初始化、cache parity、向后兼容、梯度回流
- rwkv7 + channel_wise 同启
"""
import torch
import torch.nn as nn

from models.transformer import TransformerModel


def _small(**over):
    kw = dict(vocab_size=200, embedding_dim=64, num_heads=4, num_layers=3,
              hidden_dim=128, max_seq_length=32)
    kw.update(over)
    return TransformerModel(**kw)


def _small_gated_delta(**over):
    """GatedDeltaNet mixer 专用构建（需 mixer='gated_delta'）。"""
    over.setdefault('mixer', 'gated_delta')
    return _small(**over)


# ---------------------------------------------------------------------------
# NoPE 增强：per-head 注意力温度（head_temp）
# ---------------------------------------------------------------------------

def test_head_temp_param_shape():
    """head_temp=True 时 log_temp 维度为 (num_heads,)。"""
    m = _small(head_temp=True, alibi=True)
    assert m.blocks[0].attn.log_temp.shape[0] == 4, \
        f"log_temp 应为 (num_heads=4,)，实际 {m.blocks[0].attn.log_temp.shape}"


def test_head_temp_backward_compat():
    """默认关时 log_temp 维度为 (1,)（全局标量，向后兼容）。"""
    m = _small(alibi=True)
    assert m.blocks[0].attn.log_temp.shape[0] == 1, \
        f"默认 log_temp 应为 (1,)，实际 {m.blocks[0].attn.log_temp.shape}"


def test_head_temp_init_zero():
    """head_temp=True 时 log_temp init=0（温度=1，向后兼容）。"""
    m = _small(head_temp=True, alibi=True)
    assert torch.allclose(m.blocks[0].attn.log_temp, torch.zeros(4)), \
        "log_temp 应初始化为 0"


def test_head_temp_gradient_flow():
    """head_temp log_temp 梯度回流。"""
    m = _small(head_temp=True, alibi=True)
    m.train()
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    loss = out.float().sum()
    loss.backward()
    assert m.blocks[0].attn.log_temp.grad is not None, "log_temp 无梯度"


# ---------------------------------------------------------------------------
# NoPE 增强：value-side 相对编码（value_relative_coding）
# ---------------------------------------------------------------------------

def test_value_relative_coding_param_created():
    """value_relative_coding=True 时 value_rel_lambda 参数创建。"""
    m = _small(value_relative_coding=True, alibi=True)
    assert hasattr(m.blocks[0].attn, 'value_rel_lambda'), "value_rel_lambda 未创建"
    assert m.blocks[0].attn.value_rel_lambda.shape == (1,), \
        f"value_rel_lambda 应为 (1,)，实际 {m.blocks[0].attn.value_rel_lambda.shape}"


def test_value_relative_coding_init_zero():
    """value_rel_lambda init=0（tanh(0)=0，v 不变，向后兼容）。"""
    m = _small(value_relative_coding=True, alibi=True)
    assert m.blocks[0].attn.value_rel_lambda.item() == 0.0, \
        "value_rel_lambda 应初始化为 0"


def test_value_relative_coding_backward_compat():
    """默认关时不创建 value_rel_lambda。"""
    m = _small(alibi=True)
    assert not hasattr(m.blocks[0].attn, 'value_rel_lambda'), \
        "默认不应创建 value_rel_lambda"


def test_value_relative_coding_gradient_flow():
    """value_rel_lambda 梯度回流。"""
    m = _small(value_relative_coding=True, alibi=True)
    m.train()
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    loss = out.float().sum()
    loss.backward()
    assert m.blocks[0].attn.value_rel_lambda.grad is not None, \
        "value_rel_lambda 无梯度"


# ---------------------------------------------------------------------------
# NoPE 增强 + nope_layers 组合
# ---------------------------------------------------------------------------

def test_nope_enhance_with_nope_layers():
    """head_temp + value_relative_coding + nope_layers 同启。"""
    m = _small(head_temp=True, value_relative_coding=True,
               alibi=True, nope_layers=[1])
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    assert out.shape == (2, 8, 200)
    # nope 层（layer 1）也应有 head_temp 和 value_relative_coding
    assert m.blocks[1].attn.log_temp.shape[0] == 4, "nope 层也应有 per-head log_temp"
    assert hasattr(m.blocks[1].attn, 'value_rel_lambda'), "nope 层也应有 value_rel_lambda"


def test_nope_enhance_cache_parity():
    """head_temp + value_relative_coding cache parity。"""
    m = _small(head_temp=True, value_relative_coding=True,
               alibi=True, nope_layers=[1], num_layers=2)
    m.eval()
    torch.manual_seed(42)
    x = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        full = m(x, use_cache=False)
        out, past = m(x[:, :1], past_key_values=None, use_cache=True)
        for t in range(1, 8):
            out, past = m(x[:, t:t + 1], past_key_values=past, use_cache=True)
    diff = (full[:, -1, :] - out[:, -1, :]).abs().max().item()
    assert diff < 1e-4, f"NoPE enhance cache parity max_diff={diff:.2e} 超过 1e-4"


# ---------------------------------------------------------------------------
# RWKV-7 广义 Delta Rule
# ---------------------------------------------------------------------------

def test_rwkv7_params_created():
    """rwkv7=True 时 z_proj/b_proj 创建。"""
    m = _small_gated_delta(rwkv7=True)
    attn = m.blocks[0].attn
    assert hasattr(attn, 'z_proj'), "z_proj 未创建"
    assert hasattr(attn, 'b_proj'), "b_proj 未创建"
    assert attn.z_proj.weight.shape == (4, 64), \
        f"z_proj.weight 应为 (num_heads=4, dim=64)，实际 {attn.z_proj.weight.shape}"
    assert attn.z_proj.bias.shape == (4,), \
        f"z_proj.bias 应为 (4,)，实际 {attn.z_proj.bias.shape}"
    assert attn.b_proj.weight.shape == (4 * 16, 64), \
        f"b_proj.weight 应为 (num_heads*head_dim=64, dim=64)，实际 {attn.b_proj.weight.shape}"


def test_rwkv7_specialized_init():
    """rwkv7 专用初始化：z_proj weight=0/bias=-3。b_proj 用通用 N(0,0.02)（不归零，防梯度死锁）。"""
    m = _small_gated_delta(rwkv7=True)
    attn = m.blocks[0].attn
    assert torch.allclose(attn.z_proj.weight, torch.zeros_like(attn.z_proj.weight)), \
        "z_proj.weight 应为 0（_apply_specialized_inits 重置）"
    assert torch.allclose(attn.z_proj.bias, torch.full_like(attn.z_proj.bias, -3.0)), \
        f"z_proj.bias 应为 -3，实际 {attn.z_proj.bias.tolist()}"
    # b_proj.weight 不为 0（通用 N(0,0.02) 初始化，防梯度死锁）
    assert not torch.allclose(attn.b_proj.weight, torch.zeros_like(attn.b_proj.weight)), \
        "b_proj.weight 不应为 0（防梯度死锁）"


def test_rwkv7_backward_compat():
    """默认关时不创建 z_proj/b_proj。"""
    m = _small_gated_delta()
    attn = m.blocks[0].attn
    assert not hasattr(attn, 'z_proj'), "默认不应创建 z_proj"
    assert not hasattr(attn, 'b_proj'), "默认不应创建 b_proj"
    assert not attn.rwkv7_enabled, "默认 rwkv7_enabled 应为 False"


def test_rwkv7_gradient_flow():
    """rwkv7 z_proj/b_proj 梯度回流。"""
    m = _small_gated_delta(rwkv7=True)
    m.train()
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    loss = out.float().sum()
    loss.backward()
    attn = m.blocks[0].attn
    assert attn.z_proj.weight.grad is not None, "z_proj.weight 无梯度"
    assert attn.z_proj.bias.grad is not None, "z_proj.bias 无梯度"
    assert attn.b_proj.weight.grad is not None, "b_proj.weight 无梯度"


def test_rwkv7_cache_parity():
    """rwkv7 增量解码与全量前向末位 logits 数值一致。"""
    m = _small_gated_delta(rwkv7=True, num_layers=2)
    m.eval()
    torch.manual_seed(42)
    x = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        full = m(x, use_cache=False)
        out, past = m(x[:, :1], past_key_values=None, use_cache=True)
        for t in range(1, 8):
            out, past = m(x[:, t:t + 1], past_key_values=past, use_cache=True)
    diff = (full[:, -1, :] - out[:, -1, :]).abs().max().item()
    assert diff < 1e-4, f"rwkv7 cache parity max_diff={diff:.2e} 超过 1e-4"


def test_rwkv7_init_weak_perturbation():
    """rwkv7 init 时 z_gate≈0.05（弱扰动），b_dir 非 0（防梯度死锁）。

    b_proj 不归零（通用 N(0,0.02)），b_dir 非 0 使梯度可传播；
    z_gate≈0.05 使 rank-1 扰动系数 = z_gate·|b|² 很小（弱扰动起步）。
    """
    m = _small_gated_delta(rwkv7=True)
    m.eval()
    attn = m.blocks[0].attn
    x_ids = torch.randint(0, 200, (1, 4))
    with torch.no_grad():
        x_emb = m.embedding(x_ids)
        z_gate, b_dir = attn._compute_rwkv7_gates(x_emb)
        # z_gate 应为 sigmoid(-3)≈0.05（z_proj.weight=0, bias=-3）
        expected_z = torch.sigmoid(torch.tensor(-3.0))
        assert torch.allclose(z_gate, torch.full_like(z_gate, expected_z), atol=1e-4), \
            f"z_gate 应为 sigmoid(-3)≈0.05，实际 {z_gate[0,0,0].item()}"
        # b_dir 不为 0（b_proj 用通用 N(0,0.02)，防梯度死锁）
        assert not torch.allclose(b_dir, torch.zeros_like(b_dir), atol=1e-6), \
            "b_dir 不应为 0（防梯度死锁）"


def test_rwkv7_training_changes_output():
    """rwkv7 训练一步后 b_proj.weight 不再为 0，输出与 init 时不同。

    init b_proj.weight=0 → 扰动=0；训练一步后 b_proj 有梯度更新，扰动非 0。
    用较大学习率（1.0）确保变化超过 atol。
    """
    m = _small_gated_delta(rwkv7=True, gradient_checkpointing=False)
    m.train()
    x = torch.randint(0, 200, (2, 8))
    # init 时的输出
    with torch.no_grad():
        out_init = m(x).clone()
    # 训练一步
    out = m(x)
    loss = out.float().sum()
    loss.backward()
    # 手动更新 b_proj.weight（大学习率 1.0 确保扰动可见）
    with torch.no_grad():
        m.blocks[0].attn.b_proj.weight -= 1.0 * m.blocks[0].attn.b_proj.weight.grad
    # 训练后的输出
    with torch.no_grad():
        out_after = m(x)
    # b_proj.weight 不再为 0 → 扰动非 0 → 输出不同
    assert not torch.allclose(out_init, out_after, atol=1e-5), \
        "训练一步后 rwkv7 扰动应使输出改变"


def test_rwkv7_with_channel_wise():
    """rwkv7 + channel_wise 同启。"""
    m = _small_gated_delta(rwkv7=True, gated_delta_channel_wise=True)
    attn = m.blocks[0].attn
    assert attn.rwkv7_enabled, "rwkv7 应启用"
    assert attn.channel_wise, "channel_wise 应启用"
    # 前向正常
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    assert out.shape == (2, 8, 200)


def test_rwkv7_with_yarn():
    """rwkv7 + YaRN 组合正确。"""
    m = _small_gated_delta(rwkv7=True, yarn_scale=2.0, yarn_orig_max_seq_length=32)
    attn = m.blocks[0].attn
    assert attn.rwkv7_enabled, "rwkv7 应启用"
    # 前向正常
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    assert out.shape == (2, 8, 200)
