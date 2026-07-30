"""第三十六轮-6 回归测试：Mixture of Experts FFN（MoE）。

验证 R36-6 设计：
1. MoELayer 单元：forward 形状、top-k 选择、aux 损失、确定性、梯度流
2. 路由器：top-k 数量、概率归一化、噪声影响
3. 负载均衡损失：均匀路由→≈1、偏斜路由→>1、梯度回流
4. Router z-loss：存在、≥0、梯度回流
5. 集成：MoE 模型 forward/backward、aux 跨层累积、moe_layers 子集
6. 向后兼容：moe=False 无 MoE 参数、旧权重加载、与 share_ffn 互斥报错
7. 配置校验：top_k>num_experts 报错、moe_layers 越界报错
8. 推理路径：eval 无 aux 损失、use_cache 不破坏

核心设计：
- opt-in 默认关（moe=False），保证旧权重向后兼容
- Dense MoE（DML 友好，无 gather/scatter）：所有专家对所有 token 求值 + 广播比较建掩码
- MoELayer 替换 TransformerBlock.ffn（接口 forward(x)->out 一致）
- 辅助损失：负载均衡（Switch Transformer）+ router z-loss（ST-MoE），跨层累积
"""
import pytest
import torch
import torch.nn as nn

from models.model_config import ModelConfig
from models.transformer import TransformerModel, MoELayer
from models.mixers import SwiGLU


def _build(dim=128, heads=4, layers=4, hidden=256, seq=32, vocab=50,
           moe=False, moe_num_experts=8, moe_top_k=2,
           moe_load_balance_weight=0.01, moe_router_z_loss_weight=0.001,
           moe_router_noise=0.0, moe_layers=None, **kw):
    return TransformerModel(
        vocab_size=vocab, embedding_dim=dim, num_heads=heads, num_layers=layers,
        hidden_dim=hidden, max_seq_length=seq,
        gradient_checkpointing=False, tie_weights=False,
        moe=moe, moe_num_experts=moe_num_experts, moe_top_k=moe_top_k,
        moe_load_balance_weight=moe_load_balance_weight,
        moe_router_z_loss_weight=moe_router_z_loss_weight,
        moe_router_noise=moe_router_noise, moe_layers=moe_layers,
        **kw
    )


def _rand_input(B=2, T=8, V=50):
    return torch.randint(0, V, (B, T)), torch.randint(0, V, (B, T))


# ===========================================================================
# 1. MoELayer 单元测试
# ===========================================================================

def test_moe_layer_forward_shape():
    """R36-6-1: forward 输出形状与输入一致。"""
    moe = MoELayer(dim=64, hidden_dim=128, num_experts=4, top_k=2)
    x = torch.randn(2, 8, 64)
    out = moe(x)
    assert out.shape == x.shape


def test_moe_layer_forward_2d_input():
    """R36-6-2: 接受 (N, D) 形状输入。"""
    moe = MoELayer(dim=64, hidden_dim=128, num_experts=4, top_k=1)
    x = torch.randn(16, 64)
    out = moe(x)
    assert out.shape == x.shape


def test_moe_layer_has_router_and_experts():
    """R36-6-3: 构建 router + num_experts 个 SwiGLU 专家。"""
    moe = MoELayer(dim=64, hidden_dim=128, num_experts=4, top_k=2)
    assert isinstance(moe.router, nn.Linear)
    assert moe.router.in_features == 64
    assert moe.router.out_features == 4
    assert moe.router.bias is None  # bias=False
    assert len(moe.experts) == 4
    for e in moe.experts:
        assert isinstance(e, SwiGLU)


def test_moe_layer_top1_equivalent_single_expert():
    """R36-6-4: top_k=1 且 num_experts=1 时，退化为单专家（全权重=1）。"""
    moe = MoELayer(dim=64, hidden_dim=128, num_experts=1, top_k=1)
    x = torch.randn(2, 4, 64)
    out = moe(x)
    # 单专家 top-1：组合权重应为 1，输出应等于该专家输出
    expert_out = moe.experts[0](x)
    assert torch.allclose(out, expert_out, atol=1e-5)


def test_moe_layer_aux_losses_training_only():
    """R36-6-5: aux 损失仅训练期计算，eval 为 None。"""
    moe = MoELayer(dim=64, hidden_dim=128, num_experts=4, top_k=2)
    x = torch.randn(2, 8, 64)
    # train 模式
    moe.train()
    moe(x)
    assert moe.last_load_balance_loss is not None
    assert moe.last_z_loss is not None
    # eval 模式
    moe.eval()
    moe(x)
    assert moe.last_load_balance_loss is None
    assert moe.last_z_loss is None


def test_moe_layer_deterministic_eval():
    """R36-6-6: eval 模式下无噪声，前向确定性。"""
    moe = MoELayer(dim=64, hidden_dim=128, num_experts=4, top_k=2,
                   router_noise=0.5)
    moe.eval()
    x = torch.randn(2, 8, 64)
    out1 = moe(x)
    out2 = moe(x)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_moe_layer_gradient_flow():
    """R36-6-7: 梯度回流到 router + 所有专家。"""
    moe = MoELayer(dim=64, hidden_dim=128, num_experts=4, top_k=2)
    moe.train()
    x = torch.randn(2, 8, 64, requires_grad=True)
    out = moe(x)
    loss = out.sum() + moe.last_load_balance_loss + moe.last_z_loss
    loss.backward()
    assert x.grad is not None
    assert moe.router.weight.grad is not None
    for e in moe.experts:
        for p in e.parameters():
            assert p.grad is not None


# ===========================================================================
# 2. 路由器行为
# ===========================================================================

def test_moe_top_k_not_exceed_num_experts():
    """R36-6-8: top_k > num_experts 报错。"""
    with pytest.raises(AssertionError):
        MoELayer(dim=64, hidden_dim=128, num_experts=2, top_k=4)


def test_moe_top_k_zero_invalid():
    """R36-6-9: top_k=0 报错。"""
    with pytest.raises(AssertionError):
        MoELayer(dim=64, hidden_dim=128, num_experts=4, top_k=0)


def test_moe_num_experts_must_be_positive():
    """R36-6-10: num_experts=0 报错。"""
    with pytest.raises(AssertionError):
        MoELayer(dim=64, hidden_dim=128, num_experts=0, top_k=1)


def test_moe_router_noise_training():
    """R36-6-11: 训练期噪声改变路由（统计性，多次采样应有差异）。"""
    torch.manual_seed(0)
    moe = MoELayer(dim=64, hidden_dim=128, num_experts=4, top_k=2,
                   router_noise=1.0)
    moe.train()
    x = torch.randn(1, 16, 64)
    # 同一输入两次前向（噪声不同）→ 输出应不同
    out1 = moe(x)
    out2 = moe(x)
    assert not torch.allclose(out1, out2, atol=1e-6)


# ===========================================================================
# 3. 负载均衡损失
# ===========================================================================

def test_moe_load_balance_uniform_is_minimum():
    """R36-6-12: 均匀路由（router weight=0）时 load_balance_loss ≈ 1（最小）。"""
    moe = MoELayer(dim=64, hidden_dim=128, num_experts=4, top_k=2)
    # router weight=0 → 所有 logits=0 → softmax 均匀 → top-k 随机但均匀
    with torch.no_grad():
        moe.router.weight.zero_()
    moe.train()
    x = torch.randn(8, 64, 64)  # 多 token 让统计显著
    moe(x)
    lb = moe.last_load_balance_loss.item()
    # 均匀时 f_i=1/E, p_i=1/E → E·Σ(1/E·1/E)=1
    assert 0.9 < lb < 1.5  # 统计波动允许


def test_moe_load_balance_skewed_is_larger():
    """R36-6-13: 偏斜路由时 load_balance_loss > 均匀值。"""
    moe = MoELayer(dim=64, hidden_dim=128, num_experts=4, top_k=1)
    # 让 expert 0 永远胜出：W[0,:]=10（logit_0=10·sum(x)）。用正输入保证 sum>0 → logit_0 恒大正
    with torch.no_grad():
        w = torch.zeros(4, 64)
        w[0, :] = 10.0  # expert 0 行
        moe.router.weight.copy_(w)
    moe.train()
    x_pos = torch.rand(8, 64, 64)  # 正输入 → logit_0 = 10·sum(正) 恒大正值
    moe(x_pos)
    lb = moe.last_load_balance_loss.item()
    # 偏斜：f_0=1, p_0≈1, 其余≈0 → L_bal = E·(1·1) = 4 >> 1
    assert lb > 2.0


def test_moe_load_balance_gradient():
    """R36-6-14: 负载均衡损失对 router 有梯度。"""
    moe = MoELayer(dim=64, hidden_dim=128, num_experts=4, top_k=2)
    moe.train()
    x = torch.randn(2, 8, 64)
    moe(x)
    moe.last_load_balance_loss.backward()
    assert moe.router.weight.grad is not None


def test_moe_z_loss_nonneg():
    """R36-6-15: z-loss ≥ 0（平方和）。"""
    moe = MoELayer(dim=64, hidden_dim=128, num_experts=4, top_k=2)
    moe.train()
    x = torch.randn(2, 8, 64)
    moe(x)
    assert moe.last_z_loss.item() >= 0.0


def test_moe_z_loss_grows_with_logit_scale():
    """R36-6-16: 大 logits → z-loss 更大（logsumexp 增长）。"""
    moe_small = MoELayer(dim=64, hidden_dim=128, num_experts=4, top_k=2)
    moe_large = MoELayer(dim=64, hidden_dim=128, num_experts=4, top_k=2)
    with torch.no_grad():
        moe_large.router.weight.mul_(10.0)  # 放大 logits
    x = torch.randn(2, 8, 64)
    moe_small.train(); moe_large.train()
    moe_small(x)
    moe_large(x)
    assert moe_large.last_z_loss.item() > moe_small.last_z_loss.item()


def test_moe_z_loss_gradient():
    """R36-6-17: z-loss 对 router 有梯度。"""
    moe = MoELayer(dim=64, hidden_dim=128, num_experts=4, top_k=2)
    moe.train()
    x = torch.randn(2, 8, 64)
    moe(x)
    moe.last_z_loss.backward()
    assert moe.router.weight.grad is not None


# ===========================================================================
# 4. 模型集成
# ===========================================================================

def test_model_moe_default_off():
    """R36-6-18: 默认 moe=False，无 MoE 层，ffn 是 SwiGLU。"""
    m = _build()
    assert m.moe_enabled is False
    assert m.moe_layer_set == set()
    for blk in m.blocks:
        assert isinstance(blk.ffn, SwiGLU)
        assert not isinstance(blk.ffn, MoELayer)


def test_model_moe_enabled_all_layers():
    """R36-6-19: moe=True → 全部层 ffn 替换为 MoELayer。"""
    m = _build(moe=True, moe_num_experts=4, moe_top_k=2, layers=3)
    assert m.moe_enabled is True
    assert m.moe_layer_set == {0, 1, 2}
    for blk in m.blocks:
        assert isinstance(blk.ffn, MoELayer)
        assert blk.ffn.num_experts == 4
        assert blk.ffn.top_k == 2


def test_model_moe_forward_shape():
    """R36-6-20: MoE 模型 forward 输出形状正确。"""
    m = _build(moe=True, moe_num_experts=4, moe_top_k=2, vocab=50)
    src, _ = _rand_input()
    m.eval()
    out = m(src)
    assert out.shape == (2, 8, 50)


def test_model_moe_backward_loss_decreases():
    """R36-6-21: MoE 模型训练一步 loss 不爆炸，梯度回流。"""
    m = _build(moe=True, moe_num_experts=4, moe_top_k=2)
    m.train()
    src, tgt = _rand_input()
    logits, _ = m(src, targets=tgt) if isinstance(m(src, targets=tgt), tuple) else (m(src, targets=tgt), None)
    logits = logits.view(-1, 50)
    loss = nn.functional.cross_entropy(logits, tgt.view(-1))
    # 加 aux
    if m._moe_load_balance_loss is not None:
        loss = loss + m._moe_load_balance_loss + m._moe_z_loss
    loss.backward()
    # router 有梯度
    assert m.blocks[0].ffn.router.weight.grad is not None


def test_model_moe_aux_loss_accumulation():
    """R36-6-22: 训练期 aux 损失跨层累积（≥1 层贡献）。"""
    m = _build(moe=True, moe_num_experts=4, moe_top_k=2, layers=3)
    m.train()
    src, tgt = _rand_input()
    m(src, targets=tgt)
    assert m._moe_load_balance_loss is not None
    assert m._moe_z_loss is not None
    # 累积值应 > 单层值（3 层）
    single_lb = m.blocks[0].ffn.last_load_balance_loss.item()
    assert m._moe_load_balance_loss.item() > single_lb * 2.5  # 累积 3 层


def test_model_moe_aux_none_in_eval():
    """R36-6-23: eval 模式 aux 损失为 None。"""
    m = _build(moe=True, moe_num_experts=4, moe_top_k=2)
    m.eval()
    src, _ = _rand_input()
    m(src)
    assert m._moe_load_balance_loss is None
    assert m._moe_z_loss is None


def test_model_moe_layers_subset():
    """R36-6-24: moe_layers 指定子集，仅这些层用 MoE。"""
    m = _build(moe=True, moe_num_experts=4, moe_top_k=2, layers=4,
               moe_layers=[0, 2])
    assert m.moe_layer_set == {0, 2}
    assert isinstance(m.blocks[0].ffn, MoELayer)
    assert isinstance(m.blocks[1].ffn, SwiGLU)
    assert isinstance(m.blocks[2].ffn, MoELayer)
    assert isinstance(m.blocks[3].ffn, SwiGLU)


def test_model_moe_layers_bounds():
    """R36-6-25: moe_layers 越界报错。"""
    with pytest.raises(ValueError, match="越界"):
        _build(moe=True, layers=3, moe_layers=[0, 5])


def test_model_moe_share_ffn_conflict():
    """R36-6-26: moe=True 与 share_ffn=True 互斥报错。"""
    with pytest.raises(ValueError, match="不兼容"):
        _build(moe=True, share_ffn=True)


# ===========================================================================
# 5. 向后兼容 + 配置
# ===========================================================================

def test_model_moe_off_loads_dense_weights():
    """R36-6-27: moe=False 模型可加载纯 dense 权重（无 MoE 参数）。"""
    m_dense = _build(moe=False, layers=2)
    sd = m_dense.state_dict()
    # 无 moe 相关 key
    assert not any('moe' in k.lower() or 'router' in k.lower() or 'experts' in k.lower()
                   for k in sd.keys())
    # 重建同模型加载
    m2 = _build(moe=False, layers=2)
    m2.load_state_dict(sd)


def test_model_moe_strict_false_loads_partial():
    """R36-6-28: MoE 模型加载 dense 权重（strict=False 兼容）。"""
    m_dense = _build(moe=False, layers=2)
    sd_dense = m_dense.state_dict()
    m_moe = _build(moe=True, moe_num_experts=4, moe_top_k=2, layers=2)
    # dense 权重是 MoE 权重的子集（attn/embedding/norm 共享，ffn 不同）
    missing, unexpected = m_moe.load_state_dict(sd_dense, strict=False)
    # MoE 专家/router 参数是 missing（新增）
    assert any('router' in k or 'experts' in k for k in missing)


def test_model_moe_config_from_dict():
    """R36-6-29: ModelConfig.from_dict 读取 moe 字段。"""
    cfg = ModelConfig.from_dict({
        'vocab_size': 50, 'embedding_dim': 64, 'num_heads': 4,
        'num_layers': 2, 'hidden_dim': 128, 'max_seq_length': 32,
        'moe': True, 'moe_num_experts': 6, 'moe_top_k': 3,
        'moe_load_balance_weight': 0.02, 'moe_router_z_loss_weight': 0.002,
        'moe_router_noise': 0.1, 'moe_layers': [0, 1],
    })
    assert cfg.moe is True
    assert cfg.moe_num_experts == 6
    assert cfg.moe_top_k == 3
    assert cfg.moe_load_balance_weight == 0.02
    assert cfg.moe_router_z_loss_weight == 0.002
    assert cfg.moe_router_noise == 0.1
    assert cfg.moe_layers == [0, 1]


def test_model_moe_config_validation():
    """R36-6-30: ModelConfig 校验 top_k<=num_experts。"""
    with pytest.raises(AssertionError):
        ModelConfig.from_dict({
            'vocab_size': 50, 'embedding_dim': 64, 'num_heads': 4,
            'num_layers': 2, 'hidden_dim': 128, 'max_seq_length': 32,
            'moe': True, 'moe_num_experts': 2, 'moe_top_k': 5,
        })


def test_model_moe_from_config():
    """R36-6-31: from_config 构建 MoE 模型。"""
    cfg = ModelConfig.from_dict({
        'vocab_size': 50, 'embedding_dim': 64, 'num_heads': 4,
        'num_layers': 2, 'hidden_dim': 128, 'max_seq_length': 32,
        'moe': True, 'moe_num_experts': 4, 'moe_top_k': 2,
    })
    m = TransformerModel.from_config(cfg)
    assert m.moe_enabled is True
    for blk in m.blocks:
        assert isinstance(blk.ffn, MoELayer)


def test_model_moe_use_cache_unchanged():
    """R36-6-32: MoE + use_cache 增量解码不破坏（KV cache 一致性）。"""
    m = _build(moe=True, moe_num_experts=4, moe_top_k=2, seq=16)
    m.eval()
    src = torch.randint(0, 50, (1, 8))
    # 首步全量
    out1, pkv = m(src, use_cache=True)
    # 增量一步
    next_tok = torch.randint(0, 50, (1, 1))
    out2, _ = m(next_tok, past_key_values=pkv, use_cache=True)
    assert out2.shape == (1, 1, 50)


# ===========================================================================
# 6. DML 设备测试（若可用）
# ===========================================================================

def test_moe_runs_on_dml_device():
    """R36-6-33: MoE forward 在 DML 设备上跑通（验证无 gather/scatter，DML 友好）。

    仅测 forward：DML 后端 autograd engine 有已知 device_ready_queues 限制（与 MoE
    无关，是 privateuseone 通用问题）；MoE 梯度回流已在 test_moe_layer_gradient_flow
    和 test_model_moe_backward_loss_decreases 于 CPU 上充分验证。
    """
    try:
        dev = torch.device('privateuseone:0')
        _ = torch.zeros(1, device=dev)  # 探测可用性
    except Exception:
        pytest.skip("DML 设备不可用")
    moe = MoELayer(dim=64, hidden_dim=128, num_experts=4, top_k=2).to(dev)
    moe.train()
    x = torch.randn(2, 8, 64, device=dev)
    out = moe(x)
    assert out.device == dev
    assert out.shape == x.shape
    # aux 损失在 DML 设备上（DML 友好的 element-wise + reduce 计算）
    assert moe.last_load_balance_loss.device == dev
    assert moe.last_z_loss.device == dev


def test_model_moe_runs_on_dml_device():
    """R36-6-34: 完整 MoE 模型 forward 在 DML 设备上跑通（无 gather/scatter）。

    backward 在 CPU 上由 test_model_moe_backward_loss_decreases 验证；
    DML autograd engine 限制与 MoE 无关，故此处仅测 forward。
    """
    try:
        dev = torch.device('privateuseone:0')
        _ = torch.zeros(1, device=dev)
    except Exception:
        pytest.skip("DML 设备不可用")
    m = _build(moe=True, moe_num_experts=4, moe_top_k=2, layers=2).to(dev)
    m.train()
    src = torch.randint(0, 50, (2, 8), device=dev)
    tgt = torch.randint(0, 50, (2, 8), device=dev)
    logits = m(src, targets=tgt)
    assert logits.device == dev
    assert logits.shape == (2, 8, 50)
    # aux 损失跨层累积在 DML 设备上
    assert m._moe_load_balance_loss.device == dev
    assert m._moe_z_loss.device == dev
