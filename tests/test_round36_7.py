"""第三十六轮-7 回归测试：MoE DML backward 修复（topk 仅选位置）。

R36-7 根因：models/moe.py 旧实现 `routing_probs.topk(top_k)` 的 topk_probs 参与
autograd，其 backward 用 DML scatter 回传梯度，而 DML scatter 不支持
"partially modified dimensions"，导致 backward 报错。

修复：topk_indices 在 torch.no_grad() 下获取（仅选位置，detach），组合权重改为
combined = routing_probs * topk_mask（可导，梯度经 routing_probs→softmax→router
回流），归一化 combined/combined.sum()。数学与梯度均与原 topk_probs 路径等价。

本测试验证：
1. 数值等价：新路径 combined 与「gather(topk_probs) 散布」参考实现逐元素相等
2. 梯度等价：d combined/d routing_probs = topk_mask（仅 top-k 位置非零）
3. topk_indices 不在 autograd 图中（detach）
4. DML backward：MoELayer 与完整模型在 DML 设备上 backward 不报错、router.grad 非空
5. DML 训练：mini 训练循环 loss 下降（端到端 backward + 优化器更新）
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# 提前导入 torch_directml 注册 privateuseone 后端：pytest 默认的 assertion
# rewriting 模式会干扰 DML autograd engine 的 device_ready_queues_ 初始化（致
# backward 报 INTERNAL ASSERT），提前注册可规避（与 train.py 一致）。
try:
    import torch_directml  # noqa: F401
except Exception:
    pass

from models.moe import MoELayer
from models.transformer import TransformerModel


def _dml_device():
    """返回可用的 DML 设备，不可用则 pytest.skip。

    需先 import torch_directml 注册 privateuseone 后端（与 models/device.py 一致）。
    """
    try:
        import torch_directml
    except Exception:
        pytest.skip("torch_directml 未安装，DML 不可用")
    if not getattr(torch_directml, "is_available", lambda: False)():
        pytest.skip("DML 设备不可用")
    return torch_directml.device()


def _build(dim=128, heads=4, layers=2, hidden=256, seq=32, vocab=50, **kw):
    kw.setdefault('moe_num_experts', 4)
    kw.setdefault('moe_top_k', 2)
    return TransformerModel(
        vocab_size=vocab, embedding_dim=dim, num_heads=heads, num_layers=layers,
        hidden_dim=hidden, max_seq_length=seq,
        gradient_checkpointing=False, tie_weights=False,
        moe=True, **kw
    )


# ===========================================================================
# 1. 数值等价：新路径 vs gather 参考实现
# ===========================================================================

def test_combined_numerical_matches_gather_reference():
    """R36-7-1: 新 combined = routing_probs * topk_mask 与 gather(topk_probs) 散布等价。

    参考实现（旧路径的等价数学）：
      topk_probs, topk_idx = routing_probs.topk(k)   # topk_probs 涉及 autograd
      ref = zeros(N, E); ref[n, topk_idx[n]] = topk_probs[n]   # scatter 等价
      ref = ref / ref.sum(-1)
    新路径：combined = routing_probs * topk_mask; combined / combined.sum(-1)
    两者数值应逐元素相等。
    """
    torch.manual_seed(0)
    N, E, k = 16, 6, 2
    logits = torch.randn(N, E, requires_grad=True)
    routing_probs = F.softmax(logits, dim=-1)

    # 新路径（MoELayer 内部逻辑）
    with torch.no_grad():
        _, topk_idx = routing_probs.topk(k, dim=-1)
    expert_ids = torch.arange(E)
    topk_mask = (topk_idx.unsqueeze(-1) == expert_ids.view(1, 1, -1)).any(dim=1).to(routing_probs.dtype)
    combined_new = routing_probs * topk_mask
    combined_new = combined_new / combined_new.sum(dim=-1, keepdim=True).clamp(min=1e-9)

    # 参考路径（gather + 散布，数值等价）
    topk_probs = routing_probs.gather(1, topk_idx)  # (N, k)
    ref = torch.zeros(N, E)
    ref.scatter_(1, topk_idx, topk_probs)
    ref = ref / ref.sum(dim=-1, keepdim=True).clamp(min=1e-9)

    assert torch.allclose(combined_new, ref, atol=1e-7), (
        f"新路径与参考实现不一致: max diff {(combined_new - ref).abs().max()}")


# ===========================================================================
# 2. 梯度等价：d combined/d routing_probs = topk_mask（归一化后）
# ===========================================================================

def test_combined_gradient_is_topk_mask_pattern():
    """R36-7-2: combined 对 routing_probs 的梯度仅在 top-k 位置非零（= topk_mask 模式）。

    这验证梯度回流路径正确：原 topk backward 用 scatter 回传到 top-k 位置，
    新路径通过 routing_probs * topk_mask 直接得到相同梯度模式（topk_mask 作乘子）。
    """
    torch.manual_seed(1)
    N, E, k = 8, 5, 2
    logits = torch.randn(N, E, requires_grad=True)
    routing_probs = F.softmax(logits, dim=-1)
    routing_probs.retain_grad()  # 非叶张量，需 retain_grad 才能访问 .grad

    with torch.no_grad():
        _, topk_idx = routing_probs.topk(k, dim=-1)
    expert_ids = torch.arange(E)
    topk_mask = (topk_idx.unsqueeze(-1) == expert_ids.view(1, 1, -1)).any(dim=1).to(routing_probs.dtype)
    combined = routing_probs * topk_mask
    combined = combined / combined.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    # 用非均匀权重加权求和（避免归一化致 sum 恒为 1 梯度为 0）
    weights = torch.randn(N, E)
    (combined * weights).sum().backward()
    grad = routing_probs.grad  # (N, E)

    # 梯度在非 top-k 位置应严格为 0（topk_mask=0 处 combined 不依赖 routing_probs）
    non_topk = topk_mask == 0
    assert torch.all(grad[non_topk] == 0), (
        f"非 top-k 位置梯度应严格为 0，实际 max {grad[non_topk].abs().max()}")
    # top-k 位置梯度非零（归一化后的雅可比，对 routing_probs 有依赖）
    topk_positions = topk_mask == 1
    assert torch.all(grad[topk_positions].abs() > 0), "top-k 位置梯度应非零"


def test_combined_gradient_matches_reference_via_autograd():
    """R36-7-3: 新路径梯度与参考 gather 路径梯度逐元素一致（softmax 入口）。

    两者都以 softmax 输出为入口、归一化相同，故对 router_logits 的梯度应一致。
    """
    torch.manual_seed(2)
    N, E, k = 8, 5, 2

    # 新路径
    logits_new = torch.randn(N, E, requires_grad=True)
    probs_new = F.softmax(logits_new, dim=-1)
    with torch.no_grad():
        _, topk_idx = probs_new.topk(k, dim=-1)
    expert_ids = torch.arange(E)
    topk_mask = (topk_idx.unsqueeze(-1) == expert_ids.view(1, 1, -1)).any(dim=1).to(probs_new.dtype)
    combined_new = probs_new * topk_mask
    combined_new = combined_new / combined_new.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    combined_new.sum().backward()
    grad_new = logits_new.grad.clone()

    # 参考路径（gather，相同 logits 以便对比）
    logits_ref = logits_new.detach().clone().requires_grad_(True)
    probs_ref = F.softmax(logits_ref, dim=-1)
    topk_probs = probs_ref.gather(1, topk_idx)
    ref = torch.zeros(N, E)
    ref.scatter_(1, topk_idx, topk_probs)
    ref = ref / ref.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    ref.sum().backward()
    grad_ref = logits_ref.grad.clone()

    assert torch.allclose(grad_new, grad_ref, atol=1e-6), (
        f"梯度不一致: max diff {(grad_new - grad_ref).abs().max()}")


# ===========================================================================
# 3. topk_indices 不在 autograd 图中（detach）
# ===========================================================================

def test_topk_indices_detached_from_autograd():
    """R36-7-4: topk_indices 在 no_grad 下获取，不参与 autograd 图。

    验证修复要点：topk 本身不回传梯度（绕过 DML scatter 限制）。
    """
    torch.manual_seed(3)
    moe = MoELayer(dim=64, hidden_dim=128, num_experts=4, top_k=2)
    moe.train()
    x = torch.randn(2, 8, 64)
    # 注册 forward pre-hook 在 router 后捕获 topk 调用难以隔离；改用间接验证：
    # 反传后 router 有梯度（说明 routing_probs * topk_mask 路径工作），且 backward 不报错
    out = moe(x)
    loss = out.sum() + moe.last_load_balance_loss + moe.last_z_loss
    loss.backward()  # 若 topk 参与 autograd 会在此触发 DML scatter（CPU 也会构造多余图）
    assert moe.router.weight.grad is not None
    # 数值有限（无 NaN/Inf，说明归一化路径稳定）
    assert torch.isfinite(moe.router.weight.grad).all()


# ===========================================================================
# 4. DML backward：MoELayer + 完整模型
# ===========================================================================

def test_moe_layer_dml_backward_router_grad():
    """R36-7-5: MoELayer 在 DML 上 backward 成功，router.grad 非空（修复核心目标）。

    修复前：topk_probs backward 触发 DML scatter 报错。
    修复后：topk_indices detach，梯度经 routing_probs * topk_mask 回流，DML 兼容。
    """
    dev = _dml_device()
    torch.manual_seed(4)
    moe = MoELayer(dim=64, hidden_dim=128, num_experts=4, top_k=2).to(dev)
    moe.train()
    x = torch.randn(2, 8, 64, device=dev, requires_grad=True)
    out = moe(x)
    loss = out.sum() + moe.last_load_balance_loss + moe.last_z_loss
    loss.backward()  # 修复前在此报错
    assert x.grad is not None
    assert x.grad.device == dev
    assert moe.router.weight.grad is not None
    assert moe.router.weight.grad.device == dev
    assert torch.isfinite(moe.router.weight.grad).all()
    # 所有专家有梯度
    for e in moe.experts:
        for p in e.parameters():
            assert p.grad is not None
            assert p.grad.device == dev


def test_model_moe_dml_backward_full():
    """R36-7-6: 完整 MoE 模型在 DML 上 forward + backward（含 aux 损失跨层累积）。"""
    dev = _dml_device()
    torch.manual_seed(5)
    m = _build(layers=3, vocab=50).to(dev)
    m.train()
    src = torch.randint(0, 50, (2, 8), device=dev)
    tgt = torch.randint(0, 50, (2, 8), device=dev)
    logits = m(src, targets=tgt)
    assert logits.device == dev
    logits = logits.view(-1, 50)
    loss = nn.functional.cross_entropy(logits, tgt.view(-1))
    if m._moe_load_balance_loss is not None:
        loss = loss + m._moe_load_balance_loss + m._moe_z_loss
    loss.backward()  # 修复前 MoE backward 在此报错
    # router 每层都有梯度
    for blk in m.blocks:
        assert isinstance(blk.ffn, MoELayer)
        assert blk.ffn.router.weight.grad is not None
        assert blk.ffn.router.weight.grad.device == dev


def test_model_moe_dml_backward_top1():
    """R36-7-7: DML 上 top_k=1（退化为单专家选择）backward 仍正常。"""
    dev = _dml_device()
    torch.manual_seed(6)
    m = _build(layers=2, moe_top_k=1, moe_num_experts=4).to(dev)
    m.train()
    src = torch.randint(0, 50, (2, 8), device=dev)
    tgt = torch.randint(0, 50, (2, 8), device=dev)
    logits = m(src, targets=tgt).view(-1, 50)
    loss = nn.functional.cross_entropy(logits, tgt.view(-1))
    if m._moe_load_balance_loss is not None:
        loss = loss + m._moe_load_balance_loss
    loss.backward()
    assert m.blocks[0].ffn.router.weight.grad is not None


# ===========================================================================
# 5. DML 训练：mini 训练循环 loss 下降
# ===========================================================================

def test_moe_dml_training_loss_decreases():
    """R36-7-8: DML 上 MoE 模型 mini 训练循环 loss 下降（端到端 backward + SGD 更新）。

    这是 R36-7 的端到端验证：修复前 DML 上 MoE 训练首步 backward 即报错，
    修复后可完整训练多步且 loss 下降。
    """
    dev = _dml_device()
    torch.manual_seed(7)
    m = _build(layers=2, vocab=50, moe_top_k=2, moe_num_experts=4).to(dev)
    m.train()
    # SGD + 低 lr + 梯度裁剪：随机目标下避免发散（train.py 用 lr=0.3 但配合真实数据 +
    # gradient_clip=1.0；此处随机数据梯度噪声大，故 lr 降到 0.02 并裁剪）
    opt = torch.optim.SGD(m.parameters(), lr=0.02, momentum=0.9)

    # 固定一个小数据集，重复训练（模型过拟合该 batch → loss 下降）
    src = torch.randint(0, 50, (4, 8), device=dev)
    tgt = torch.randint(0, 50, (4, 8), device=dev)

    losses = []
    for _ in range(30):
        opt.zero_grad()
        logits = m(src, targets=tgt).view(-1, 50)
        loss = nn.functional.cross_entropy(logits, tgt.view(-1))
        if m._moe_load_balance_loss is not None:
            loss = loss + m._moe_load_balance_loss + m._moe_z_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)  # 与 train.py 一致
        opt.step()
        losses.append(loss.item())

    # 全程无 NaN（梯度裁剪生效）
    assert all(not (isinstance(l, float) and l != l) for l in losses), (
        f"训练出现 NaN: {losses}")
    # loss 下降：末段均值显著低于首段（模型过拟合固定 batch）
    first_avg = sum(losses[:3]) / 3
    last_avg = sum(losses[-3:]) / 3
    assert last_avg < first_avg, (
        f"DML 训练 loss 未下降: first3={first_avg:.4f} last3={last_avg:.4f} "
        f"全程={losses}")
