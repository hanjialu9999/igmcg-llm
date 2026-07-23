"""量化感知训练（QAT）：LSQ 风格可学习步长伪量化，让模型对量化鲁棒。

设计：
  - 不替换模块结构（保持 state_dict 兼容，checkpoint 可直接加载）
  - 通过 monkey-patch nn.Linear.forward 注入量化（权重 + 激活双量化）
  - 前向 round+clip 伪量化，反向 STE 直通梯度（w + (w_q - w).detach()）
  - 共享可学习步长（per-tensor，参数量极小：1 个标量）
  - 仅 training=True 时量化；eval 时恒等（推理无开销、与未量化模型完全一致）
  - 位宽 b 时 qmax = 2^(b-1)-1（对称量化）

用法：
    from models.qat import enable_qat
    enable_qat(model, bits=8)  # 训练前调用
    # 训练...（步长随 CE loss 学）
    # eval 时自动恒等，可正常推理（无需 disable）

设计权衡：
  - 共享步长（per-tensor）而非 per-channel/per-layer：参数量最小（1 标量），
    量化精度略低但足以让模型学适应；如需更精细可改为 per-layer 步长。
  - 仅量化 Linear（Embedding 不量化）：Embedding 是查表操作，量化会破坏索引；
    Linear 权重和激活是 QAT 主要受益点。
  - monkey-patch 而非 hook：forward_pre_hook 无法替换 module.weight 引用，
    只能改输入；要量化权重必须 patch forward。
"""

from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _fake_quant(x: torch.Tensor, scale: torch.Tensor, qmax: float) -> torch.Tensor:
    """LSQ 风格伪量化：前向 round+clip，反向 STE(x) + LSQ(scale)。

    前向：x_q = clip(round(x / s), -qmax, qmax) * s
    反向：
      - dx = 1（STE 直通，round/clip 视为恒等）
      - ds = r·grad（LSQ 步长梯度，r 为量化后的整数值）

    实现技巧（组合两项实现 dx=1 且 ds=r）：
      - x + (x_q - x).detach()  → forward=x_q, dx=1, ds=0  （STE for x）
      - (x_q - x_q.detach())    → forward=0,   dx=0, ds=r  （LSQ for s）
      和 → forward=x_q, dx=1, ds=r ✓
    """
    s = scale.abs() + 1e-8  # 步长恒正（避免除零）
    # r = round(x/s) clipped，detach 使 round/clip 反向恒等（STE）
    r = torch.clamp((x / s).round(), -qmax, qmax).detach()
    x_q = r * s  # 量化值；反向：ds=r·grad（LSQ），dx=0（r 已 detach）
    return x + (x_q - x).detach() + (x_q - x_q.detach())


def enable_qat(model: nn.Module, bits: int = 8) -> nn.Module:
    """启用 QAT：为所有 Linear 注入伪量化 forward（不修改 state_dict 结构）。

    Args:
        model: 目标模型
        bits: 量化位宽（4/8 等），<=0 时不启用

    Returns:
        model（已启用 QAT；eval 时恒等，可正常推理）

    注意：
        - 调用时机：模型构建 + checkpoint 加载之后、训练开始之前
        - 步长 _qat_scale 注册为 model 的 Parameter，随 model.to(device) 迁移
        - 优化器会自动包含 _qat_scale（model.parameters() 遍历到）
        - 如需关闭：调用 disable_qat(model)
    """
    if bits <= 0:
        return model
    if getattr(model, '_qat_enabled', False):
        return model  # 已启用，避免重复 patch
    qmax = float(2 ** (bits - 1) - 1)
    # 共享可学习步长：init 用模型所有 Linear 权重 std 估，避免过小导致 round 全 0
    # 在 CPU 上计算 std（DML 的 aten::std.correction 回退 CPU，直接 CPU 省一次传输）
    with torch.no_grad():
        all_w = torch.cat([m.weight.detach().cpu().flatten()
                           for m in model.modules() if isinstance(m, nn.Linear)])
        init_s = (all_w.std() / qmax).clamp(min=1e-6).item()
    model._qat_scale = nn.Parameter(torch.tensor(init_s))
    model._qat_bits = bits
    model._qat_enabled = True

    # monkey-patch 每个 Linear 的 forward：训练时伪量化权重+激活，eval 时恒等
    # 注意：权重缓存优化在 LSQ（可学步长）下不适用——_qat_scale 每步被优化器更新，
    # r=round(w/s) 随 s 变化，缓存 r 会用过时 scale 导致数值偏差（Val Loss 升高）。
    for m in model.modules():
        if isinstance(m, nn.Linear):
            def make_q_forward(mod: nn.Linear):
                def q_forward(x: torch.Tensor) -> torch.Tensor:
                    if not mod.training:
                        return F.linear(x, mod.weight, mod.bias)
                    w_q = _fake_quant(mod.weight, model._qat_scale, qmax)
                    out = F.linear(x, w_q, mod.bias)
                    out = _fake_quant(out, model._qat_scale, qmax)
                    return out
                return q_forward
            m.forward = make_q_forward(m)  # type: ignore[assignment]
    return model


def disable_qat(model: nn.Module) -> nn.Module:
    """禁用 QAT：恢复原始 Linear forward（删除 monkey-patch）。

    用于训练后想完全移除 QAT 行为（如导出模型）。
    eval 时 QAT 本就恒等，通常无需调用此函数。
    """
    if not getattr(model, '_qat_enabled', False):
        return model
    # 恢复原始 Linear.forward（重新绑定到 nn.Linear.forward 实现）
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # 重新绑定 forward 到 nn.Linear 的标准实现（删除 monkey-patch 闭包）
            m.forward = types.MethodType(nn.Linear.forward, m)  # type: ignore[assignment]
    if hasattr(model, '_qat_scale'):
        del model._qat_scale
    model._qat_enabled = False
    return model


def qat_status(model: nn.Module) -> dict:
    """返回 QAT 状态信息（调试用）。"""
    if not getattr(model, '_qat_enabled', False):
        return {'enabled': False}
    return {
        'enabled': True,
        'bits': model._qat_bits,
        'scale': float(model._qat_scale.detach().abs().item()),
        'qmax': float(2 ** (model._qat_bits - 1) - 1),
    }
