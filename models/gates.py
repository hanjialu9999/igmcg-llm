"""统一门控抽象（第十六轮代码债清理）。

项目内 17+ 种门控机制的公共模式提取，不改变 state_dict 结构。
所有函数对 gate=None 返回原值，消除 forward 中散乱的 if-None 分支。

5 种模式：
  1. direct: gate * h（静态标量直接乘，不过 sigmoid）
  2. sigmoid_scalar: sigmoid(param) * h（标量经 sigmoid）
  3. linear_sigmoid: h * sigmoid(W·x + b)（Linear 门控）
  4. convex_combine: g*h1 + (1-g)*h2（凸组合，标量或 Linear）
  5. correct: h + sigmoid(param)*(lh - h)（线性注意力修正）

对应现有门控：
  direct          → sub1_gate, ffn_gate, hybrid_attn_gate, hybrid_ssm_gate
  sigmoid_scalar  → skip_gate
  linear_sigmoid  → sub1_highway, ffn_highway
  convex_combine  → mixer_gate(标量), hybrid_mix(Linear)
  correct         → correction_gate
"""
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn


@dataclass
class GateConfig:
    """TransformerBlock 门控配置，打包所有门控开关。

    替代 __init__ 中散落的 6 个 bool 参数，减少签名复杂度。
    所有字段默认值与原 __init__ 默认值一致，保证向后兼容。
    """
    residual_gate: bool = True        # sub1_gate/ffn_gate 静态残差门控
    hybrid_gate: bool = True          # hybrid_attn_gate/hybrid_ssm_gate
    highway_gate: bool = False        # sub1_highway/ffn_highway 动态残差门控
    skip: bool = False                # skip_gate 跳层门控
    hybrid_single_gate: bool = False  # hybrid_mix 单门控凸组合
    linear_correction: bool = False   # correction_gate 线性注意力修正

    @classmethod
    def from_kwargs(cls, **kwargs) -> 'GateConfig':
        """从散落的 bool 参数构造（兼容旧调用方式）。"""
        return cls(
            residual_gate=kwargs.get('residual_gate', True),
            hybrid_gate=kwargs.get('hybrid_gate', True),
            highway_gate=kwargs.get('highway_gate', False),
            skip=kwargs.get('skip', False),
            hybrid_single_gate=kwargs.get('hybrid_single_gate', False),
            linear_correction=kwargs.get('linear_correction', False),
        )


# ---- 模式 1: direct（静态标量直接乘） ----

def apply_direct(gate: Optional[torch.Tensor], h: torch.Tensor) -> torch.Tensor:
    """gate * h。gate=None 时返回原值。

    用于: sub1_gate, ffn_gate, hybrid_attn_gate, hybrid_ssm_gate
    """
    return gate * h if gate is not None else h


# ---- 模式 2: sigmoid_scalar（标量经 sigmoid） ----

def apply_sigmoid_scalar(param: Optional[nn.Parameter], h: torch.Tensor) -> torch.Tensor:
    """sigmoid(param) * h。param=None 时返回原值。

    用于: skip_gate
    """
    return torch.sigmoid(param) * h if param is not None else h


# ---- 模式 3: linear_sigmoid（Linear 门控） ----

def apply_linear_gate(linear: Optional[nn.Linear], x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """h * sigmoid(W·x + b)。linear=None 时返回原值。

    用于: sub1_highway, ffn_highway
    """
    return h * torch.sigmoid(linear(x)) if linear is not None else h


# ---- 模式 4: convex_combine（凸组合） ----

def convex_combine_scalar(param: nn.Parameter, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    """g*h1 + (1-g)*h2，g=sigmoid(param)。

    用于: mixer_gate
    """
    g = torch.sigmoid(param)
    return g * h1 + (1.0 - g) * h2


def convex_combine_linear(linear: nn.Linear, x: torch.Tensor,
                          h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    """g*h1 + (1-g)*h2，g=sigmoid(W·x+b)。

    用于: hybrid_mix
    """
    g = torch.sigmoid(linear(x))
    return g * h1 + (1.0 - g) * h2


# ---- 模式 5: correct（线性注意力修正） ----

def apply_correction(param: nn.Parameter, h: torch.Tensor, lh: torch.Tensor) -> torch.Tensor:
    """h + sigmoid(param) * (lh - h)（线性注意力修正）。

    用于: correction_gate
    """
    cg = torch.sigmoid(param)
    return h + cg * (lh - h)
