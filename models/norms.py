from __future__ import annotations
from typing import Optional
import math
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization（LLaMA 风格，比 LayerNorm 更省且更稳定）。

    第十五轮新增 zero_centered 选项（灵感：Qwen3-Next Zero-Centered RMSNorm）：
    计算 rms(x - mean) 而非 rms(x)，先去均值再归一化。
    防止 norm 权重异常增大（Massive Activation），DML 上提升数值稳定性。
    默认 False（向后兼容），config 显式开启。
    """
    def __init__(self, dim: int, eps: float = 1e-6, zero_centered: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.zero_centered = zero_centered

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.zero_centered:
            # Zero-Centered: 先去均值，再 rms 归一化（防止 Massive Activation）
            mean = x.mean(-1, keepdim=True)
            x = x - mean
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class GPASNorm(nn.Module):
    """Gradient-Preserving Activation Scaling (GPAS)。

    Pre-LN 架构中深层残差通路方差指数增长（x_n 方差 >> 子层输出方差），
    导致深层子层贡献被淹没。GPAS 在 LN 输出后、子层前用可学标量 α∈(0,1)
    缩放激活：out = α · LN(x)，控制子层输入幅度。

    α 通过 sigmoid 限幅到 (0,1)，init raw=logit(init_alpha)。
    默认 init_alpha=0.5（raw=0 → sigmoid=0.5，中等缩放）。
    默认关（gpas=False），config 显式开启。灵感：arXiv:2506.22049。
    """
    def __init__(self, init_alpha: float = 0.5):
        super().__init__()
        clamped = max(min(init_alpha, 0.999), 0.001)
        raw_val = math.log(clamped / (1.0 - clamped))
        self.gpas_raw = nn.Parameter(torch.tensor(raw_val))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gpas_raw) * x
