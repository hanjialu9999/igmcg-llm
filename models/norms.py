from __future__ import annotations
from typing import Optional
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
