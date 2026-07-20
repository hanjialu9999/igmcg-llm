from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.norms import RMSNorm


class CharMergeLayer(nn.Module):
    """轻量学习型分词层（Learned Segmentation）。

    输入为字符级序列 (B, T, D)，通过双向门控卷积把相邻字符向量融合成
    "词级表示"。融合门控由模型自己学习，受 LM loss 监督 —— 即"切词/合并词"
    变成可微、可优化的过程，无需静态 BPE 词表。开销仅约注意力的 1~2%。

    设计：
    - depthwise 因果卷积（kernel=3）提取左右邻域聚合；
    - 门控 z = sigmoid(线性) 在"原始字符向量"与"邻域聚合"间插值；
    - 残差连接保证字符信息不丢。
    """

    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.0,
                 gate_bias_init: float = -1.0):
        super().__init__()
        self.dim = dim
        # 因果卷积：仅左侧填充（kernel-1 个零），位置 t 只看 t-(k-1)..t，不窥未来
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=True)
        # 门控偏置初始化：bias=-1 → sigmoid(-1)≈0.27，初期偏向字符表示（少融合），
        # 避免早期训练方差过大；随训练推进 gate 自适应调整融合比例。
        nn.init.constant_(self.gate.bias, gate_bias_init)
        self.norm = RMSNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        # 邻域聚合：因果左侧填充（仅 pad 左边），卷积后取前 T 个位置
        x_t = x.transpose(1, 2)  # (B, D, T)
        # 左侧零填充 pad 个位置（因果：位置 t 只看 t-pad..t）
        x_padded = F.pad(x_t, (self.pad, 0))  # (B, D, T+pad)
        agg = F.conv1d(x_padded, self.conv.weight, None, groups=D)  # (B, D, T)
        agg = agg.transpose(1, 2)  # (B, T, D)
        # 门控：当前字符 vs 邻域聚合
        z = torch.sigmoid(self.gate(x))
        out = z * agg + (1 - z) * x
        out = self.norm(out)
        return self.drop(out)
