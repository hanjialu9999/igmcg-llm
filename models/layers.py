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

    增量滚动缓冲（`char_merge_incremental_buffer`，默认关）：
    - 逐 token（T=1）路径原本 `F.pad(x_t,(self.pad,0))` 只左补零，窗口退化为
      `[0,0,x_t]`，丢掉真实的前 pad 个 token —— 这是第二处训推不一致的根因
      （旧实现 `:42`，全量 prefix=5.378725 / 增量 incr=5.907944）。
    - 开启后保存最近 `self.pad` 个输入作左邻域，逐 token 用它替代零填充；
      算式与全量 conv1d 逐位置等价 → 增量结果与整段前向对齐。
    - 只用 `cat` + 切片（DML 禁 in-place / scatter，R35 教训）。
    """

    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.0,
                 gate_bias_init: float = -1.0, incremental_buffer: bool = False):
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
        # 增量滚动缓冲开关（默认 False = 走原零填充路径，逐位与旧码一致）
        self.incremental_buffer = incremental_buffer
        # 跨步状态：最近 self.pad 个输入 (B, pad, D)；None = 未初始化/已复位
        self._cm_buffer: Optional[torch.Tensor] = None

    def reset_buffer(self) -> None:
        """清空跨步滚动缓冲。

        与 `_controller_past` 同处调用（`transformer.reset_ngram_state()`），
        在新一次生成开头把上一序列的尾巴丢掉，防止跨序列串扰（M9/N3 同类坑）。
        """
        self._cm_buffer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        if self.incremental_buffer and self.pad > 0:
            buf = self._cm_buffer
            # batch 尺寸/宽度变化必须重建（形状不同无法复用旧行）；重建为零
            # 等价旧码零填充，首个 token 仍从"无历史"起步。
            if buf is None or buf.shape[0] != B or buf.shape[2] != D:
                buf = x.new_zeros(B, self.pad, D)
            # 左邻域 = [历史 pad 个, 当前 T 个]；先 cat 再转置，避免 in-place
            merged = torch.cat([buf, x], dim=1)          # (B, pad+T, D)
            x_padded = merged.transpose(1, 2)             # (B, D, pad+T)
            # 缓冲只留最近 pad 个：cat + 切片（无 scatter，DML 安全）
            self._cm_buffer = merged[:, -self.pad:, :]
        else:
            # 邻域聚合：因果左侧填充（仅 pad 左边），卷积后取前 T 个位置
            x_t = x.transpose(1, 2)  # (B, D, T)
            # 左侧零填充 pad 个位置（因果：位置 t 只看 t-pad..t）
            x_padded = F.pad(x_t, (self.pad, 0))  # (B, D, T+pad)
        agg = F.conv1d(x_padded, self.conv.weight, None, groups=D)  # (B, D, T)
        agg = agg.transpose(1, 2)  # (B, T, D)
        # 门控：当前字符 vs 邻域聚合
        # R33 convex_combine：z*agg + (1-z)*x → x + z*(agg-x)，5 算子→4 算子（与 R29.5 同模式）
        # R35 优化：addcmul 融合（DML 前向 1.28x），但 addcmul 的 backward 在 DML
        #   广播路径报错（[1,1,1,64] vs [1,256,1,64]，R38 实测），不可用。
        # R35 续：torch.lerp fused kernel——DML 不支持 aten::lerp.Tensor_out，
        #   函数式 torch.lerp 未被 train.py 的 lerp_/_foreach_lerp_ patch 覆盖 → 每层
        #   每步 CPU 回退 + DML↔CPU 同步（R38 实测 err 警告）。
        # R38 最终：x + z*(agg-x) 4 算子，mul/sub/add 全 DML 原生，前向/反向均无回退。
        z = torch.sigmoid(self.gate(x))
        out = x + z * (agg - x)
        out = self.norm(out)
        return self.drop(out)
