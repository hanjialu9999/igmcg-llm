from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Any
import torch


@dataclass
class BlockState:
    """单层 TransformerBlock 的增量解码状态（替代裸元组嵌套）。

    用法：
        # 从旧元组转换
        state = BlockState.from_tuple(past_kv)
        # 访问字段（替代 past_kv[0][0] 等索引）
        k, v = state.attn_kv
        # 转回元组（供旧接口使用）
        tuple_repr = state.to_tuple()
    """
    # 注意力层 KV-cache：(k, v) 或 (k, v, linear_S, z) 或 None
    attn_kv: Optional[Tuple] = None
    # SSM 隐藏状态 (B, d_inner, d_state) 或 None
    ssm_hidden: Optional[torch.Tensor] = None
    # SSM 卷积状态 (B, d_inner, conv_kernel-1) 或 None
    ssm_conv: Optional[torch.Tensor] = None

    @classmethod
    def from_tuple(cls, past_kv: Optional[Tuple]) -> Optional[BlockState]:
        """从旧的 (attn_kv, ssm_state, ssm_conv_state) 元组转换。"""
        if past_kv is None:
            return None
        return cls(
            attn_kv=past_kv[0] if len(past_kv) > 0 else None,
            ssm_hidden=past_kv[1] if len(past_kv) > 1 else None,
            ssm_conv=past_kv[2] if len(past_kv) > 2 else None,
        )

    def to_tuple(self) -> Tuple[Optional[Tuple], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """转回旧的三元组格式（保持向后兼容）。"""
        return (self.attn_kv, self.ssm_hidden, self.ssm_conv)

    @property
    def is_fresh(self) -> bool:
        """所有状态均为 None（新序列起点）。"""
        return self.attn_kv is None and self.ssm_hidden is None and self.ssm_conv is None

    @property
    def start_pos(self) -> int:
        """从 attn_kv 推断当前位置。

        两种缓存格式：
        - 标准：(k, v)，k 为 4D (B, H, T, head_dim) → seq_len = k.size(2)
        - MLA：(c_kv, None)，c_kv 为 3D (B, T, kv_latent_dim) → seq_len = c_kv.size(1)
        """
        if self.attn_kv is not None and isinstance(self.attn_kv, tuple):
            if len(self.attn_kv) >= 1 and isinstance(self.attn_kv[0], torch.Tensor):
                t = self.attn_kv[0]
                # 4D = 标准 KV cache (B,H,T,D)；3D = MLA 潜空间 cache (B,T,latent)
                return t.size(2) if t.dim() == 4 else t.size(1)
        return 0
