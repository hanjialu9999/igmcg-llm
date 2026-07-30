from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.mixers import SwiGLU


class MoELayer(nn.Module):
    """Mixture of Experts FFN（top-k 路由 + N 个 SwiGLU 专家）。

    架构（LLaMA/Mixtral 风格）：
      - Router: Linear(D, E, bias=False) → softmax → top-k 选择
      - Experts: N 个 SwiGLU(D, hidden_dim, fuse_swiglu)
      - 输出: top-k 专家输出的加权和（路由权重归一化后）

    DML 兼容设计（关键约束）：
      DirectML 后端有两个限制：(1) 不支持 gather/scatter 的 per-sample 动态索引；
      (2) topk/max 的 backward 用 scatter 回传梯度，但 DML scatter 不支持
      "partially modified dimensions"（R36-7 定位）。本模块应对：
      - **dense 计算**：所有专家对所有 token 求值，再用广播比较构建 top-k 掩码
        （element-wise == + any/sum，无 gather/scatter）。代价是损失了 MoE 的
        FLOPs 节省（计算所有专家），但保证 DML 可运行；对小模型可接受。
        路由稀疏性仍由组合权重（非 top-k 专家权重为 0）体现，负载均衡损失仍生效。
      - **topk 仅选位置（R36-7 修复）**：topk_indices 在 torch.no_grad() 下获取
        （绕过 topk backward 的 scatter 限制），组合权重改为
        combined = routing_probs * topk_mask（可导，梯度经 routing_probs→softmax→router
        回流）。数学/梯度均与原 topk_probs 路径等价（数值 0 diff 已验证）。

    辅助损失（forward 后存入 last_load_balance_loss / last_z_loss，
    由 TransformerModel.forward 累积，train.py 按各自权重加到主 loss）：
      - 负载均衡损失（Switch Transformer 公式，top-k 归一化）：
          L_bal = E · Σ_i f_i · p_i
          f_i = 选中专家 i 的 token 数占比（Σ_i f_i = 1；每 token 选 top_k 个）
          p_i = 平均 softmax 路由概率（Σ_i p_i = 1）
        均匀路由时 L_bal = 1（最小值）；偏斜路由时 >1。
      - Router z-loss（ST-MoE 公式，数值稳定性，防 logits 爆炸致 softmax 溢出）：
          L_z = (1/N_tok) · Σ_t (logsumexp(logits_t))²
        ST-MoE 标准系数 0.001。

    噪声（训练期探索，Shazeer 2017 noisy top-k gating）：
      router_noise > 0 时 logits += router_noise · randn_like(logits)，鼓励
      路由探索防坍塌。默认 0（关，opt-in）。仅训练期生效。
    """

    def __init__(self, dim: int, hidden_dim: int, num_experts: int = 8,
                 top_k: int = 2, fuse_swiglu: bool = False,
                 router_noise: float = 0.0):
        super().__init__()
        assert num_experts >= 1, f"num_experts must be >= 1, got {num_experts}"
        assert 1 <= top_k <= num_experts, (
            f"top_k must be in [1, num_experts], got top_k={top_k}, num_experts={num_experts}")
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_noise = float(router_noise)
        # Router: Linear(D, E, bias=False)
        self.router = nn.Linear(dim, num_experts, bias=False)
        # Experts: N 个 SwiGLU（共享 forward(x)->out 接口，与块内 self.ffn 一致）
        self.experts = nn.ModuleList([
            SwiGLU(dim, hidden_dim, fuse_swiglu=fuse_swiglu)
            for _ in range(num_experts)
        ])
        # 最近一次 forward 的辅助损失（仅训练期计算；eval/推理为 None 以省算力）
        self.last_load_balance_loss: Optional[torch.Tensor] = None
        self.last_z_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D) 任意前导维度 -> 输出同形状
        orig_shape = x.shape
        # 展平到 (N, D) 便于路由统计（N = 总 token 数；不影响结果，仅简化维度）
        x_flat = x.reshape(-1, self.dim)  # (N, D)
        # N = x_flat.size(0)

        # --- Router ---
        router_logits = self.router(x_flat)  # (N, E)
        # 训练期加噪声（noisy top-k gating，Shazeer 2017）——鼓励路由探索防坍塌
        if self.training and self.router_noise > 0:
            router_logits = router_logits + self.router_noise * torch.randn_like(router_logits)
        routing_probs = F.softmax(router_logits, dim=-1)  # (N, E)

        # --- Top-k 选择（DML 兼容：topk_indices 在 no_grad 下获取，绕过 topk backward 的 scatter 限制）---
        # R36-7: DML 后端 topk/max 的 backward 用 scatter 回传梯度，但 DML scatter 不支持
        # "partially modified dimensions"。故 topk 只用于选位置（detach），权重值取自
        # routing_probs（可导），梯度经 routing_probs→softmax→router 回流，绕过 topk backward。
        # 数学等价：combined[n,e] = routing_probs[n,e] if e∈topk else 0，归一化后与原
        # (match * topk_probs).sum 数值相等；梯度等价：combined 对 routing_probs 梯度 = topk_mask
        # （只 top-k 位置非零），经 softmax 映射到 router_logits 与原 topk backward scatter 一致。
        with torch.no_grad():
            _, topk_indices = routing_probs.topk(self.top_k, dim=-1)  # (N, k) 仅选位置，不回传梯度
        expert_ids = torch.arange(self.num_experts, device=x_flat.device)  # (E,)
        # topk_mask: (N, E) 0/1，每 token 选中的 top-k 专家（与原 match.any(dim=1) 等价；detached）
        topk_mask = (topk_indices.unsqueeze(-1) == expert_ids.view(1, 1, -1)).any(dim=1).to(routing_probs.dtype)

        # --- Dense 专家计算（DML 友好：无 gather/scatter）---
        # 所有专家对所有 token 求值（损失 FLOPs 但保证 DML 可运行）
        # expert_outs: (N, E, D)
        expert_outs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)

        # --- 构建组合权重（DML 友好 + 可导：routing_probs * topk_mask）---
        # combined[n, e] = routing_probs[n,e] if e ∈ topk_indices[n] else 0
        # 数值等价于原 (match * topk_probs).sum（topk_probs = gather(routing_probs)）；
        # 梯度等价：combined 对 routing_probs 梯度 = topk_mask（只 top-k 位置非零），经 softmax 回流 router
        combined = routing_probs * topk_mask  # (N, E) 可导
        combined = combined / combined.sum(dim=-1, keepdim=True).clamp(min=1e-9)  # 归一化（与原 topk_probs/sum 等价）

        # --- 加权求和（element-wise 乘 + reduce，DML 原生支持）---
        # out[n, d] = Σ_e combined[n,e] * expert_outs[n,e,d]
        out = (combined.unsqueeze(-1) * expert_outs).sum(dim=1)  # (N, D)

        # --- 辅助损失（仅训练期计算；eval/推理跳过省算力）---
        if self.training:
            # 负载均衡损失：L_bal = E · Σ_i f_i · p_i
            # f_i = 选中专家 i 的 token 数占比（每 token 选 top_k 个 → Σ f_i = top_k → 归一化使 Σ=1）
            # any(dim=1): 每个 token 在 top-k 中是否选中专家 e（top-k 返回 distinct 索引，故 0/1）
            selected = topk_mask  # (N, E) 0/1，每 token 是否选中专家 e（detached，与原 match.any 一致）
            f_i = selected.mean(dim=0) / float(self.top_k)  # (E,) Σ=1
            p_i = routing_probs.mean(dim=0)  # (E,) Σ=1
            self.last_load_balance_loss = self.num_experts * (f_i * p_i).sum()
            # Router z-loss：L_z = (1/N) · Σ_t (logsumexp(logits_t))²  (ST-MoE 公式)
            logsumexp = torch.logsumexp(router_logits, dim=-1)  # (N,)
            self.last_z_loss = (logsumexp ** 2).mean()
        else:
            self.last_load_balance_loss = None
            self.last_z_loss = None

        return out.reshape(orig_shape)

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, hidden_dim={self.hidden_dim}, "
                f"num_experts={self.num_experts}, top_k={self.top_k}, "
                f"router_noise={self.router_noise}")
