"""DML 兼容性回归测试：覆盖 3 个已修复的 DML bug，防止退化。

覆盖清单：
1. CrossLayerRouter top-k 掩码（广播比较替代 torch.eye/F.one_hot）
   - bug: torch.eye 回退 CPU + F.one_hot 触发 scatter 错误
   - 修复: (topk_idx.unsqueeze(-1) == pos).any(dim=1) 广播比较
2. _parallel_prefix_scan（cat+where 替代 roll+切片赋值）
   - bug: torch.roll 回退 CPU + 切片赋值触发 scatter 错误
   - 修复: torch.cat 替代 roll + torch.where 替代切片赋值
3. AdamW lerp_ monkey-patch（mul_+add_ 替代 lerp_）
   - bug: aten::lerp.Scalar_out 不支持 DML，每步回退 CPU
   - 修复: lerp_(end, w) → mul_(1-w).add_(end*w)
   - 注意：不能用 add_(end, alpha=w)——DML 的 add_ alpha 参数有 bug（曾导致 val_loss 16→正常 7.5）
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer import TransformerModel, CrossLayerRouter
from models.mixers import _parallel_prefix_scan, SlidingWindowCausalSelfAttention


# ============================================================
# 1. CrossLayerRouter top-k 掩码正确性
# ============================================================

def test_cross_layer_router_topk_mask():
    """验证广播比较构建的 top-k 掩码与 F.one_hot 等价。

    抓住：torch.eye/F.one_hot 在 DML 上的 scatter 错误，
    确保广播比较替代方案数学等价。
    """
    B, num_prev, k = 4, 6, 2
    scores = torch.randn(B, num_prev)
    _, topk_idx = torch.topk(scores, k, dim=-1)  # (B, k)

    # 修复后的实现：广播比较
    pos = torch.arange(num_prev)
    mask_new = (topk_idx.unsqueeze(-1) == pos.view(1, 1, -1)).any(dim=1).float()

    # 参考实现：F.one_hot（CPU 上可用）
    mask_ref = torch.eye(num_prev)[topk_idx].sum(dim=1)

    assert torch.equal(mask_new, mask_ref), "top-k 掩码与参考实现不一致"
    # 每行恰好 k 个 1
    assert (mask_new.sum(dim=1) == k).all(), f"每行应有 {k} 个 1"


def test_cross_layer_router_forward():
    """验证 CrossLayerRouter 前向不崩溃且输出形状正确。"""
    D = 64
    router = CrossLayerRouter(num_layers=4, dim=D, topk=2)
    x = torch.randn(2, 8, D)
    prev_outputs = [torch.randn(2, 8, D), torch.randn(2, 8, D)]
    out = router.route(2, x, prev_outputs)
    assert out.shape == x.shape, f"输出形状 {out.shape} != 输入 {x.shape}"


def test_cross_layer_router_first_layer_skip():
    """验证第一层（layer_idx=0）直接返回，不执行路由。"""
    D = 64
    router = CrossLayerRouter(num_layers=4, dim=D, topk=2)
    x = torch.randn(2, 8, D)
    out = router.route(0, x, [])
    assert torch.equal(out, x), "第一层应直接返回 x"


# ============================================================
# 2. _parallel_prefix_scan 正确性
# ============================================================

def test_parallel_prefix_scan_correctness():
    """验证 cat+where 实现的 prefix scan 与朴素递推等价。

    抓住：torch.roll + 切片赋值在 DML 上的 scatter 错误，
    确保 cat+where 替代方案数学等价。
    """
    B, L, d_inner, d_state = 2, 8, 4, 3
    a = torch.rand(B, L, d_inner, d_state)
    b = torch.rand(B, L, d_inner, d_state)

    # 修复后的实现
    out_new = _parallel_prefix_scan(a, b)

    # 朴素递推参考实现
    out_ref = torch.zeros_like(b)
    h = torch.zeros(B, d_inner, d_state)
    for t in range(L):
        h = a[:, t] * h + b[:, t]
        out_ref[:, t] = h

    assert torch.allclose(out_new, out_ref, atol=1e-6), \
        "prefix scan 与朴素递推不一致"


def test_parallel_prefix_scan_with_past_state():
    """验证带 past_state 的 prefix scan 正确性。"""
    B, L, d_inner, d_state = 2, 4, 3, 2
    a = torch.rand(B, L, d_inner, d_state)
    b = torch.rand(B, L, d_inner, d_state)
    past = torch.randn(B, d_inner, d_state)

    out = _parallel_prefix_scan(a, b, past_state=past)

    # 朴素递推验证
    h = past
    for t in range(L):
        h = a[:, t] * h + b[:, t]
        assert torch.allclose(out[:, t], h, atol=1e-6), f"step {t} 不一致"


def test_parallel_prefix_scan_no_grad_path():
    """验证 requires_grad=False 路径也正确（推理路径）。"""
    B, L, d_inner, d_state = 2, 8, 4, 3
    a = torch.rand(B, L, d_inner, d_state, requires_grad=False)
    b = torch.rand(B, L, d_inner, d_state, requires_grad=False)

    out = _parallel_prefix_scan(a, b)

    # 朴素递推验证
    h = torch.zeros(B, d_inner, d_state)
    for t in range(L):
        h = a[:, t] * h + b[:, t]
        assert torch.allclose(out[:, t], h, atol=1e-6), f"step {t} 不一致"


# ============================================================
# 3. AdamW lerp_ monkey-patch 数学等价性
# ============================================================

def test_lerp_replacement_equivalence():
    """验证 mul_+add_ 替代 lerp_ 的数学等价性。

    抓住：aten::lerp.Scalar_out 不支持 DML 回退 CPU，
    确保 monkey-patch 后的 mul_+add_ 数学等价。
    """
    x = torch.randn(3, 4)
    end = torch.randn(3, 4)
    weight = 0.1

    # 参考：lerp_
    x_ref = x.clone()
    x_ref.lerp_(end, weight)

    # 修复：mul_+add_（DML-safe，不能用 alpha= 参数）
    x_new = x.clone()
    x_new.mul_(1 - weight).add_(end * weight)

    assert torch.allclose(x_ref, x_new, atol=1e-7), \
        "lerp_ 与 mul_+add_ 不等价"


def test_lerp_replacement_various_weights():
    """验证不同 weight 值下的等价性。"""
    for weight in [0.0, 0.1, 0.5, 0.9, 1.0]:
        x = torch.randn(5)
        end = torch.randn(5)

        x_ref = x.clone()
        x_ref.lerp_(end, weight)

        x_new = x.clone()
        x_new.mul_(1 - weight).add_(end * weight)

        assert torch.allclose(x_ref, x_new, atol=1e-7), \
            f"weight={weight} 时 lerp_ 与 mul_+add_ 不等价"


def test_adamw_step_with_patched_lerp():
    """验证 monkey-patch lerp_ 后 AdamW 仍能正常更新参数。"""
    # 创建简单模型
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    # 保存初始参数
    init_weight = model.weight.clone()

    # 模拟训练步
    x = torch.randn(3, 4)
    y = torch.randn(3, 2)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()

    # 应用 monkey-patch 后的 lerp_（模拟 DML 环境，DML-safe 版本）
    _orig_lerp = torch.Tensor.lerp_
    def _dml_lerp(self, end, weight):
        return self.mul_(1 - weight).add_(end * weight)
    torch.Tensor.lerp_ = _dml_lerp

    try:
        optimizer.step()
    finally:
        torch.Tensor.lerp_ = _orig_lerp

    # 验证参数已更新
    assert not torch.equal(model.weight, init_weight), "参数未更新"


# ============================================================
# 4. 端到端：全特性模型 forward 不崩溃
# ============================================================

def test_all_features_forward_cpu():
    """验证全特性模型在 CPU 上 forward 不崩溃（DML bug 修复的回归测试）。

    这是一个综合测试，确保所有 DML 兼容性修复在 CPU 上也正确工作。
    """
    from models.config_loader import build_model

    config = {
        'model': {
            'vocab_size': 100, 'embedding_dim': 32, 'num_heads': 4,
            'num_layers': 4, 'hidden_dim': 64, 'max_seq_length': 16,
            'layer_plan': 'attn,hybrid,hybrid,attn', 'mixer': 'attn_linear',
            'alibi': True, 'ssm_d_state': 8,
            # DML bug 相关特性
            'cross_layer_routing': True, 'cross_layer_topk': 2,
            'char_merge': True, 'ssm_as_memory': True,
            'rope_dim_fraction': 0.5, 'output_gate': True,
            'intra_hybrid_rope': True, 'gpas': True,
        }
    }
    model = build_model(config, device='cpu')
    model.eval()
    x = torch.randint(0, 100, (2, 8))
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 8, 100), f"输出形状错误: {out.shape}"


# ============================================================
# 5. 死代码清除回归：_build_masks / _mask / _rbias 不应存在
# ============================================================

def test_no_dead_code_build_masks():
    """回归：_build_masks 创建的 self._mask / self._rbias 从未被读取，
    是每步浪费 DML 算力的死代码。确保不会回归。

    历史 bug：_build_masks 每步创建 bool mask (torch.triu + torch.ones)，
    但实际 SDPA 用的是 _build_causal_window_mask 返回的 float mask 或 is_causal=True。
    删除后每步少一次 T×T bool 张量分配 + window>0 时的 arange×2。
    """
    attn = SlidingWindowCausalSelfAttention(64, 4, max_seq_length=32, window=4)
    # 死代码属性不应存在
    assert not hasattr(attn, '_mask'), "_mask 是死代码，不应存在"
    assert not hasattr(attn, '_rbias'), "_rbias 是死代码，不应存在"
    assert not hasattr(attn, '_cached_T'), "_cached_T 是死代码，不应存在"
    assert not hasattr(attn, '_build_masks'), "_build_masks 是死代码方法，不应存在"
    # 实际使用的缓存应存在
    assert hasattr(attn, '_bias_cache'), "_bias_cache 是实际使用的掩码缓存，应存在"
    assert hasattr(attn, '_bias_key'), "_bias_key 是缓存键，应存在"


def test_attention_forward_without_dead_code():
    """验证删除 _build_masks 后注意力前向仍正确（窗口+因果掩码正确生效）。"""
    D, H, window, T = 32, 2, 4, 10
    attn = SlidingWindowCausalSelfAttention(D, H, max_seq_length=32, window=window)
    attn.eval()
    x = torch.randn(1, T, D)
    with torch.no_grad():
        _ = attn(x, use_cache=False)
    # _bias_cache 是实际传给 SDPA 的 float mask
    mask = attn._bias_cache[0, 0]  # (T, T)
    for q in range(T):
        for k in range(T):
            should_mask = (k > q) or (q - k > window)  # 因果 OR 窗口外
            is_masked = mask[q, k].item() != 0
            assert is_masked == should_mask, (
                f"删除死代码后掩码错误 mask[{q},{k}]: 期望 masked={should_mask}, 实际 {is_masked}")
