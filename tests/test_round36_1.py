"""第三十六轮-1 回归测试：gradient_checkpointing 自动禁用（grad_ckpt_auto）。

验证 R36-1 优化：
1. grad_ckpt_auto 默认关，不改变 gradient_checkpointing 原行为
2. 开启时小模型自动关闭 gradient_checkpointing（激活内存 < 512MB）
3. 大模型保持开启
4. 显式 gradient_checkpointing=False 时 grad_ckpt_auto 不强制开
5. 启发式阈值边界（刚好低于/高于 512MB）
6. 从 config dict / ModelConfig dataclass 构建
7. block 级同步
8. set_gradient_checkpointing 方法
9. 训练前向+反向
10. cache parity（推理路径不经过 ckpt）

启发式公式：total_bytes = L * B * T * (4D + H*T + 3*HD) * 4 (fp32, B=32)
阈值 512MB = 536870912 bytes
"""
import pytest
import torch

from models.model_config import ModelConfig
from models.config_loader import build_model
from models.transformer import TransformerModel


def _build_direct(dim=256, heads=4, layers=4, hidden=512, seq=64,
                  gc=True, auto=False, **kw):
    """直接构造 TransformerModel（快，不依赖 config 加载）。"""
    return TransformerModel(
        vocab_size=100, embedding_dim=dim, num_heads=heads, num_layers=layers,
        hidden_dim=hidden, max_seq_length=seq,
        gradient_checkpointing=gc, grad_ckpt_auto=auto, **kw
    )


def _build_from_config(dim=256, heads=4, layers=4, hidden=512, seq=64,
                       gc=True, auto=False):
    """从 config dict 构建（验证 config_loader 路径）。"""
    config = {
        'model': {
            'vocab_size': 100, 'embedding_dim': dim, 'num_heads': heads,
            'num_layers': layers, 'hidden_dim': hidden, 'max_seq_length': seq,
            'gradient_checkpointing': gc, 'grad_ckpt_auto': auto,
        }
    }
    return build_model(config, device='cpu')


# 小模型（~6MB / ~88MB，远低于 512MB → 应自动关闭）
SMALL = dict(dim=256, heads=4, layers=4, hidden=512, seq=64)
# 默认 _build_direct 小模型（~6MB）
TINY = dict(dim=64, heads=4, layers=2, hidden=128, seq=32)
# 边界低于 512MB（~432MB → 应自动关闭）
BOUNDARY_BELOW = dict(dim=384, heads=6, layers=6, hidden=768, seq=128)
# 边界高于 512MB（~576MB → 应保持开启）
BOUNDARY_ABOVE = dict(dim=384, heads=6, layers=8, hidden=768, seq=128)
# 大模型（~1.75GB → 应保持开启）
LARGE = dict(dim=512, heads=8, layers=8, hidden=1024, seq=256)


def test_grad_ckpt_auto_default_off():
    """R36-1-1: 默认 grad_ckpt_auto=False，不改变 gradient_checkpointing。"""
    m = _build_direct(gc=True, auto=False, **SMALL)
    assert m.grad_ckpt_auto is False
    assert m.gradient_checkpointing is True  # 保持原值


def test_grad_ckpt_auto_small_model_disables():
    """R36-1-2: 小模型 + grad_ckpt_auto=True → 自动关闭 gradient_checkpointing。"""
    m = _build_direct(gc=True, auto=True, **SMALL)
    assert m.grad_ckpt_auto is True
    assert m.gradient_checkpointing is False  # 自动关闭
    # block 级同步
    for blk in m.blocks:
        assert blk.gradient_checkpointing is False


def test_grad_ckpt_auto_tiny_model_disables():
    """R36-1-3: 极小模型 + grad_ckpt_auto=True → 自动关闭。"""
    m = _build_direct(gc=True, auto=True, **TINY)
    assert m.gradient_checkpointing is False


def test_grad_ckpt_auto_large_model_keeps():
    """R36-1-4: 大模型 + grad_ckpt_auto=True → 保持开启。"""
    m = _build_direct(gc=True, auto=True, **LARGE)
    assert m.grad_ckpt_auto is True
    assert m.gradient_checkpointing is True  # 保持开启
    for blk in m.blocks:
        assert blk.gradient_checkpointing is True


def test_grad_ckpt_auto_explicit_gc_false():
    """R36-1-5: gradient_checkpointing=False 时 grad_ckpt_auto 不强制开。"""
    m = _build_direct(gc=False, auto=True, **SMALL)
    assert m.gradient_checkpointing is False  # 保持 False
    assert m.grad_ckpt_auto is True


def test_should_disable_ckpt_small_true():
    """R36-1-6: _should_disable_ckpt 小模型返回 True。"""
    m = _build_direct(gc=False, auto=False, **SMALL)
    assert m._should_disable_ckpt() is True


def test_should_disable_ckpt_large_false():
    """R36-1-7: _should_disable_ckpt 大模型返回 False。"""
    m = _build_direct(gc=False, auto=False, **LARGE)
    assert m._should_disable_ckpt() is False


def test_should_disable_ckpt_boundary_below():
    """R36-1-8: 边界值（刚好低于 512MB，~432MB）→ True。"""
    m = _build_direct(gc=False, auto=False, **BOUNDARY_BELOW)
    assert m._should_disable_ckpt() is True


def test_should_disable_ckpt_boundary_above():
    """R36-1-9: 边界值（刚好高于 512MB，~576MB）→ False。"""
    m = _build_direct(gc=False, auto=False, **BOUNDARY_ABOVE)
    assert m._should_disable_ckpt() is False


def test_grad_ckpt_auto_from_config_dict():
    """R36-1-10: 从 config dict 构建（build_model 路径）正确触发自动关闭。"""
    m = _build_from_config(gc=True, auto=True, **SMALL)
    assert m.grad_ckpt_auto is True
    assert m.gradient_checkpointing is False  # 自动关闭


def test_grad_ckpt_auto_from_dataclass():
    """R36-1-11: 从 ModelConfig dataclass 构建（from_config 路径）。"""
    cfg = ModelConfig(
        vocab_size=100, embedding_dim=256, num_heads=4, num_layers=4,
        hidden_dim=512, max_seq_length=64,
        gradient_checkpointing=True, grad_ckpt_auto=True,
    )
    m = TransformerModel.from_config(cfg)
    assert m.gradient_checkpointing is False  # 自动关闭
    assert m.grad_ckpt_auto is True


def test_grad_ckpt_auto_from_config_dict_keeps_large():
    """R36-1-12: 从 config dict 构建大模型 → 保持开启。"""
    m = _build_from_config(gc=True, auto=True, **LARGE)
    assert m.gradient_checkpointing is True


def test_grad_ckpt_auto_default_in_modelconfig():
    """R36-1-13: ModelConfig 默认 grad_ckpt_auto=False。"""
    cfg = ModelConfig(
        vocab_size=100, embedding_dim=64, num_heads=4, num_layers=2,
        hidden_dim=128, max_seq_length=32,
    )
    assert cfg.grad_ckpt_auto is False


def test_grad_ckpt_auto_set_method_sync():
    """R36-1-14: set_gradient_checkpointing 方法正确同步到 block。"""
    m = _build_direct(gc=True, auto=False, **SMALL)
    # 关闭
    m.set_gradient_checkpointing(False)
    assert m.gradient_checkpointing is False
    for blk in m.blocks:
        assert blk.gradient_checkpointing is False
    # 重新开启
    m.set_gradient_checkpointing(True)
    assert m.gradient_checkpointing is True
    for blk in m.blocks:
        assert blk.gradient_checkpointing is True


def test_grad_ckpt_auto_training_forward_backward():
    """R36-1-15: grad_ckpt_auto 关闭后训练前向+反向不报错，梯度回流正常。"""
    m = _build_direct(gc=True, auto=True, **SMALL)
    m.train()
    assert m.gradient_checkpointing is False  # 确认自动关闭
    x = torch.randint(0, 100, (2, 8))
    logits = m(x)
    assert logits.shape == (2, 8, 100)
    loss = logits.sum()
    loss.backward()
    # 检查梯度回流
    has_grad = False
    for p in m.parameters():
        if p.requires_grad and p.grad is not None:
            has_grad = True
            break
    assert has_grad, "无梯度回流"


def test_grad_ckpt_auto_training_with_gc_enabled():
    """R36-1-16: 大模型保持 gc=True 时训练前向+反向正常（重算路径）。"""
    m = _build_direct(gc=True, auto=True, **LARGE)
    m.train()
    assert m.gradient_checkpointing is True  # 确认保持开启
    x = torch.randint(0, 100, (1, 16))
    logits = m(x)
    assert logits.shape == (1, 16, 100)
    loss = logits.sum()
    loss.backward()
    # 梯度回流
    has_grad = False
    for p in m.parameters():
        if p.requires_grad and p.grad is not None:
            has_grad = True
            break
    assert has_grad, "无梯度回流"


def test_grad_ckpt_auto_cache_parity():
    """R36-1-17: cache parity 不受 grad_ckpt_auto 影响（推理路径不经过 ckpt）。"""
    m = _build_direct(gc=True, auto=True, **SMALL)
    m.eval()
    torch.manual_seed(42)
    ids = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        # 全量前向
        out_full = m(ids, use_cache=False)
        if isinstance(out_full, dict):
            full = out_full["logits"]
        elif isinstance(out_full, tuple):
            full = out_full[0]
        else:
            full = out_full
        # 增量解码
        out, past = m(ids[:, :1], past_key_values=None, use_cache=True)
        for t in range(1, ids.size(1)):
            out, past = m(ids[:, t:t + 1], past_key_values=past, use_cache=True)
        if isinstance(out, dict):
            incr = out["logits"]
        elif isinstance(out, tuple):
            incr = out[0]
        else:
            incr = out
    # 数值等价（推理不经过 ckpt，应一致）
    diff = (full[:, -1, :] - incr[:, -1, :]).abs().max().item()
    assert diff < 1e-4, f"cache parity 失败: max_diff={diff:.2e}"


def test_grad_ckpt_auto_no_print_when_disabled():
    """R36-1-18: grad_ckpt_auto=False 时不触发自动关闭逻辑。"""
    # auto=False 时即使小模型也不应自动关闭
    m = _build_direct(gc=True, auto=False, **SMALL)
    assert m.gradient_checkpointing is True  # 保持原值


def test_grad_ckpt_auto_hidden_dim_stored():
    """R36-1-19: hidden_dim 属性正确存储（供启发式使用）。"""
    m = _build_direct(hidden=512, **{k: v for k, v in SMALL.items() if k != 'hidden'})
    assert m.hidden_dim == 512


def test_grad_ckpt_auto_backward_compat_no_field():
    """R36-1-20: 旧 config 不含 grad_ckpt_auto 字段时默认 False（向后兼容）。"""
    config = {
        'model': {
            'vocab_size': 100, 'embedding_dim': 256, 'num_heads': 4,
            'num_layers': 4, 'hidden_dim': 512, 'max_seq_length': 64,
            'gradient_checkpointing': True,
            # 不含 grad_ckpt_auto
        }
    }
    m = build_model(config, device='cpu')
    assert m.grad_ckpt_auto is False
    assert m.gradient_checkpointing is True  # 保持原值


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
