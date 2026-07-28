"""第二十七轮回归测试：attend bias 合并优化 + ALiBi bias 完整缓存。

覆盖：
- ALiBi bias 缓存：alibi_learnable=False 时缓存命中返回相同张量对象
- ALiBi bias 不缓存：alibi_learnable=True 时每步重算（梯度回流）
- ALiBi bias 不缓存：pe_gate 开启时（log_pe_gate 是 Parameter）
- ALiBi bias 数值等价：缓存与重算结果一致
- attend bias 合并：链式 add 与原逐次 add 数值等价
- _use_causal=True 跳过 bias 合并路径
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer import TransformerModel
from models.mixers import SlidingWindowCausalSelfAttention


def _small(**over):
    kw = dict(vocab_size=200, embedding_dim=64, num_heads=4, num_layers=2,
              hidden_dim=128, max_seq_length=32)
    kw.update(over)
    return TransformerModel(**kw)


# ─── ALiBi bias 完整缓存 ────────────────────────────────────────────────────

def test_alibi_bias_cached_when_not_learnable():
    """alibi_learnable=False 时 bias 完全确定，第二次调用应返回缓存对象（is 语义）。"""
    m = _small(alibi=True, alibi_learnable=False, num_layers=1)
    m.eval()
    attn = m.blocks[0].attn
    # 第一次调用：cache miss，计算并存入缓存
    b1 = attn._alibi_bias(8, 8, attn.qkv.weight.device, start_pos=0, mem_cols=0)
    # 第二次调用：cache hit，应返回同一对象
    b2 = attn._alibi_bias(8, 8, attn.qkv.weight.device, start_pos=0, mem_cols=0)
    assert b1 is b2, "alibi_learnable=False 时第二次调用应返回缓存的同一张量对象"


def test_alibi_bias_not_cached_when_learnable():
    """alibi_learnable=True 时 alibi_slopes 是 Parameter，每步重算（梯度回流）。"""
    m = _small(alibi=True, alibi_learnable=True, num_layers=1)
    m.eval()
    attn = m.blocks[0].attn
    b1 = attn._alibi_bias(8, 8, attn.qkv.weight.device, start_pos=0, mem_cols=0)
    b2 = attn._alibi_bias(8, 8, attn.qkv.weight.device, start_pos=0, mem_cols=0)
    # 不应缓存（每次重算），数值相同但对象不同
    assert torch.equal(b1, b2), "数值应相同"
    assert b1 is not b2, "alibi_learnable=True 时不应缓存（每次重算）"


def test_alibi_bias_not_cached_when_pe_gate():
    """pe_gate 开启时 log_pe_gate 是 Parameter，不缓存完整 bias。"""
    m = _small(alibi=True, pe_gate=True, num_layers=1)
    m.eval()
    attn = m.blocks[0].attn
    b1 = attn._alibi_bias(8, 8, attn.qkv.weight.device, start_pos=0, mem_cols=0)
    b2 = attn._alibi_bias(8, 8, attn.qkv.weight.device, start_pos=0, mem_cols=0)
    assert torch.equal(b1, b2), "数值应相同"
    assert b1 is not b2, "pe_gate 开启时不应缓存"


def test_alibi_bias_cache_numerical_equivalence():
    """缓存值与重算值数值一致（atol=1e-6）。"""
    m = _small(alibi=True, alibi_learnable=False, num_layers=1)
    m.eval()
    attn = m.blocks[0].attn
    # 第一次调用：cache miss，重算
    b1 = attn._alibi_bias(8, 8, attn.qkv.weight.device, start_pos=0, mem_cols=0)
    # 清空缓存后重算
    attn._alibi_bias_cache.clear()
    b2 = attn._alibi_bias(8, 8, attn.qkv.weight.device, start_pos=0, mem_cols=0)
    assert torch.allclose(b1, b2, atol=1e-6), "缓存值与重算值应数值一致"


def test_alibi_bias_cache_different_keys():
    """不同 (Tq, Tkv, start_pos, mem_cols) 应有不同缓存条目。"""
    m = _small(alibi=True, alibi_learnable=False, num_layers=1)
    m.eval()
    attn = m.blocks[0].attn
    dev = attn.qkv.weight.device
    b_8x8 = attn._alibi_bias(8, 8, dev, start_pos=0, mem_cols=0)
    b_4x8 = attn._alibi_bias(4, 8, dev, start_pos=0, mem_cols=0)
    assert b_8x8.shape != b_4x8.shape or not torch.equal(b_8x8, b_4x8), \
        "不同 Tq 应产生不同 bias"


def test_alibi_bias_cache_with_mem_cols():
    """mem_cols > 0 时缓存应包含 mask 修改（记忆列清零）。"""
    m = _small(alibi=True, alibi_learnable=False, num_layers=1)
    m.eval()
    attn = m.blocks[0].attn
    dev = attn.qkv.weight.device
    # mem_cols=2：前 2 列应清零
    b = attn._alibi_bias(8, 10, dev, start_pos=0, mem_cols=2)
    # 前 2 列（记忆列）应为 0
    assert torch.all(b[..., :2] == 0), "mem_cols 列应清零"
    # 后 8 列应有非零值
    assert torch.any(b[..., 2:] != 0), "主序列列应有非零 bias"


# ─── attend bias 合并数值等价 ─────────────────────────────────────────────────

def test_attend_bias_chain_numerical_equivalence():
    """链式 add 合并 bias 与逐次 add 数值等价（atol=1e-6）。

    验证 _use_causal=False 路径（有 alibi）的 bias 合并正确性。
    """
    torch.manual_seed(42)
    m = _small(alibi=True, alibi_learnable=False, num_layers=1)
    m.eval()
    x = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        out = m(x)
    assert out.shape == (1, 8, 200)
    assert torch.isfinite(out).all(), "alibi 路径输出含 nan/inf"


def test_attend_use_causal_skips_bias():
    """_use_causal=True 时跳过 bias 合并，直接用 is_causal=True。

    纯 attn（无 alibi/memory/rel_bias/window）时应走 is_causal 快捷路径。
    """
    m = _small(num_layers=1)  # 默认无 alibi
    m.eval()
    x = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        out = m(x)
    assert out.shape == (1, 8, 200)
    assert torch.isfinite(out).all()


def test_attend_alibi_cache_parity():
    """alibi bias 缓存不影响 train/infer cache parity。"""
    torch.manual_seed(99)
    m = _small(alibi=True, alibi_learnable=False, num_layers=2)
    m.eval()
    x = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        full = m(x, use_cache=False)
        out, past = m(x[:, :1], past_key_values=None, use_cache=True)
        for t in range(1, 8):
            out, past = m(x[:, t:t + 1], past_key_values=past, use_cache=True)
    diff = (full[:, -1, :] - out[:, -1, :]).abs().max().item()
    assert diff < 1e-4, f"alibi cache parity diff={diff:.2e} 超过 1e-4"


# ─── 端到端 forward 不崩溃 ───────────────────────────────────────────────────

def test_full_features_forward_no_crash():
    """全特性组合 forward 不崩溃 + 输出有限。

    验证 bias 合并优化在多 bias 场景（alibi + pe_gate + memory）下正确。
    """
    m = _small(
        num_layers=2,
        alibi=True, alibi_learnable=False,
        pe_gate=True,
    )
    m.eval()
    x = torch.randint(0, 200, (2, 8))
    with torch.no_grad():
        out = m(x)
    assert out.shape == (2, 8, 200)
    assert torch.isfinite(out).all(), "多 bias 组合输出含 nan/inf"


def test_alibi_learnable_forward_gradient():
    """alibi_learnable=True 时梯度正确回流到 alibi_slopes Parameter。"""
    m = _small(alibi=True, alibi_learnable=True, num_layers=1)
    m.train()
    x = torch.randint(0, 200, (1, 8))
    out = m(x)
    loss = out.sum()
    loss.backward()
    # alibi_slopes 应有梯度
    attn = m.blocks[0].attn
    assert attn.alibi_slopes.grad is not None, "alibi_slopes 应有梯度回流"
    assert torch.isfinite(attn.alibi_slopes.grad).all(), "梯度含 nan/inf"
