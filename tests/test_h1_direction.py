"""H1 修复（10-05）：controller_direction_causal=True 时 direction 改逐位置前缀滚动均值。

旧行为（开关关，r42）：x.mean(dim=1) 整段均值——训练看到未来 token，增量第 2 步起只覆盖 1 token，
见 test_round42.py 的 xfail（test_r42_direction_incremental_parity_known_gap），那条保留登记旧行为。
"""

import torch
import torch.nn as nn

try:
    import torch_directml  # noqa: F401  与 test_round42 同：提前注册 DML 后端
except Exception:
    pass

from models.model_config import ModelConfig
from models.transformer import TransformerModel


def _build(causal=True, **kw):
    cfg = ModelConfig(vocab_size=50, embedding_dim=64, num_heads=4, num_layers=3, hidden_dim=128,
                      max_seq_length=32, controller=True, gradient_checkpointing=False,
                      controller_direction_causal=causal, **kw)
    m = TransformerModel.from_config(cfg)
    m.eval()
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():  # 放开 direction / film 投影（零初始化下测不出 direction 分量）
        c = m.controller
        c.direction_proj.weight.copy_(torch.randn(c.direction_proj.weight.shape, generator=g) * 0.1)
        for proj in c.film_projs:
            if isinstance(proj, nn.Linear):
                proj.weight.copy_(torch.randn(proj.weight.shape, generator=g) * 0.05)
    return m


def _incremental(m, x, prompt=3):
    y, past = m(x[:, :prompt], use_cache=True)
    ys = [y]
    for t in range(prompt, x.size(1)):
        y, past = m(x[:, t:t + 1], past_key_values=past, use_cache=True)
        ys.append(y)
    return torch.cat(ys, dim=1)


def test_h1_causal_full_vs_incremental_parity():
    torch.manual_seed(0)
    m = _build()
    x = torch.randint(0, 50, (2, 8))
    with torch.no_grad():
        diff = (m(x) - _incremental(m, x)).abs().max().item()
    assert diff < 1e-4, f"direction_causal 增量 parity diff={diff}"


def test_h1_causal_no_future_leak():
    torch.manual_seed(0)
    m = _build()
    x = torch.randint(0, 50, (2, 10))
    x2 = x.clone()
    x2[:, 6:] = (x2[:, 6:] + 7) % 50  # 只改位置 6 之后
    with torch.no_grad():
        d = (m(x)[:, :6] - m(x2)[:, :6]).abs().max().item()
    assert d < 1e-5, f"前 6 位 logits 随未来 token 变化 {d}"


def test_h1_old_behaviour_leaks_future():
    """对照：开关关时同一构造确实漏未来（证明上面的测试测得出 direction 分量）。"""
    torch.manual_seed(0)
    m = _build(causal=False)
    x = torch.randint(0, 50, (2, 10))
    x2 = x.clone()
    x2[:, 6:] = (x2[:, 6:] + 7) % 50
    with torch.no_grad():
        d = (m(x)[:, :6] - m(x2)[:, :6]).abs().max().item()
    assert d > 1e-4


def test_h1_direction_shapes_and_cache_tail():
    m = _build()
    c = m.controller
    x = torch.randint(0, 50, (2, 5))
    with torch.no_grad():
        sig, pres = c(x, use_cache=True)
        assert sig.direction.shape == (2, 5, 64)
        assert len(pres) == c.ctrl_layers + 1 and pres[-1][1] == 5
        sig2, pres2 = c(x[:, :1], past_kv=pres, use_cache=True, start_pos=5)
        assert sig2.direction.shape == (2, 1, 64) and pres2[-1][1] == 6
    m_old = _build(causal=False)
    with torch.no_grad():
        sig, pres = m_old.controller(x, use_cache=True)
    assert sig.direction.shape == (2, 64) and len(pres) == m_old.controller.ctrl_layers
