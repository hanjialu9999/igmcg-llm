"""10-05：value_relative_matmul 开关（VRC 因果卷积改下三角 Toeplitz matmul，免重训）。

同一个核展开成矩阵，开/关前向与梯度必须等价；默认关保 r42 旧路径。
"""
import pytest
import torch

from models.model_config import AttnConfig, ModelConfig
from tests.test_value_relative_safe_pow import _NEEDS_DML, _DML, _small, _set_lambda


def test_flag_default_off_and_plumbed():
    assert AttnConfig().value_relative_matmul is False
    base = dict(vocab_size=200, embedding_dim=64, num_heads=4, num_layers=2, hidden_dim=128, max_seq_length=64)
    assert ModelConfig.from_dict(dict(base, value_relative_matmul=True)).attn.value_relative_matmul is True
    assert _small(value_relative_coding=True).blocks[0].attn.value_relative_matmul is False
    m = _small(value_relative_coding=True, value_relative_matmul=True, num_layers=3)
    assert all(blk.attn.value_relative_matmul is True for blk in m.blocks)


def _on_off_diff(dev, lam, safe, T=64):
    torch.manual_seed(0)
    a = _small(value_relative_coding=True, value_relative_safe_pow=safe)
    b = _small(value_relative_coding=True, value_relative_safe_pow=safe, value_relative_matmul=True)
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 200, (2, T))
    out = []
    for m in (a, b):
        _set_lambda(m, lam)
        m.to(dev).eval()  # 关 dropout；eval 下照样求梯度
        y = m(x.to(dev))
        y.float().pow(2).mean().backward()
        lg = m.blocks[0].attn.value_rel_lambda.grad
        out.append((y.detach().cpu(), lg.detach().cpu()))
    return max(float((p - q).abs().max() / (1 + p.abs().max())) for p, q in zip(*out))


@pytest.mark.parametrize('lam', (0.0, 0.3, -0.7))
@pytest.mark.parametrize('safe', (False, True))
@pytest.mark.parametrize('T', (2, 64))
def test_cpu_matmul_on_vs_off_equivalent(lam, safe, T):
    """CPU 上开/关的输出与 λ 梯度相对差 <1e-5。"""
    d = _on_off_diff('cpu', lam, safe, T)
    assert d < 1e-5, f'λ={lam} safe={safe} T={T} 开/关相对差 {d:.3e}'


@_NEEDS_DML
@pytest.mark.parametrize('lam', (0.0, 0.3))
@pytest.mark.parametrize('safe', (False, True))
def test_dml_matmul_on_vs_off_equivalent(lam, safe):
    """DML 上同核开/关等价（含 r42/r43 的 safe_pow=False 造核）。"""
    d = _on_off_diff(_DML, lam, safe)
    assert d < 1e-4, f'λ={lam} safe={safe} DML 开/关相对差 {d:.3e}'
