"""10-19：value_relative_coding 运行时开关（B 类，可进 enhancement_schedule 交替）。

关掉即恒等（v 不编码，等于没开 VRC 的模型）；True / 不给该键时保持原行为。
"""
import torch

from models.transformer import TransformerModel
from tests.test_value_relative_safe_pow import _small, _set_lambda


def _pair(lam=0.3):
    torch.manual_seed(0)
    on = _small(value_relative_coding=True)
    off = _small(value_relative_coding=False)
    off.load_state_dict({k: v for k, v in on.state_dict().items() if 'value_rel_lambda' not in k})
    _set_lambda(on, lam)
    on.eval(); off.eval()
    return on, off, torch.randint(0, 200, (2, 32))


def test_key_in_enhancement_keys():
    assert 'value_relative_coding' in TransformerModel.ENHANCEMENT_KEYS


def test_rt_off_equals_no_vrc_and_no_lambda_grad():
    on, off, x = _pair()
    ref = off(x)
    on.set_enhancements_active({'value_relative_coding': False})
    y = on(x)
    assert torch.allclose(y, ref, atol=1e-6)
    y.float().pow(2).mean().backward()
    assert on.blocks[0].attn.value_rel_lambda.grad is None


def test_rt_on_restored_by_true_and_dict_without_key():
    on, off, x = _pair()
    base = on(x)
    assert not torch.allclose(base, off(x), atol=1e-4)  # λ=0.3 时确实在编码
    on.set_enhancements_active({'value_relative_coding': False})
    on.set_enhancements_active({'qk_norm': True})  # 不含该键：保持关
    assert torch.allclose(on(x), off(x), atol=1e-6)
    on.set_enhancements_active(True)
    assert torch.allclose(on(x), base, atol=1e-6)
    on.set_enhancements_active(False)  # 全关也关它
    assert all(blk.attn._rt['value_relative_coding'] is False for blk in on.blocks)
