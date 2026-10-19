"""10-19：Controller FiLM 单路运行时开关（B 类，可进 enhancement_schedule 交替，r52 用）。

关掉 = 只丢 FiLM 信号（等于 controller_film=False 的模型），direction / mem_kv 照常；
True / 不给该键时保持原行为。
"""
import torch

from models.transformer import TransformerModel
from tests.test_value_relative_safe_pow import _small


def _pair():
    torch.manual_seed(0)
    on = _small(controller=True, controller_layers=1)
    off = _small(controller=True, controller_layers=1, controller_film=False)
    with torch.no_grad():  # film_projs 零初始化 = 恒等，先扰动才测得出差别
        for p in on.controller.film_projs.parameters():
            p.normal_(0, 0.1)
    off.load_state_dict({k: v for k, v in on.state_dict().items() if 'film_projs' not in k})
    on.eval(); off.eval()
    return on, off, torch.randint(0, 200, (2, 32))


def test_key_in_enhancement_keys():
    assert 'controller_film' in TransformerModel.ENHANCEMENT_KEYS


def test_rt_off_equals_no_film_and_no_grad():
    on, off, x = _pair()
    ref = off(x)
    on.set_enhancements_active({'controller_film': False})
    y = on(x)
    assert torch.allclose(y, ref, atol=1e-6)
    y.float().pow(2).mean().backward()
    assert all(p.grad is None for p in on.controller.film_projs.parameters())
    # Controller 本体照常跑（direction / mem_kv 两路还在图里）
    assert any(p.grad is not None for n, p in on.controller.named_parameters()
               if 'film_projs' not in n)


def test_rt_on_restored_by_true_and_dict_without_key():
    on, off, x = _pair()
    base = on(x)
    assert not torch.allclose(base, off(x), atol=1e-4)  # 开着时 FiLM 确实起作用
    on.set_enhancements_active({'controller_film': False})
    on.set_enhancements_active({'qk_norm': True})  # 不含该键：保持关
    assert torch.allclose(on(x), off(x), atol=1e-6)
    on.set_enhancements_active(True)
    assert torch.allclose(on(x), base, atol=1e-6)
    on.set_enhancements_active(False)  # 全关也关它
    assert on._rt_controller_film is False


def test_rt_key_harmless_without_controller_or_film():
    for kw in ({}, {'controller': True, 'controller_layers': 1, 'controller_film': False}):
        m = _small(**kw)
        m.set_enhancements_active({'controller_film': True})
        assert m._rt_controller_film is False
