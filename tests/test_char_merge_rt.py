"""10-19：char_merge 运行时开关（B 类，可进 enhancement_schedule 交替，r51 用）。

关掉 = 整层旁路（等于没开 char_merge 的模型）；True / 不给该键时保持原行为。
"""
import torch

from models.transformer import TransformerModel
from tests.test_value_relative_safe_pow import _small


def _pair():
    torch.manual_seed(0)
    on = _small(char_merge=True)
    off = _small(char_merge=False)
    off.load_state_dict({k: v for k, v in on.state_dict().items() if not k.startswith('char_merge.')})
    on.eval(); off.eval()
    return on, off, torch.randint(0, 200, (2, 32))


def test_key_in_enhancement_keys():
    assert 'char_merge' in TransformerModel.ENHANCEMENT_KEYS


def test_rt_off_equals_no_char_merge_and_no_grad():
    on, off, x = _pair()
    ref = off(x)
    on.set_enhancements_active({'char_merge': False})
    y = on(x)
    assert torch.allclose(y, ref, atol=1e-6)
    y.float().pow(2).mean().backward()
    assert all(p.grad is None for p in on.char_merge.parameters())


def test_rt_on_restored_by_true_and_dict_without_key():
    on, off, x = _pair()
    base = on(x)
    assert not torch.allclose(base, off(x), atol=1e-4)  # 开着时确实走了这一层
    on.set_enhancements_active({'char_merge': False})
    on.set_enhancements_active({'qk_norm': True})  # 不含该键：保持关
    assert torch.allclose(on(x), off(x), atol=1e-6)
    on.set_enhancements_active(True)
    assert torch.allclose(on(x), base, atol=1e-6)
    on.set_enhancements_active(False)  # 全关也关它
    assert on._rt_char_merge is False


def test_rt_key_harmless_without_char_merge():
    m = _small(char_merge=False)
    m.set_enhancements_active({'char_merge': True})
    assert m._rt_char_merge is False
