"""R40 回归测试：value_relative_safe_pow 开关（修 DirectML `base ** exp` broadcast bug）。

覆盖：
- _vrc_decay_kernel 纯连乘构造与旧 pow 路径在 CPU 上等价（λ=0 逐位、λ≠0 ~1e-9）
- λ=0 时 d(kernel.sum)/dλ = 1.0（旧 pow 在 DML 上得 0.0，自锁）
- 开关 plumbing：AttnConfig / ModelConfig.from_dict / TransformerModel 默认全为 False
- 真模型（4 层、T=64）DML vs CPU 前向：开修复 ≤1e-5，关修复 >1e-2（bug 复现 + 变异守卫）
- DML 上 kernel 与梯度与 CPU 逐位一致

无 DirectML 的机器自动跳过标记 _NEEDS_DML 的用例（本机 DML 检测走
torch_directml.is_available，与 models/device.py 同源）。
"""
import pytest
import torch

from models.mixers import _vrc_decay_kernel
from models.transformer import TransformerModel
from models.model_config import ModelConfig, AttnConfig


def _dml_device():
    """返回可用 DirectML 设备，否则 None（不打印日志，供 skipif 用）。"""
    try:
        import torch_directml
    except Exception:
        return None
    try:
        if getattr(torch_directml, 'is_available', lambda: False)():
            return torch_directml.device()
    except Exception:
        return None
    return None


_DML = _dml_device()
_NEEDS_DML = pytest.mark.skipif(_DML is None, reason='需要可用的 DirectML 设备')
_LAMS = (0.0, 0.3, 1.0, -0.7)
_KLENS = (1, 8, 64)


def _small(**over):
    kw = dict(vocab_size=200, embedding_dim=64, num_heads=4, num_layers=4,
              hidden_dim=128, max_seq_length=64)
    kw.update(over)
    return TransformerModel(**kw)


def _old_pow_kernel(lam: torch.Tensor, klen: int) -> torch.Tensor:
    """旧实现（r42 路径）：view(1,1,1,1) ** arange，留作对照。"""
    return (lam.view(1, 1, 1, 1) ** torch.arange(klen - 1, -1, -1, device=lam.device)).reshape(-1)


def _set_lambda(model, value):
    for blk in model.blocks:
        if hasattr(blk.attn, 'value_rel_lambda'):
            with torch.no_grad():
                blk.attn.value_rel_lambda.fill_(value)


# ---------------------------------------------------------------------------
# 内核等价（CPU，永远跑）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('lam', _LAMS)
@pytest.mark.parametrize('klen', _KLENS)
def test_safe_kernel_close_to_old_pow_on_cpu(lam, klen):
    """CPU 上新旧构造等价：λ=0 逐位一致，λ≠0 差 <1e-6（仅舍入次序不同）。"""
    lc = torch.tensor(lam, dtype=torch.float32)
    new = _vrc_decay_kernel(lc, klen)
    old = _old_pow_kernel(lc, klen)
    assert new.shape == (klen,)
    assert torch.isfinite(new).all(), 'kernel 出现 nan/inf'
    if lam == 0.0:
        assert torch.equal(new, old), 'λ=0 时新旧 kernel 必须逐位一致'
        assert torch.equal(new, torch.tensor([0.0] * (klen - 1) + [1.0])), \
            'λ=0 的核应为 [0,…,0,1]（恒等卷积）'
    else:
        assert float((new - old).abs().max()) < 1e-6


@pytest.mark.parametrize('lam', (0.0, 0.3, 1.0, -0.7))
def test_safe_kernel_grad_dml_matches_cpu(lam):
    """λ 梯度 DML 与 CPU 逐位一致；λ=0 时两侧都必须精确 = 1.0（旧路径 DML 上是 0.0，自锁）。"""
    grads = {}
    devs = {'cpu': torch.device('cpu')}
    if _DML is not None:
        devs['dml'] = _DML
    for tag, dev in devs.items():
        p = torch.tensor(lam, dtype=torch.float32, device=dev, requires_grad=True)
        _vrc_decay_kernel(p, 64).sum().backward()
        assert p.grad is not None
        grads[tag] = float(p.grad)
    if lam == 0.0:
        assert grads['cpu'] == 1.0, f'CPU λ=0 梯度应为 1.0，实际 {grads["cpu"]!r}'
        if 'dml' in grads:
            assert grads['dml'] == 1.0, f'DML λ=0 梯度应为 1.0，实际 {grads["dml"]!r}'
    if 'dml' in grads:
        assert grads['dml'] == grads['cpu'], \
            f'λ={lam} 梯度 DML {grads["dml"]!r} != CPU {grads["cpu"]!r}'


# ---------------------------------------------------------------------------
# 内核逐位（DML vs CPU）
# ---------------------------------------------------------------------------

@_NEEDS_DML
@pytest.mark.parametrize('lam', _LAMS)
@pytest.mark.parametrize('klen', _KLENS)
def test_safe_kernel_bitwise_dml_vs_cpu(lam, klen):
    """修复分支在 DML 与 CPU 上逐位一致（含 λ=0）。"""
    lc = torch.tensor(lam, dtype=torch.float32)
    ld = torch.tensor(lam, dtype=torch.float32, device=_DML)
    kc = _vrc_decay_kernel(lc, klen)
    kd = _vrc_decay_kernel(ld, klen).cpu()
    assert torch.equal(kc, kd), \
        f'λ={lam} klen={klen} 新 kernel 非逐位：cpu={kc.tolist()[:4]}… dml={kd.tolist()[:4]}…'


@_NEEDS_DML
@pytest.mark.parametrize('lam', (0.0, 0.3, -0.7))
def test_old_pow_kernel_reproduces_dml_broadcast_bug(lam):
    """旧 `view(1,1,1,1) ** arange` 在 DML 与 CPU 不等（bug 复现，修复的对照证据）。

    不含 λ=1.0：此时恒常数解恰好等于真值 [1,…,1]，无法区分对错。"""
    klen = 64
    kc = _old_pow_kernel(torch.tensor(lam, dtype=torch.float32), klen)
    kd = _old_pow_kernel(torch.tensor(lam, dtype=torch.float32, device=_DML), klen).cpu()
    assert not torch.equal(kc, kd), \
        f'λ={lam} 旧 kernel 竟然 DML/CPU 一致——DirectML pow bug 可能已被上游修掉，' \
        f'需重新评估是否继续保留 value_relative_safe_pow 开关'


# ---------------------------------------------------------------------------
# 开关 plumbing（默认关，r42 行为字节不变）
# ---------------------------------------------------------------------------

def test_flag_default_off_everywhere():
    """AttnConfig / ModelConfig.from_dict / TransformerModel 三级默认全为 False。"""
    assert AttnConfig().value_relative_safe_pow is False
    base = {'vocab_size': 50, 'embedding_dim': 64, 'num_heads': 4,
            'num_layers': 3, 'hidden_dim': 128, 'max_seq_length': 32}
    assert ModelConfig.from_dict(dict(base)).attn.value_relative_safe_pow is False
    assert ModelConfig.from_dict(dict(base, value_relative_safe_pow=True)).attn.value_relative_safe_pow is True
    m = _small(value_relative_coding=True)
    assert m.blocks[0].attn.value_relative_safe_pow is False


def test_flag_plumbed_to_mixer():
    """value_relative_safe_pow=True 能一路传到每层 mixer。"""
    m = _small(value_relative_coding=True, value_relative_safe_pow=True, num_layers=3)
    for blk in m.blocks:
        assert blk.attn.value_relative_safe_pow is True
    assert hasattr(m.blocks[0].attn, 'value_rel_lambda'), \
        '开关不能影响 value_relative_coding 的参数创建'


# ---------------------------------------------------------------------------
# 真模型前向（4 层、T=64）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('lam', (0.0, 0.3, -0.7))
def test_model_cpu_safe_on_vs_off_equivalent(lam):
    """CPU 上开关开/关的前向差 <1e-5（新构造数学等价，不改语义）。"""
    torch.manual_seed(0)
    a = _small(value_relative_coding=True, value_relative_safe_pow=False)
    b = _small(value_relative_coding=True, value_relative_safe_pow=True)
    b.load_state_dict(a.state_dict())
    _set_lambda(a, lam)
    _set_lambda(b, lam)
    a.eval(); b.eval()
    x = torch.randint(0, 200, (2, 64))
    with torch.no_grad():
        d = float((a(x) - b(x)).abs().max())
    assert d < 1e-5, f'λ={lam} CPU 上开关开/关前向差 {d:.3e} 超 1e-5'


@_NEEDS_DML
@pytest.mark.parametrize('lam', (0.0, 0.3, -0.7))
def test_model_dml_vs_cpu_safe_on_close(lam):
    """开修复时真模型 DML 与 CPU 前向差 <1e-5（变异守卫：回退到 pow 会退化到 ~0.2）。"""
    torch.manual_seed(0)
    m = _small(value_relative_coding=True, value_relative_safe_pow=True)
    _set_lambda(m, lam)
    m.eval()
    x = torch.randint(0, 200, (2, 64))
    with torch.no_grad():
        oc = m(x)
    m.to(_DML)
    with torch.no_grad():
        od = m(x.to(_DML)).cpu()
    d = float((oc - od).abs().max())
    assert d < 1e-5, f'λ={lam} 开修复后 DML vs CPU 前向差 {d:.6e} 应 <1e-5'


@_NEEDS_DML
@pytest.mark.parametrize('lam', (0.0, 0.3, -0.7))
def test_model_dml_vs_cpu_safe_off_diverges(lam):
    """关修复时 DML 与 CPU 前向差 >1e-2（证明 bug 存在，且反向守住「修复必须有效」）。"""
    torch.manual_seed(0)
    m = _small(value_relative_coding=True, value_relative_safe_pow=False)
    _set_lambda(m, lam)
    m.eval()
    x = torch.randint(0, 200, (2, 64))
    with torch.no_grad():
        oc = m(x)
    m.to(_DML)
    with torch.no_grad():
        od = m(x.to(_DML)).cpu()
    d = float((oc - od).abs().max())
    assert d > 1e-2, f'λ={lam} 关修复时 DML vs CPU 差仅 {d:.3e}，bug 复现失败（DML pow 可能已修）'
