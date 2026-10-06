"""H3 修复（10-06）：alibi_mem_offset=True 时 ALiBi 距离按真实 token 位置算。

KV 布局 [mk | pk | cur]，真实 token j 的列号是 mem_cols + j；旧行为（开关关，r42/r43）直接用列号算距离，
位置先验整体平移 mem_cols。全量 / 增量两条路径同公式，parity 测不出，这里直接比 bias。
"""

import torch

try:
    import torch_directml  # noqa: F401  与 test_round42 同：提前注册 DML 后端
except Exception:
    pass

from models.mixers import SlidingWindowCausalSelfAttention
from models.model_config import AttnConfig, ModelConfig
from models.transformer import TransformerModel


def _attn(offset, learnable=False):
    return SlidingWindowCausalSelfAttention(32, 4, alibi=True, alibi_learnable=learnable,
                                            alibi_mem_offset=offset)


def _ref_bias(a, Tq, Tkv, start_pos, mem_cols, k_off):
    qpos = torch.arange(start_pos, start_pos + Tq).unsqueeze(1)
    kpos = torch.arange(Tkv).unsqueeze(0) - k_off
    b = -a.alibi_slopes.detach().view(1, -1, 1, 1) * (qpos - kpos).abs()
    b[..., :mem_cols] = 0
    return b


def test_flag_default_off_and_plumbed():
    assert AttnConfig().alibi_mem_offset is False
    base = dict(vocab_size=50, embedding_dim=64, num_heads=4, num_layers=2, hidden_dim=128,
                max_seq_length=32, alibi=True)
    assert ModelConfig.from_dict(base).attn.alibi_mem_offset is False
    m = TransformerModel.from_config(ModelConfig.from_dict(dict(base, alibi_mem_offset=True)))
    assert all(blk.attn.alibi_mem_offset is True for blk in m.blocks)


def test_h3_warning_only_when_off():
    import warnings
    base = dict(vocab_size=50, embedding_dim=64, num_heads=4, num_layers=2, hidden_dim=128,
                max_seq_length=32, alibi=True, controller=True, controller_mem_slots=4,
                controller_memory_compress=True)
    for offset, n in ((False, 1), (True, 0)):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ModelConfig.from_dict(dict(base, alibi_mem_offset=offset))
        assert sum('H3' in str(x.message) for x in w) == n


def test_off_keeps_old_column_distance():
    for learnable in (False, True):
        a = _attn(False, learnable)
        for Tq, Tkv, sp, M in ((8, 12, 0, 4), (1, 10, 5, 4)):
            got = a._alibi_bias(Tq, Tkv, torch.device('cpu'), sp, mem_cols=M).detach()
            assert torch.equal(got, _ref_bias(a, Tq, Tkv, sp, M, 0))


def test_on_matches_memoryless_distance():
    """开：token 列的 bias 与无记忆时完全相同，记忆列为 0；训练（Tq=T）和增量（Tq=1）都成立。"""
    for learnable in (False, True):
        a = _attn(True, learnable)
        M = 4
        for Tq, T, sp in ((8, 8, 0), (1, 6, 5)):
            got = a._alibi_bias(Tq, M + T, torch.device('cpu'), sp, mem_cols=M).detach()
            plain = a._alibi_bias(Tq, T, torch.device('cpu'), sp, mem_cols=0).detach()
            assert torch.equal(got[..., M:], plain)
            assert torch.equal(got[..., :M], torch.zeros_like(got[..., :M]))
            assert torch.equal(got, _ref_bias(a, Tq, M + T, sp, M, M))


def test_on_off_share_no_cache_entry():
    """同一实例上 dist 缓存按偏移区分，开关切换不会读到另一种距离。"""
    a = _attn(False)
    old = a._alibi_bias(8, 12, torch.device('cpu'), 0, mem_cols=4).clone()
    a.alibi_mem_offset = True
    a._alibi_bias_cache.clear()
    new = a._alibi_bias(8, 12, torch.device('cpu'), 0, mem_cols=4)
    assert not torch.equal(old, new)


def _build(offset):
    cfg = ModelConfig.from_dict(dict(
        vocab_size=50, embedding_dim=64, num_heads=4, num_layers=3, hidden_dim=128, max_seq_length=32,
        gradient_checkpointing=False, alibi=True, alibi_learnable=True, alibi_mem_offset=offset,
        controller=True, controller_mem_slots=4, controller_memory_compress=True,
        controller_causal_memory=True))
    torch.manual_seed(0)
    m = TransformerModel.from_config(cfg)
    m.eval()
    return m


def _incremental(m, x, prompt=3):
    y, past = m(x[:, :prompt], use_cache=True)
    ys = [y]
    for t in range(prompt, x.size(1)):
        y, past = m(x[:, t:t + 1], past_key_values=past, use_cache=True)
        ys.append(y)
    return torch.cat(ys, dim=1)


def test_model_with_controller_memory():
    """r43 式 Controller 记忆 + ALiBi：开关真的改变输出，且开后全量 / 增量一致。"""
    off, on = _build(False), _build(True)
    on.load_state_dict(off.state_dict())
    x = torch.randint(0, 50, (2, 8))
    with torch.no_grad():
        y_off, y_on = off(x), on(x)
        assert (y_off - y_on).abs().max().item() > 1e-4
        diff = (y_on - _incremental(on, x)).abs().max().item()
    assert diff < 1e-4, f"alibi_mem_offset 增量 parity diff={diff}"
