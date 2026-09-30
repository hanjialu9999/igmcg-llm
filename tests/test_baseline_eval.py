# -*- coding: utf-8 -*-
"""质量基准脚本 scripts/baseline_eval.py 的正确性锚点。

三条口径为什么都必须测（脚本 docstring 同款结论，这里用可执行断言钉住）：
  tf          全序列前向 = validate 同算法（泄漏可见）
  prefix      逐 t 只喂前缀 = 无泄漏参照
  incremental use_cache=True 逐 token = 真实生成路径

本文件用"干净小模型"（无 controller/memory）验证两条硬性质：
  1. 因果掩码成立 → prefix == tf（若不等，说明有地方窥未来，脚本结论作废）
  2. KV 缓存正确 → incremental == tf（无 char_merge 时；若不等，说明增量循环写错了）
再用 r42 同款开关（char_merge=True）钉住已知差异。
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.config_loader import build_model
from scripts.train import validate
from scripts.baseline_eval import eval_tf, eval_prefix, eval_incremental, _finish

DEVICE = torch.device('cpu')
VOCAB = 50
IGNORE = -100


class _FixedDS(Dataset):
    """固定 token 序列（不做随机，保证三次评测喂同一份输入）。"""

    def __len__(self):
        return 6

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(1000 + i)
        x = torch.randint(0, VOCAB, (16,), generator=g)
        y = x.clone()
        # 每条样本 pad 数不同（1..6）→ 各 batch 有效 token 数不等，才测得到加权差异
        y[:i + 1] = IGNORE
        return {'input_ids': x, 'target_ids': y}


def _build(char_merge=True, controller=False, seed=7):
    torch.manual_seed(seed)
    cfg = {'model': {
        'vocab_size': VOCAB, 'embedding_dim': 32, 'num_heads': 4,
        'num_layers': 2, 'hidden_dim': 64, 'max_seq_length': 16,
        'dropout': 0, 'mixer': 'attn', 'char_merge': char_merge,
        'alibi': False, 'controller': controller,
    }}
    m = build_model(cfg, device=DEVICE)
    m.eval()
    m.set_enhancements_active(True)
    return m


def _loader():
    return DataLoader(_FixedDS(), batch_size=3, shuffle=False)


def _crit():
    return nn.CrossEntropyLoss(ignore_index=IGNORE)


def _tf(m):
    return _finish(eval_tf(m, _loader(), DEVICE, IGNORE))


def _prefix(m):
    return _finish(eval_prefix(m, _loader(), DEVICE, IGNORE))


def _inc(m):
    return _finish(eval_incremental(m, _loader(), DEVICE, IGNORE))


def test_tf_matches_validate_token_weighted_caliber():
    """eval_tf 必须等于 validate 的 M11 口径（全数据集逐 token 加权）。

    两条算法路径独立实现（一个在 train.py 的 validate，一个在 baseline_eval），
    算出同一个数 = 脚本没把 NLL/token 计数写错，也让"基准 ppl"与训练日志里的
    Val(token加权) 可直接对照。
    """
    m = _build(char_merge=True, controller=True, seed=11)
    loader = _loader()
    legacy, weighted = validate(m, loader, _crit(), DEVICE)
    tf = _tf(m)
    assert tf['loss'] == pytest.approx(weighted, abs=1e-5)
    # 样本 pad 数不同 → 旧口径（batch 等权）与逐 token 口径必不等，证明加权真生效
    assert legacy != pytest.approx(weighted, abs=1e-6)


def test_prefix_equals_tf_on_causal_model():
    """干净模型（无 controller/memory）下 prefix == tf：因果性成立，prefix 是无泄漏参照。"""
    m = _build(char_merge=True, controller=False, seed=13)
    tf = _tf(m)
    pf = _prefix(m)
    assert pf['loss'] == pytest.approx(tf['loss'], abs=1e-4), (
        f'prefix({pf["loss"]}) 与 tf({tf["loss"]}) 不等 → 有位置窥视了未来，'
        f'基准的"无泄漏参照"不成立')


def test_incremental_equals_tf_without_char_merge():
    """关掉 char_merge 后 incremental == tf：证明 KV 缓存增量循环本身没写错。"""
    m = _build(char_merge=False, controller=False, seed=17)
    tf = _tf(m)
    inc = _inc(m)
    assert inc['loss'] == pytest.approx(tf['loss'], abs=1e-4), (
        f'增量({inc["loss"]}) 与全量({tf["loss"]}) 不等 → 增量循环实现有误')


def test_char_merge_layer_has_no_incremental_state():
    """默认配置（`incremental_buffer=False`）下的结构性局限，按红线只记录不修：

    CharMergeLayer 是 kernel=3 的因果卷积，只在整段前向里拿到左侧 2 个 token；
    KV 缓存增量解码每步只喂 1 个 token → 卷积窗口退化为 [0, 0, x_t]，与整段
    前向同一位置的窗口 [x_{t-2}, x_{t-1}, x_t] 不同。因此"全量前向 ppl"与
    "增量推理 ppl"之间天然存在由该层带来的差值，读基准时必须把它和 H2 泄漏区分开。

    第四轮已给出可选修法：把 layer 切到 `incremental_buffer=True`（滚动缓冲，
    `_cm_buffer`）后逐步喂法与整段前向一致，见 `tests/test_charmerge_buffer.py`。
    本测试锁住**默认关**时的原行为，防止开关默认值被悄悄改动。
    """
    from models.layers import CharMergeLayer
    torch.manual_seed(3)
    layer = CharMergeLayer(16, kernel_size=3, dropout=0).eval()
    x = torch.randn(1, 8, 16)
    with torch.no_grad():
        full = layer(x)
        stepwise = torch.cat([layer(x[:, t:t + 1]) for t in range(8)], dim=1)
    # 位置 0：两种喂法窗口都是零左邻 → 完全一致（排除"卷积本身实现不同"）
    assert torch.allclose(full[:, 0], stepwise[:, 0], atol=1e-6)
    # 位置 1 起：整段有左邻、逐步没有 → 不一致（结构性差异，非 bug 而是缺增量状态）
    assert not torch.allclose(full[:, 1], stepwise[:, 1], atol=1e-6)


def test_finish_returns_exp_of_loss():
    acc = {'nll_sum': 4.0 * 8, 'tokens': 8}
    out = _finish(acc)
    assert out['loss'] == pytest.approx(4.0)
    assert out['ppl'] == pytest.approx(float(torch.exp(torch.tensor(4.0))), rel=1e-5)
    assert out['tokens'] == 8


def test_value_relative_safe_pow_cli_defaults_off():
    """第七轮新增开关必须默认 off（= r42 原路径逐位可比），防止被悄悄改成 on。

    同 test_char_merge_layer_has_no_incremental_state 的"锁默认"红线。
    """
    from scripts.baseline_eval import build_parser
    args = build_parser().parse_args([])
    assert args.value_relative_safe_pow == 'off'
    # 与既有三个开关一起核对，避免整组默认值被改动
    assert (args.controller, args.direction, args.char_merge_buffer) == ('on', 'on', 'off')


def test_value_relative_safe_pow_applies_to_every_mixer():
    """apply_value_relative_safe_pow 必须命中每个带该属性的子模块并可逆。"""
    from scripts.baseline_eval import apply_value_relative_safe_pow
    m = _build(char_merge=False, controller=False, seed=19)
    mix = [s for s in m.modules() if hasattr(s, 'value_relative_safe_pow')]
    assert mix, 'r42 同款 attn mixer 应带 value_relative_safe_pow'
    assert all(s.value_relative_safe_pow is False for s in mix), '代码默认必须是 False'

    assert apply_value_relative_safe_pow(m, True) == len(mix)
    assert all(s.value_relative_safe_pow is True for s in mix)

    assert apply_value_relative_safe_pow(m, False) == len(mix)
    assert all(s.value_relative_safe_pow is False for s in mix)
