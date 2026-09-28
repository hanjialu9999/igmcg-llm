# -*- coding: utf-8 -*-
"""M11：validate 按 token 加权（best/early-stop 用新口径，旧口径保留可对比）。

背景：validate() 原本把每个 batch 的 per-token 平均 CE 再按 batch **等权**平均。
当 batch 的有效 token 数不等（变长 pad、epoch 末尾短批）时，短批被过度代表，
得到的不是全数据集逐 token均值 —— best/early-stop 依据因此有偏。

修法（用户拍板）：一次前向同时累出两个口径
  legacy   = Σ(batch CE) / batch 数            （历史曲线，逐点可比）
  weighted = Σ(CE × n_valid) / Σ n_valid       （真·逐 token 加权，判 best/停用）
两口径都进日志（"双记录"），不改任何模型前向数值，无需重训。
"""
import inspect

import pytest
import torch

from scripts.train import validate


class _FixedLoss:
    """替身 criterion：按调用次序返回预设 CE，忽略 logits 真实内容。"""

    def __init__(self, losses, ignore_index=-100):
        self.losses = list(losses)
        self.ignore_index = ignore_index
        self.calls = 0

    def __call__(self, logits, targets):
        v = self.losses[self.calls]
        self.calls += 1
        return torch.tensor(v, dtype=torch.float32)


class _FakeModel:
    """validate 只碰 eval()/set_enhancements_active()/controller_enabled。"""

    controller_enabled = False

    def eval(self):
        pass

    def set_enhancements_active(self, active):
        self.active = active

    def __call__(self, input_ids):
        # 假 logits：替身 criterion 不用它，形状对即可
        return torch.zeros(input_ids.size(0), input_ids.size(1), 7)


def _batch(n_valid, n_total=None, ignore=-100):
    """构造 target_ids：前 n_valid 个有效，其余 n_total-n_valid 个是 pad。"""
    n_total = n_valid if n_total is None else n_total
    tgt = torch.full((1, n_total), ignore, dtype=torch.long)
    tgt[0, :n_valid] = 3
    return {'input_ids': torch.zeros(1, n_total, dtype=torch.long), 'target_ids': tgt}


def test_m11_returns_both_calibers_when_token_counts_differ():
    """10 token 批 (CE=2.0) 与 2 token 批 (CE=3.0)：
    旧口径按 batch 等权 → 2.5；新口径按 token 加权 → (20+6)/12 ≈ 2.1667。
    """
    ds = [_batch(10), _batch(2)]
    legacy, weighted = validate(_FakeModel(), ds, _FixedLoss([2.0, 3.0]),
                                torch.device('cpu'))
    assert legacy == pytest.approx(2.5)
    assert weighted == pytest.approx(26.0 / 12.0)
    # 两个口径必须不同，否则"加权"没起作用
    assert legacy != pytest.approx(weighted)


def test_m11_both_calibers_agree_when_token_counts_equal():
    """token 数相等时两口径相等（回归：证明新口径在等权退化下不引入偏差）。"""
    ds = [_batch(4), _batch(4), _batch(4)]
    legacy, weighted = validate(_FakeModel(), ds, _FixedLoss([1.0, 2.0, 3.0]),
                                torch.device('cpu'))
    assert legacy == pytest.approx(2.0)
    assert weighted == pytest.approx(2.0)


def test_m11_all_pad_batch_does_not_contribute_weight():
    """全 pad 批（n_valid=0）不进新口径的分母，也不把 0 乘进去污染分子。"""
    ds = [_batch(10), _batch(0, n_total=6)]
    legacy, weighted = validate(_FakeModel(), ds, _FixedLoss([4.0, 99.0]),
                                torch.device('cpu'))
    assert legacy == pytest.approx((4.0 + 99.0) / 2)   # 旧口径仍计入（历史行为不变）
    assert weighted == pytest.approx(4.0)               # 新口径只认有效 token


def test_m11_validate_returns_tuple_of_two_floats():
    ds = [_batch(3)]
    out = validate(_FakeModel(), ds, _FixedLoss([1.5]), torch.device('cpu'))
    assert isinstance(out, tuple) and len(out) == 2
    assert all(isinstance(v, float) for v in out)


def test_m11_best_early_stop_uses_token_weighted_caliber():
    """main() 里 epoch_loss 必须取 val_loss_tok（旧口径 val_loss 只记录）。"""
    src = inspect.getsource(__import__('scripts.train', fromlist=['main']).main)
    assert 'epoch_loss = val_loss_tok' in src, (
        'M11：best/early-stop 依据必须是逐 token 加权 val loss')
    assert 'val_loss if val_loss is not None' not in src, (
        'M11：旧口径 val_loss 不应再作为 best/early-stop 依据')


def test_m11_both_calibers_are_logged():
    """双记录：同一个 print 里必须同时出现旧口径和 token 加权口径。"""
    src = inspect.getsource(__import__('scripts.train', fromlist=['main']).main)
    assert 'Val Loss: {val_loss:.4f}' in src
    assert 'Val(token加权): {val_loss_tok:.4f}' in src
