"""10-06 用户定：Controller 训练默认交替。

r45 起开了 `controller` 的 config_train_8k_rN.yaml 必须带 `training.enhancement_schedule`，
且各段里 controller 既有开也有关；r42–r44 保持一直开以便复现，不在检查范围。
"""

import glob
import os
import re

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    with open(os.path.join(ROOT, 'configs', name), encoding='utf-8') as f:
        return yaml.safe_load(f)


def _controller_ratio(cfg):
    """按 scripts/train.py:780-783 的展开规则（缺省键补 True）算 controller 开的段占比。"""
    sched = cfg['training'].get('enhancement_schedule') or []
    on = [bool(seg.get('controller', True)) for seg in sched]
    return sum(on), len(on)


def test_new_controller_configs_alternate():
    names = [os.path.basename(p) for p in glob.glob(os.path.join(ROOT, 'configs', 'config_train_8k_r*.yaml'))
             if int(re.search(r'_r(\d+)(?:s\d+)?\.yaml$', p).group(1)) >= 45]  # rNsK = rN 换 init_seed
    assert 'config_train_8k_r45.yaml' in names
    for name in names:
        cfg = _load(name)
        if not cfg['model'].get('controller'):
            continue
        on, n = _controller_ratio(cfg)
        assert 0 < on < n, f'{name}：开了 controller 却没有按步交替（10-06 约定默认交替）'
        # 与 enhancement_off_prob / curriculum_anneal 互斥（scripts/train.py:792），否则会打 warn
        assert not cfg['training'].get('enhancement_off_prob') and cfg['training'].get('curriculum_anneal') is None, name


def test_r45_half_r46_third():
    assert _controller_ratio(_load('config_train_8k_r45.yaml')) == (1, 2)
    assert _controller_ratio(_load('config_train_8k_r46.yaml')) == (1, 3)
    # r44 是一直开的对照，不能被顺手改掉
    assert _load('config_train_8k_r44.yaml')['training'].get('enhancement_schedule') is None
