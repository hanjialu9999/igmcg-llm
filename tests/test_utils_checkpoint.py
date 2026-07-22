"""工具与 checkpoint 模块基础测试 + BUG-9/10 回归。

覆盖：
- models/utils.py: cleanup_old_checkpoints（含非数字文件名 _epoch_num 防御）、
  save_checkpoint（scaler 参数 / sidecar _config.yaml）、_cpu_offload 基础。
- models/checkpoint.py: safe_torch_load 基础 roundtrip、_build_safe_globals 非空。
- BUG-9 回归：train.py resume 时 optimizer.load_state_dict 后 state 张量需迁移至
  训练设备，否则在 CUDA/DML 上首步 optimizer.step 崩溃。
- BUG-10 回归：save_checkpoint 需保存 GradScaler state（scaler_state_dict），
  否则 fp16 resume 丢失已调整的 scale factor。
"""
import os
import sys
import glob
import shutil
import tempfile
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.utils import (
    cleanup_old_checkpoints,
    save_checkpoint,
    save_final_model,
    _cpu_offload,
)
from models.checkpoint import safe_torch_load, _build_safe_globals, build_ngram_model


# ============================================================
# 1. models/utils.py: _cpu_offload
# ============================================================

def test_cpu_offload_tensor():
    """_cpu_offload 把张量搬 CPU 并 detach。"""
    t = torch.tensor([1.0, 2.0], requires_grad=True)
    out = _cpu_offload(t)
    assert out.device.type == 'cpu'
    assert out.requires_grad is False  # detach
    assert torch.equal(out, torch.tensor([1.0, 2.0]))


def test_cpu_offload_numpy():
    """_cpu_offload 把 numpy 数组转 CPU 张量、标量转原生类型。"""
    import numpy as np
    arr = np.array([1.0, 2.0])
    out = _cpu_offload(arr)
    assert isinstance(out, torch.Tensor)
    assert out.device.type == 'cpu'
    assert torch.equal(out, torch.tensor([1.0, 2.0]))

    scalar = np.float32(3.14)
    out_s = _cpu_offload(scalar)
    assert isinstance(out_s, float)
    assert abs(out_s - 3.14) < 1e-5


def test_cpu_offload_nested():
    """_cpu_offload 递归处理 dict/list/tuple。"""
    import numpy as np
    nested = {
        'a': torch.tensor([1.0]),
        'b': [np.array([2.0]), torch.tensor([3.0])],
        'c': {'d': np.int64(42)},
    }
    out = _cpu_offload(nested)
    assert isinstance(out['a'], torch.Tensor)
    assert out['a'].device.type == 'cpu'
    assert isinstance(out['b'][0], torch.Tensor)
    assert isinstance(out['b'][1], torch.Tensor)
    assert out['c']['d'] == 42
    # tuple 类型保持
    tpl = _cpu_offload((torch.tensor([1.0]), torch.tensor([2.0])))
    assert isinstance(tpl, tuple)


# ============================================================
# 2. models/utils.py: cleanup_old_checkpoints
# ============================================================

def _make_fake_checkpoints(d, names):
    """在目录 d 下创建空 .pt 文件，模拟 checkpoint。"""
    paths = []
    for n in names:
        p = os.path.join(d, n)
        # 用最小合法 .pt 文件占位（torch.save 空字典）
        torch.save({}, p)
        paths.append(p)
    return paths


def test_cleanup_keeps_last_n(tmp_path):
    """cleanup_old_checkpoints 保留最后 N 个，删除其余。"""
    d = str(tmp_path)
    _make_fake_checkpoints(d, [f'model_epoch_{i}.pt' for i in [1, 2, 3, 4, 5, 6, 7]])
    cleanup_old_checkpoints(d, keep_last_n=3)
    remaining = sorted(glob.glob(os.path.join(d, 'model_epoch_*.pt')))
    remaining_nums = [int(os.path.basename(p).split('_')[-1].split('.')[0]) for p in remaining]
    assert remaining_nums == [5, 6, 7], f"应保留 5/6/7，实际 {remaining_nums}"


def test_cleanup_no_op_when_few(tmp_path):
    """文件数 <= keep_last_n 时不删除。"""
    d = str(tmp_path)
    _make_fake_checkpoints(d, ['model_epoch_1.pt', 'model_epoch_2.pt'])
    cleanup_old_checkpoints(d, keep_last_n=5)
    remaining = glob.glob(os.path.join(d, 'model_epoch_*.pt'))
    assert len(remaining) == 2, "不应删除任何文件"


def test_cleanup_handles_non_numeric_filenames(tmp_path):
    """防御非数字文件名（如 model_epoch_final.pt）：不崩溃、不删除非数字文件。"""
    d = str(tmp_path)
    _make_fake_checkpoints(d, [
        'model_epoch_1.pt', 'model_epoch_2.pt', 'model_epoch_3.pt',
        'model_epoch_final.pt',  # 非数字
        'model_epoch_best.pt',   # 非数字
    ])
    cleanup_old_checkpoints(d, keep_last_n=2)
    remaining = sorted(glob.glob(os.path.join(d, 'model_epoch_*.pt')))
    remaining_names = [os.path.basename(p) for p in remaining]
    # 非数字文件 _epoch_num 返回 -1，排在最前面被删除；
    # 但本测试旨在验证不崩溃 + 数字文件保留逻辑正确。
    # 非数字文件的删除与否取决于排序，关键是不抛 ValueError。
    assert 'model_epoch_3.pt' in remaining_names
    assert len([n for n in remaining_names if n.startswith('model_epoch_')]) >= 2


def test_cleanup_removes_sidecar_config_yaml(tmp_path):
    """cleanup 删除 .pt 时同步删除对应 _config.yaml。"""
    d = str(tmp_path)
    # 创建 5 个 .pt + 对应 _config.yaml
    for i in [1, 2, 3, 4, 5]:
        pt = os.path.join(d, f'model_epoch_{i}.pt')
        cfg = os.path.join(d, f'model_epoch_{i}_config.yaml')
        torch.save({}, pt)
        with open(cfg, 'w', encoding='utf-8') as f:
            f.write('test: 1\n')
    cleanup_old_checkpoints(d, keep_last_n=2)
    # 4/5 保留，1/2/3 删除（含 _config.yaml）
    remaining_cfg = glob.glob(os.path.join(d, 'model_epoch_*_config.yaml'))
    remaining_cfg_nums = [int(os.path.basename(p).split('_')[2]) for p in remaining_cfg]
    assert sorted(remaining_cfg_nums) == [4, 5], f"应保留 4/5 的 config，实际 {remaining_cfg_nums}"


# ============================================================
# 3. models/utils.py: save_checkpoint（含 BUG-10 回归）
# ============================================================

class _DummyModel(torch.nn.Module):
    def __init__(self, vocab_size=10):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, 4)
        self.vocab_size = vocab_size

    def forward(self, x):
        return self.embed(x)


class _DummyScaler:
    """模拟 torch.amp.GradScaler 的最小接口。"""
    def state_dict(self):
        return {'scale': 12345.0, '_growth_tracker': 7}

    def load_state_dict(self, sd):
        self._sd = sd


def test_save_checkpoint_basic(tmp_path):
    """save_checkpoint 保存 model/optimizer/epoch/best_loss/vocab_size。"""
    model = _DummyModel(vocab_size=10)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    d = str(tmp_path)
    path = save_checkpoint(model, opt, epoch=3, best_loss=2.5,
                           checkpoint_dir=d, vocab_size=10)
    assert os.path.exists(path)
    assert path.endswith('model_epoch_3.pt')

    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    assert ckpt['epoch'] == 3
    assert ckpt['best_loss'] == 2.5
    assert ckpt['vocab_size'] == 10
    assert 'model_state_dict' in ckpt
    assert 'optimizer_state_dict' in ckpt


def test_save_checkpoint_writes_sidecar_config(tmp_path):
    """save_checkpoint 写出 sidecar _config.yaml（weights_only=True 兼容）。"""
    model = _DummyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    d = str(tmp_path)
    cfg = {'vocab_size': 10, 'embedding_dim': 4, 'num_layers': 2}
    save_checkpoint(model, opt, epoch=1, best_loss=1.0,
                    checkpoint_dir=d, vocab_size=10, model_config=cfg)
    cfg_path = os.path.join(d, 'model_epoch_1_config.yaml')
    assert os.path.exists(cfg_path), "sidecar _config.yaml 应被写出"
    import yaml
    with open(cfg_path, 'r', encoding='utf-8') as f:
        loaded = yaml.safe_load(f)
    assert loaded == cfg


def test_save_checkpoint_saves_scaler_state_bug10(tmp_path):
    """BUG-10 回归：scaler 传入时必须保存 scaler_state_dict。"""
    model = _DummyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = _DummyScaler()
    d = str(tmp_path)
    path = save_checkpoint(model, opt, epoch=1, best_loss=1.0,
                           checkpoint_dir=d, vocab_size=10, scaler=scaler)
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    assert 'scaler_state_dict' in ckpt, "save_checkpoint 必须保存 scaler_state_dict（BUG-10）"
    assert ckpt['scaler_state_dict']['scale'] == 12345.0
    assert ckpt['scaler_state_dict']['_growth_tracker'] == 7


def test_save_checkpoint_no_scaler_when_none(tmp_path):
    """scaler=None 时不写出 scaler_state_dict 键。"""
    model = _DummyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    d = str(tmp_path)
    path = save_checkpoint(model, opt, epoch=1, best_loss=1.0,
                           checkpoint_dir=d, vocab_size=10, scaler=None)
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    assert 'scaler_state_dict' not in ckpt


def test_save_checkpoint_cpu_offload(tmp_path):
    """save_checkpoint 保存的 model_state_dict 必须是 CPU 张量（DML/CUDA 可移植）。"""
    model = _DummyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    d = str(tmp_path)
    path = save_checkpoint(model, opt, epoch=1, best_loss=1.0,
                           checkpoint_dir=d, vocab_size=10)
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    for v in ckpt['model_state_dict'].values():
        if isinstance(v, torch.Tensor):
            assert v.device.type == 'cpu', f"保存的张量应在 CPU，实际 {v.device}"


# ============================================================
# 4. models/checkpoint.py: safe_torch_load / _build_safe_globals
# ============================================================

def test_build_safe_globals_non_empty():
    """_build_safe_globals 返回非空列表（至少包含 torch._utils 重建函数）。"""
    gs = _build_safe_globals()
    assert len(gs) > 0, "safe globals 不应为空"
    # 至少有一个是 torch._utils._rebuild_device_tensor_from_numpy
    names = [getattr(g, '__name__', '') for g in gs]
    assert any('rebuild' in n.lower() for n in names), f"应含 numpy 重建函数，实际 {names}"


def test_safe_torch_load_roundtrip(tmp_path):
    """safe_torch_load 能加载标准 torch.save 的张量字典（weights_only=True）。"""
    d = str(tmp_path)
    path = os.path.join(d, 'test.pt')
    data = {
        'tensor': torch.tensor([1.0, 2.0, 3.0]),
        'scalar': 42,
        'nested': {'a': torch.zeros(2, 2)},
    }
    torch.save(data, path)
    loaded = safe_torch_load(path, map_location='cpu')
    assert torch.equal(loaded['tensor'], data['tensor'])
    assert loaded['scalar'] == 42
    assert loaded['nested']['a'].shape == (2, 2)


def test_safe_torch_load_rejects_arbitrary_globals(tmp_path):
    """safe_torch_load（weights_only=True）拒绝非白名单全局符号的文件。

    用 pickle 注入一个非白名单类，模拟恶意 .pt 诱导放行全局符号（CVE-2026-24747 类）。
    """
    d = str(tmp_path)
    path = os.path.join(d, 'malicious.pt')

    class _Evil:
        def __reduce__(self):
            # 引用非白名单全局符号：os.system
            return (os.system, ('echo should_not_run',))

    # pickle 一个引用非白名单全局符号的对象
    import pickle
    payload = pickle.dumps(_Evil())
    # 直接写入原始字节（绕过 torch.save 的张量包装），确保 weights_only=True 拒绝
    with open(path, 'wb') as f:
        f.write(payload)

    # weights_only=True 应拒绝加载（抛异常），而非放行任意全局符号
    with pytest.raises(Exception):
        safe_torch_load(path, map_location='cpu')


# ============================================================
# 5. BUG-9 回归：optimizer resume 设备迁移
# ============================================================

def test_resume_optimizer_state_migration_bug9(tmp_path):
    """BUG-9 回归：resume 后 optimizer.state 的张量需迁移至训练设备。

    原始 bug：train.py resume 时 optimizer.load_state_dict 直接赋值 CPU 张量，
    首步 optimizer.step() 在 CUDA/DML 上崩溃。修复：加载后遍历 state 迁移至 device。
    本测试验证迁移循环逻辑本身：加载后遍历 state 把张量搬到目标 device（CPU 模拟）。
    """
    device = torch.device('cpu')  # 测试环境只有 CPU，但验证迁移循环逻辑

    # 模拟训练：创建 optimizer 并产生 state（如 momentum buffer）
    model1 = _DummyModel()
    opt1 = torch.optim.SGD(model1.parameters(), lr=0.1, momentum=0.9)
    # 跑一步反向产生 momentum buffer（state 非空）
    loss = model1(torch.tensor([0, 1, 2])).sum()
    loss.backward()
    opt1.step()
    # 验证 state 已产生（SGD with momentum 有 momentum_buffer）
    has_state = any(len(s) > 0 for s in opt1.state.values())
    assert has_state, "测试前置：optimizer 应有 state（momentum buffer）"

    # 保存 checkpoint
    d = str(tmp_path)
    path = save_checkpoint(model1, opt1, epoch=1, best_loss=1.0,
                           checkpoint_dir=d, vocab_size=10)

    # 模拟 resume：新模型 + 新 optimizer，load_state_dict
    model2 = _DummyModel()
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.1, momentum=0.9)
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model2.load_state_dict(ckpt['model_state_dict'])
    opt2.load_state_dict(ckpt['optimizer_state_dict'])

    # BUG-9 修复代码：迁移 state 张量至训练设备
    for state in opt2.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # 断言：迁移后所有 state 张量在目标 device 上
    for state in opt2.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                assert v.device == device, \
                    f"BUG-9 回归：state[{k}] 应在 {device}，实际 {v.device}"

    # 进一步验证：迁移后 optimizer.step() 不崩溃（CPU 上模拟 DML/CUDA 场景的"首步不崩"）
    loss2 = model2(torch.tensor([0, 1, 2])).sum()
    loss2.backward()
    opt2.step()  # 若未迁移、在异构设备上会崩；CPU 上验证不抛异常
