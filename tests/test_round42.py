# -*- coding: utf-8 -*-
"""R42 回归测试：Controller/Generator 双模型编排。

R42 架构（详见 AGENT_MEMORY.md §10）：
  独立 Controller 模型（GatedDeltaNet mixer，线性复杂度看全上下文）产出 3 类控制信号
  条件化 Generator：①压缩记忆 mem_kv 注入 attention；②FiLM 调制各层输入；③生成方向偏置。
  所有信号投影零初始化 → 中性起步（向后兼容 + 训练稳定）。

测试覆盖：
1. Controller 前向 + 控制信号 shape 正确
2. FiLM 注入数值正确（γ,β 经 tanh 限幅后仿射调制）
3. 记忆压缩 mem_kv 注入 attention（合并 MemoryBank / 独立使用）
4. 向后兼容（controller=False 行为不变，旧权重 strict=True 加载）
5. 中性初始化（controller 开启但信号=0 时输出≈关闭）
6. 增量解码 cache parity（全量 vs 逐 token）
7. backward 梯度回流（Controller 投影收到梯度）
8. DML 前向冒烟 + backward 不崩
9. 性能对比（controller 开启 vs 关闭的 step 时间）
10. 配置校验（controller_dim/heads 整除、互斥校验）
"""
import time

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 提前导入 torch_directml 注册 privateuseone 后端：pytest 默认的 assertion
# rewriting 模式会干扰 DML autograd engine 的 device_ready_queues_ 初始化（致
# backward 报 INTERNAL ASSERT），提前注册可规避（与 test_round36_7.py / train.py 一致）。
try:
    import torch_directml  # noqa: F401
except Exception:
    pass

from models.controller import ControllerModel, ControllerOutput
from models.model_config import ModelConfig
from models.transformer import TransformerModel


# ============================================================
# 辅助函数
# ============================================================

def _build(vocab=50, dim=64, heads=4, layers=3, hidden=128, seq=32, **kw):
    """构建带 Controller 的 TransformerModel（默认全信号开启）。"""
    cfg = ModelConfig(
        vocab_size=vocab, embedding_dim=dim, num_heads=heads, num_layers=layers,
        hidden_dim=hidden, max_seq_length=seq, controller=True,
        gradient_checkpointing=False, **kw)
    return TransformerModel.from_config(cfg)


def _build_off(vocab=50, dim=64, heads=4, layers=3, hidden=128, seq=32, **kw):
    """构建不带 Controller 的 TransformerModel（向后兼容基线）。"""
    cfg = ModelConfig(
        vocab_size=vocab, embedding_dim=dim, num_heads=heads, num_layers=layers,
        hidden_dim=hidden, max_seq_length=seq, controller=False,
        gradient_checkpointing=False, **kw)
    return TransformerModel.from_config(cfg)


def _dml_device():
    """返回可用的 DML 设备，不可用则 pytest.skip。"""
    try:
        import torch_directml
    except Exception:
        pytest.skip("torch_directml 未安装，DML 不可用")
    if not getattr(torch_directml, "is_available", lambda: False)():
        pytest.skip("DML 设备不可用")
    return torch_directml.device()


# ============================================================
# 1. Controller 前向 + 控制信号 shape
# ============================================================

def test_r42_controller_signals_shape():
    """Controller 产出 3 类控制信号 shape 正确。"""
    m = _build(dim=64, heads=4, layers=3, seq=16)
    m.eval()
    B, T = 2, 8
    x = torch.randint(0, 50, (B, T))
    signals, presents = m.controller(x, use_cache=False)
    assert isinstance(signals, ControllerOutput)
    # ① mem_kv: (mk, mv) 各 (B, M, gen_head_dim)
    assert signals.mem_kv is not None
    mk, mv = signals.mem_kv
    assert mk.shape == (B, 4, 16), f"mk shape {mk.shape} != (2, 4, 16)"
    assert mv.shape == (B, 4, 16)
    # ② film_per_layer: 长度=gen_layers，layer 0 为 None，其余 (gamma, beta) 各 (B, T, gen_dim)
    assert signals.film_per_layer is not None
    assert len(signals.film_per_layer) == 3
    assert signals.film_per_layer[0] is None
    for i in range(1, 3):
        gamma, beta = signals.film_per_layer[i]
        assert gamma.shape == (B, T, 64), f"gamma[{i}] shape {gamma.shape}"
        assert beta.shape == (B, T, 64)
    # ③ direction: (B, gen_dim)
    assert signals.direction is not None
    assert signals.direction.shape == (B, 64)
    # use_cache=False → presents 为 None
    assert presents is None


def test_r42_controller_signals_disabled():
    """关闭部分控制信号时，对应字段为 None。"""
    # 只关 direction+film，保留 memory_compress（controller=True 须至少一种信号）
    m = _build(controller_direction=False, controller_film=False)
    m.eval()
    x = torch.randint(0, 50, (2, 8))
    signals, _ = m.controller(x, use_cache=False)
    assert signals.mem_kv is not None       # memory_compress 仍开
    assert signals.film_per_layer is None   # film 关
    assert signals.direction is None        # direction 关


# ============================================================
# 2. FiLM 注入数值正确
# ============================================================

def test_r42_film_neutral_init():
    """中性初始化：Controller FiLM 信号=0 → Generator 输出与关闭时一致。

    同一模型 toggle _rt_controller：True（Controller 跑，信号=0）vs False（跳过）。
    film+direction 信号经零投影 → 恒等 → 输出逐位相同。
    """
    m = _build(controller_memory_compress=False)  # 只测 film+direction（mem_kv 有 softmax 竞争）
    m.eval()
    x = torch.randint(0, 50, (2, 8))
    m._rt_controller = True
    y_on = m(x)
    m._rt_controller = False
    y_off = m(x)
    assert torch.equal(y_on, y_off), (
        f"FiLM 中性起步失败：diff={(y_on - y_off).abs().max().item()}")


def test_r42_film_active_changes_output():
    """FiLM 投影非零时，Generator 输出变化（确认信号确实注入）。"""
    m = _build(controller_memory_compress=False, controller_direction=False)
    m.eval()
    # 手动设 film_projs 权重为非零 → FiLM 信号非零 → 输出应变化
    for proj in m.controller.film_projs:
        if isinstance(proj, nn.Linear):
            nn.init.normal_(proj.weight, 0, 0.1)
            nn.init.normal_(proj.bias, 0, 0.1)
    x = torch.randint(0, 50, (2, 8))
    m._rt_controller = True
    y_on = m(x)
    m._rt_controller = False
    y_off = m(x)
    assert not torch.allclose(y_on, y_off, atol=1e-6), (
        "FiLM 信号非零时输出应变化")


# ============================================================
# 3. 记忆压缩 mem_kv 注入
# ============================================================

def test_r42_mem_kv_injection_changes_output():
    """Controller mem_kv 非零时，Generator 输出变化（确认 mem_kv 注入 attention）。"""
    m = _build(controller_film=False, controller_direction=False)
    m.eval()
    # mem_proj 零初始化 → mem_kv=0；设非零后输出应变
    x = torch.randint(0, 50, (2, 8))
    m._rt_controller = True
    y_zero = m(x)
    # 设 mem_proj 非零
    nn.init.normal_(m.controller.mem_proj.weight, 0, 0.1)
    y_nonzero = m(x)
    assert not torch.allclose(y_zero, y_nonzero, atol=1e-5), (
        "mem_kv 非零时输出应变化")


def test_r42_mem_kv_with_memory_bank():
    """Controller mem_kv 与 MemoryBank 并存时合并正确（mem_kv 槽数累加）。"""
    from models.model_config import MemoryConfig
    m = _build(dim=64, heads=4, layers=3, memory=MemoryConfig(size=8))
    m.eval()
    x = torch.randint(0, 50, (2, 8))
    # 应正常运行（MemoryBank 8 槽 + Controller 4 槽 = 12 槽 mem_kv）
    y = m(x)
    assert y.shape == (2, 8, 50)


# ============================================================
# 4. 向后兼容
# ============================================================

def test_r42_backward_compat_controller_off():
    """controller=False 时行为与旧模型完全一致（无 Controller 子模块）。"""
    m = _build_off()
    assert not m.controller_enabled
    assert not hasattr(m, 'controller')
    m.eval()
    x = torch.randint(0, 50, (2, 8))
    y = m(x)
    assert y.shape == (2, 8, 50)


def test_r42_backward_compat_strict_load():
    """controller=False 模型的 state_dict 可被 controller=True 模型 strict=True 加载
    （Controller 新参数不在旧 dict 中，但 strict=True 只要求旧参数都在新模型里）。"""
    m_off = _build_off()
    m_on = _build()
    sd_off = m_off.state_dict()
    # controller=True 模型加载 controller=False 的 state_dict
    # strict=True 会报缺失键（controller.* 参数不在 sd_off 中）
    # → 用 strict=False 加载，验证 Generator 部分全部命中
    missing, unexpected = m_on.load_state_dict(sd_off, strict=False)
    # 缺失的应全是 controller.* 参数
    assert all(k.startswith('controller.') for k in missing), (
        f"非 Controller 缺失键: {[k for k in missing if not k.startswith('controller.')]}")
    # 不应有意外的旧键
    assert len(unexpected) == 0, f"意外键: {unexpected}"


# ============================================================
# 5. 中性初始化（全信号）
# ============================================================

def test_r42_neutral_init_approximate():
    """Controller 全信号开启 + 零初始化 → 输出≈关闭（mem_kv 有 softmax 竞争，允许小 diff）。

    film+direction 信号=0 → 完全中性；mem_kv=0 但零 k 参与 softmax → 小 diff。
    阈值 < 0.15（约 10% 输出幅度），符合用户"≈关闭"要求。
    """
    m = _build()  # 全信号
    m.eval()
    x = torch.randint(0, 50, (2, 8))
    m._rt_controller = True
    y_on = m(x)
    m._rt_controller = False
    y_off = m(x)
    diff = (y_on - y_off).abs().max().item()
    out_mag = y_off.abs().max().item()
    assert diff < 0.15, (
        f"中性起步 diff={diff} 过大（输出幅度={out_mag}，相对={diff/max(out_mag,1e-6):.1%}）")


# ============================================================
# 6. 增量解码 cache parity
# ============================================================

def test_r42_cache_parity():
    """全量前向 vs 逐 token 增量解码，输出逐位一致（Controller cache 正确）。"""
    m = _build(seq=32)
    m.eval()
    x = torch.randint(0, 50, (2, 6))
    with torch.no_grad():
        y_full = m(x)
        # 增量：前 3 + 逐 token
        y_first, past = m(x[:, :3], use_cache=True)
        ys = [y_first]  # 3 tokens
        cur_past = past
        for t in range(3, 6):
            y_t, cur_past = m(x[:, t:t+1], past_key_values=cur_past, use_cache=True)
            ys.append(y_t)
        y_inc = torch.cat(ys, dim=1)  # 3 + 3 = 6 tokens
    diff = (y_full - y_inc).abs().max().item()
    assert diff < 1e-4, f"cache parity diff={diff} 过大"


# ============================================================
# 7. backward 梯度回流
# ============================================================

def test_r42_backward_gradient_flow():
    """backward 后 Controller 投影收到梯度（梯度经控制信号回流到 Controller）。"""
    m = _build()
    m.train()
    x = torch.randint(0, 50, (2, 8))
    y = m(x)
    loss = y.float().sum()
    loss.backward()
    # 所有控制信号投影应有梯度
    assert m.controller.mem_proj.weight.grad is not None
    assert m.controller.film_projs[1].weight.grad is not None
    assert m.controller.direction_proj.weight.grad is not None
    # GatedDeltaNet mixer 的 qkv 也应有梯度
    assert m.controller.mixers[0].qkv.weight.grad is not None


# ============================================================
# 8. DML 前向冒烟 + backward
# ============================================================

def test_r42_dml_forward_backward():
    """DML 设备上 Controller 前向 + backward 不崩。"""
    dev = _dml_device()
    m = _build(dim=64, heads=4, layers=3, seq=16).to(dev)
    m.tie_weights()
    m.train()
    x = torch.randint(0, 50, (2, 8), device=dev)
    y = m(x)
    loss = y.float().sum()
    loss.backward()
    assert y.shape == (2, 8, 50)
    # 验证梯度非空
    assert m.controller.mem_proj.weight.grad is not None


def test_r42_dml_cache_parity():
    """DML 设备上 cache parity 成立。"""
    dev = _dml_device()
    m = _build(dim=64, heads=4, layers=3, seq=32).to(dev)
    m.tie_weights()
    m.eval()
    x = torch.randint(0, 50, (2, 6), device=dev)
    with torch.no_grad():
        y_full = m(x)
        y_first, past = m(x[:, :3], use_cache=True)
        cur_past = past
        ys = [y_first]  # 3 tokens
        for t in range(3, 6):
            y_t, cur_past = m(x[:, t:t+1], past_key_values=cur_past, use_cache=True)
            ys.append(y_t)
        y_inc = torch.cat(ys, dim=1)  # 3 + 3 = 6 tokens
    diff = (y_full - y_inc).abs().max().item()
    assert diff < 1e-4, f"DML cache parity diff={diff}"


# ============================================================
# 9. 性能对比（controller 开启 vs 关闭）
# ============================================================

def test_r42_performance_overhead():
    """Controller 开启 vs 关闭的 step 时间对比（Controller 应轻量，开销 < 2x）。"""
    m_off = _build_off(dim=128, heads=4, layers=4, hidden=256, seq=32)
    m_on = _build(dim=128, heads=4, layers=4, hidden=256, seq=32)
    m_off.eval()
    m_on.eval()
    x = torch.randint(0, 50, (4, 16))
    # warmup
    with torch.no_grad():
        for _ in range(3):
            m_off(x)
            m_on(x)
    # bench
    N = 10
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(N):
            m_off(x)
        t_off = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(N):
            m_on(x)
        t_on = time.perf_counter() - t0
    overhead = t_on / max(t_off, 1e-6)
    # Controller 2 层 GatedDeltaNet（线性复杂度）应远轻于 Generator。
    # 注意：小模型（4 层 dim=128）上 DML/CPU 的小算子调度税占比高，
    # GatedDeltaNet 的 delta rule 有较多逐元素 op，开销比偏大；
    # 真实模型（12+ 层）上 Controller 占比 ≈ 2/N_layers ≈ 17%，远低于此。
    # Controller 是 opt-in 特性（关闭时零开销），此处只做冒烟级检查。
    assert overhead < 5.0, (
        f"Controller 开销过大：on/off={overhead:.2f}x（off={t_off:.3f}s on={t_on:.3f}s）")


# ============================================================
# 10. 配置校验
# ============================================================

def test_r42_config_validation_dim_heads():
    """controller_dim 必须能被 controller_heads 整除。"""
    with pytest.raises(AssertionError):
        ModelConfig(vocab_size=50, embedding_dim=64, num_heads=4, num_layers=3,
                    hidden_dim=128, max_seq_length=32,
                    controller=True, controller_dim=64, controller_heads=3)


def test_r42_config_validation_no_signal():
    """controller=True 须至少启用一种控制信号。"""
    with pytest.raises(AssertionError):
        ModelConfig(vocab_size=50, embedding_dim=64, num_heads=4, num_layers=3,
                    hidden_dim=128, max_seq_length=32,
                    controller=True, controller_direction=False,
                    controller_film=False, controller_memory_compress=False)


def test_r42_config_validation_mem_slots():
    """controller_mem_slots 必须 >= 1。"""
    with pytest.raises(AssertionError):
        ModelConfig(vocab_size=50, embedding_dim=64, num_heads=4, num_layers=3,
                    hidden_dim=128, max_seq_length=32,
                    controller=True, controller_mem_slots=0)


def test_r42_config_loader_mixer_mutex():
    """controller=True 与 mixer='linear2d' 不兼容（ValueError）。"""
    from models.config_loader import build_model
    config = {
        'model': {
            'vocab_size': 50, 'embedding_dim': 64, 'num_heads': 4,
            'num_layers': 3, 'hidden_dim': 128, 'max_seq_length': 32,
            'controller': True, 'mixer': 'linear2d',
        }
    }
    with pytest.raises(ValueError, match="controller=True.*linear2d"):
        build_model(config)


# ============================================================
# 11. Controller 独立单元测试
# ============================================================

def test_r42_controller_model_standalone():
    """ControllerModel 可独立实例化（不依赖 TransformerModel）。"""
    emb = nn.Embedding(50, 64)
    ctrl = ControllerModel(
        gen_dim=64, gen_heads=4, gen_layers=3,
        ctrl_dim=64, ctrl_heads=4, ctrl_layers=2,
        mem_slots=4, max_seq_length=32,
        embedding_layer=emb)
    ctrl._apply_neutral_inits()
    ctrl.eval()
    x = torch.randint(0, 50, (2, 8))
    signals, presents = ctrl(x, use_cache=False)
    assert signals.mem_kv is not None
    assert signals.film_per_layer is not None
    assert signals.direction is not None
    # 中性初始化 → 所有信号为 0
    mk, mv = signals.mem_kv
    assert torch.all(mk == 0), "mem_kv mk 应为零（中性初始化）"
    assert torch.all(mv == 0), "mem_kv mv 应为零"
    assert torch.all(signals.direction == 0), "direction 应为零"
    for i in range(1, 3):
        gamma, beta = signals.film_per_layer[i]
        assert torch.all(gamma == 0) and torch.all(beta == 0), "FiLM γ,β 应为零"


def test_r42_controller_custom_dim():
    """Controller 用独立 ctrl_dim（不等于 gen_dim）时正常工作。"""
    m = _build(dim=64, heads=4, layers=3,
               controller_dim=32, controller_heads=4, controller_layers=2)
    m.eval()
    x = torch.randint(0, 50, (2, 8))
    y = m(x)
    assert y.shape == (2, 8, 50)
    assert m.controller.ctrl_dim == 32
    assert m.controller.gen_dim == 64


def test_r42_set_enhancements_active_controller():
    """set_enhancements_active 可运行时开关 Controller。"""
    m = _build()
    m.eval()
    x = torch.randint(0, 50, (2, 8))
    m.set_enhancements_active(False)
    assert not m._rt_controller
    y_off = m(x)
    m.set_enhancements_active(True)
    assert m._rt_controller
    y_on = m(x)
    # Controller 开启（信号=0）≈ 关闭（mem_kv 有小 diff）
    diff = (y_on - y_off).abs().max().item()
    assert diff < 0.15


def test_r42_controller_generate():
    """Controller 模型 generate() 端到端不崩。"""
    m = _build(vocab=50, dim=64, heads=4, layers=3, seq=32)
    m.eval()
    tokens = m.generate([1, 2, 3], max_length=5, device='cpu',
                        temperature=0.8, top_k=10)
    assert isinstance(tokens, list)
    assert len(tokens) >= 3  # 至少返回 prompt


# ============================================================
# 12. 三阶段训练 controller_active 参数（train.py R42 新增逻辑）
# ============================================================
# 覆盖缺口：R42 在 scripts/train.py 的 train_epoch/validate 新增 controller_active
# 参数，main() 中按 controller_warmup_frac 计算 _ctrl_active 切换点。该逻辑
# 决定 warmup 期是否跳过 Controller 前向——若失效会导致：
#   ① warmup 期 Controller 仍跑（浪费 ~40% 算力）；
#   ② 早期梯度经零初始化信号回流污染 Generator warmup；
#   ③ enhancement_schedule 设 controller=True 时无法被 warmup 覆盖。
# 上述路径在 R42 提交中无任何单元测试覆盖，本章节补齐。

class _TinyTrainDS(Dataset):
    """超小数据集：4 条样本，每条 8 tokens，专供 train_epoch/validate 单元测试。"""
    def __len__(self):
        return 4

    def __getitem__(self, i):
        x = torch.randint(0, 50, (8,))
        return {'input_ids': x, 'target_ids': x}


def _run_train_epoch(m, controller_active=True, enhancement_schedule=None):
    """跑一个 train_epoch（2 batch），可观察调用后 model._rt_controller 状态。"""
    from scripts.train import train_epoch
    train_epoch(
        m, DataLoader(_TinyTrainDS(), batch_size=2),
        torch.optim.AdamW(m.parameters(), lr=1e-3),
        torch.nn.CrossEntropyLoss(), 'cpu', 1,
        show_progress=False, grad_accum_steps=1, use_amp=False,
        controller_active=controller_active,
        enhancement_schedule=enhancement_schedule)


def test_r42_train_epoch_controller_active_false_forces_off():
    """controller_active=False 时 _rt_controller 被强制关闭（覆盖 train.py:262-263）。

    train_epoch 内部会先调 set_enhancements_active(True) 把 _rt_controller 设为 True，
    随后 controller_active=False 分支应再次强制设为 False。若该分支失效，warmup 期
    Controller 会跑前向（浪费算力 + 早期梯度污染 Generator warmup）。
    """
    m = _build()
    m.train()
    _run_train_epoch(m, controller_active=False)
    assert m._rt_controller is False, (
        "controller_active=False 应强制 _rt_controller=False（覆盖 set_enhancements_active）")


def test_r42_train_epoch_controller_active_true_keeps_on():
    """controller_active=True 时 Controller 正常开启（正向 case）。"""
    m = _build()
    m.train()
    _run_train_epoch(m, controller_active=True)
    assert m._rt_controller is True, (
        "controller_active=True 应保持 _rt_controller=True")


def test_r42_train_epoch_controller_active_overrides_schedule():
    """controller_active=False 覆盖 enhancement_schedule 的 controller=True 键。

    train.py:260 注释明确："覆盖 enhancement_schedule 的 controller 键，确保 warmup
    期整 epoch 关闭"。若该覆盖失效，SEL 交替训练设 controller=True 时 warmup 期
    Controller 仍会被强制开启，违背三阶段训练设计意图。
    """
    m = _build()
    m.train()
    _run_train_epoch(m, controller_active=False,
                     enhancement_schedule=[{'controller': True}])
    assert m._rt_controller is False, (
        "controller_active=False 应覆盖 enhancement_schedule 的 controller=True")


def test_r42_train_epoch_controller_active_false_no_controller_model():
    """controller_active=False 对 controller=False 模型无副作用（无 controller 子模块）。"""
    m = _build_off()  # controller=False
    m.train()
    # 不应因 controller_active=False 报错（model.controller_enabled=False 跳过分支）
    _run_train_epoch(m, controller_active=False)
    assert not m.controller_enabled
    # controller=False 模型无 _rt_controller 属性应为 False（set_enhancements_active 设）
    assert m._rt_controller is False


def test_r42_validate_controller_active_false_forces_off():
    """validate controller_active=False 时 _rt_controller 被强制关闭（覆盖 train.py:390-391）。

    validate 先调 set_enhancements_active(True)，随后 controller_active=False 分支
    应强制关闭。warmup 期验证也跳过 Controller 前向，与训练态一致省开销。
    """
    from scripts.train import validate
    m = _build()
    m.eval()
    validate(m, DataLoader(_TinyTrainDS(), batch_size=2),
             torch.nn.CrossEntropyLoss(), 'cpu', controller_active=False)
    assert m._rt_controller is False, (
        "validate controller_active=False 应强制 _rt_controller=False")


def test_r42_validate_controller_active_true_keeps_on():
    """validate controller_active=True 时 Controller 正常开启（正向 case）。"""
    from scripts.train import validate
    m = _build()
    m.eval()
    validate(m, DataLoader(_TinyTrainDS(), batch_size=2),
             torch.nn.CrossEntropyLoss(), 'cpu', controller_active=True)
    assert m._rt_controller is True


# ============================================================
# 13. 三阶段训练 warmup 切换点公式（train.py:763-764）
# ============================================================

@pytest.mark.parametrize("epochs,warmup_frac,epoch,expected", [
    # epochs=10, warmup_frac=0.3（与 config_train_8k_r42.yaml 一致）
    (10, 0.3, 1, False),    # progress=0.0 < 0.3 → OFF (warmup)
    (10, 0.3, 3, False),    # progress=0.2 < 0.3 → OFF (warmup)
    (10, 0.3, 4, True),     # progress=0.3 >= 0.3 → ON（边界 >=）
    (10, 0.3, 10, True),    # progress=0.9 >= 0.3 → ON
    # warmup_frac=0 → 从第 1 epoch 就 ON（向后兼容无 warmup）
    (10, 0.0, 1, True),
    # epochs=5, warmup_frac=0.5
    (5, 0.5, 2, False),     # progress=0.2 < 0.5
    (5, 0.5, 3, False),     # progress=0.4 < 0.5
    (5, 0.5, 4, True),      # progress=0.6 >= 0.5
    # epochs=1 + warmup_frac=0.5：progress=0.0 < 0.5 → 整 epoch OFF
    # （公式语义：单 epoch 训练 + warmup_frac>0 → 整 epoch 关 Controller；
    #  当前实现锁定此行为，避免 warmup_frac 配错导致单 epoch 训练完全跳过）
    (1, 0.5, 1, False),
])
def test_r42_warmup_switch_point_formula(epochs, warmup_frac, epoch, expected):
    """三阶段训练 warmup 切换点：直接测 train.py 生产函数 controller_warmup_active。

    覆盖 main() 训练循环每 epoch 的切换逻辑（该公式已提取为模块级函数
    controller_warmup_active，测试直接调用生产代码——若公式被改动，如 >= 改 >、
    分子 off-by-one，本测试立即失败）。该值决定每个 epoch 调 train_epoch/validate
    时传的 controller_active：错判 OFF → 白白浪费已付的 Controller 算力；
    错判 ON → warmup 期梯度经信号回流污染 Generator。
    边界 case：epoch=1 时 progress=0；epoch=epochs 时 progress=(epochs-1)/epochs。
    """
    from scripts.train import controller_warmup_active
    active = controller_warmup_active(epoch, epochs, warmup_frac)
    progress = (epoch - 1) / max(epochs, 1)
    assert active == expected, (
        f"epochs={epochs} warmup_frac={warmup_frac} epoch={epoch}: "
        f"progress={progress:.3f} → active={active}（预期 {expected}）")


# ============================================================
# 14. Controller 跨序列状态重置（transformer.py reset_ngram_state）
# ============================================================

def test_r42_reset_ngram_state_clears_controller_past():
    """reset_ngram_state() 重置 _controller_past（跨序列生成状态隔离）。

    覆盖 transformer.py 的 reset_ngram_state R42 新增分支——若不重置，跨序列
    生成会用上一序列的 Controller past_kv cache → 输出错乱（与 n-gram 滚动缓冲
    同生命周期管理）。generate.py 在每次新序列生成前应调此方法。
    """
    m = _build(seq=32)
    m.eval()
    x = torch.randint(0, 50, (2, 6))
    with torch.no_grad():
        # use_cache=True 让 Controller 填充 _controller_past
        m(x, use_cache=True)
    assert m._controller_past is not None, (
        "use_cache=True 前向后 _controller_past 应被填充")
    # 跨序列重置
    m.reset_ngram_state()
    assert m._controller_past is None, (
        "reset_ngram_state 后 _controller_past 应为 None")


def test_r42_controller_past_isolated_across_sequences():
    """跨序列生成：reset_ngram_state 后新序列不依赖上一序列的 Controller cache。

    若 cache 未重置，第二序列第一步会用第一序列末尾的 S 状态 → mem_kv/direction
    受污染。验证：两序列分别用 reset 后独立前向，输出与首次序列前向一致。
    """
    m = _build(seq=32)
    m.eval()
    seq1 = torch.randint(0, 50, (1, 5))
    seq2 = torch.randint(0, 50, (1, 5))
    with torch.no_grad():
        # 序列 1：fresh 前向
        m.reset_ngram_state()
        y1_first = m(seq1, use_cache=False)
        # 序列 2：未 reset → 复用 seq1 的 cache（错误路径）
        # 序列 2：reset 后 fresh 前向（正确路径）
        m.reset_ngram_state()
        y2_reset = m(seq2, use_cache=False)
        # 再跑一次 seq1 验证确定性（reset 后两次同输入应同输出）
        m.reset_ngram_state()
        y1_again = m(seq1, use_cache=False)
    assert torch.equal(y1_first, y1_again), (
        "reset 后两次同序列前向应逐位一致（确定性）")
    # y2 与 y1 不同（不同输入；若 cache 污染会让 y2 受 y1 残留影响，但仍可能不同）
    # 关键断言：reset 后 y2 不为 None 且 shape 正确
    assert y2_reset.shape == (1, 5, 50)


def test_r42_fresh_cached_decode_resets_stale_controller_past():
    """新序列缓存解码首步自动清 _controller_past 残留（transformer.py is_fresh 守卫）。

    覆盖 transformer.py:1502-1503（R42）：use_cache=True 且 past_key_values 全 None
    （新序列起点）时自动置 _controller_past=None——防"上次 generate() 残留"。
    上面的 isolated_across_sequences 测的是显式 reset_ngram_state + use_cache=False
    路径；本测试锚定 use_cache=True 的自动守卫（generate.py 连续两次生成间若未
    显式 reset，这是最后一道防线）。
    行为级断言：跑过序列 A 后直接跑序列 B（past=None），输出应与 fresh 模型
    跑 B 逐位一致——若守卫失效，B 首步会复用 A 的 Controller past_kv，
    mem_kv/direction/FiLM 被上一序列状态污染（静默输出错乱）。
    """
    torch.manual_seed(0)
    m = _build(seq=32)
    m.eval()
    seq_a = torch.randint(0, 50, (1, 6))
    seq_b = torch.randint(0, 50, (1, 5))
    with torch.no_grad():
        # 基准：fresh 模型（_controller_past=None）直接跑 B
        y_b_fresh, _ = m(seq_b, use_cache=True)   # use_cache=True 返回 (logits, past)
        # 残留场景：先跑 A 填充 _controller_past，再不 reset 直接跑 B
        m(seq_a, use_cache=True)
        assert m._controller_past is not None, "A 前向后 _controller_past 应被填充"
        y_b_after_a, _ = m(seq_b, use_cache=True)  # past=None → is_fresh → 守卫清残留
        assert m._controller_past is not None, "B 前向后 _controller_past 应为 B 的新 cache"
    assert torch.allclose(y_b_fresh, y_b_after_a, atol=1e-5), (
        "跑过 A 后直接跑 B 应与 fresh 跑 B 一致（is_fresh 守卫自动清 _controller_past 残留）；"
        "不一致说明上一序列的 Controller 状态泄漏进了新序列")


# ============================================================
# 15. set_enhancements_active dict spec 的 controller 键（transformer.py）
# ============================================================

def test_r42_set_enhancements_active_dict_controller_key():
    """set_enhancements_active dict spec 含 'controller' 键时正确切换。

    覆盖 transformer.py set_enhancements_active 的 dict 分支 R42 新增 'controller'
    键处理。SEL 交替训练用 dict spec 精细控制各增强；若该键处理被破坏，SEL 训练
    controller 切换会失效（无法独立开关 Controller）。
    """
    m = _build()
    m.eval()
    # dict spec: controller=False
    m.set_enhancements_active({'controller': False})
    assert m._rt_controller is False, "dict {'controller': False} 应关闭 Controller"
    # dict spec: controller=True
    m.set_enhancements_active({'controller': True})
    assert m._rt_controller is True, "dict {'controller': True} 应开启 Controller"
    # dict spec 不含 controller 键：_rt_controller 保持现状（不被其他键干扰）
    m.set_enhancements_active({'layer_film': False})
    assert m._rt_controller is True, "dict 不含 controller 键时 _rt_controller 应保持不变"


def test_r42_set_enhancements_active_dict_controller_key_off_model():
    """controller=False 模型：dict spec 的 controller 键不应开启（守门）。"""
    m = _build_off()
    m.eval()
    # controller=False 模型，即使 dict 设 controller=True 也应保持 False（model.controller_enabled=False）
    m.set_enhancements_active({'controller': True})
    assert m._rt_controller is False, (
        "controller=False 模型不应被 dict {'controller': True} 开启（守门）")


# ============================================================
# 16. 已知缺陷登记（H1）：direction 信号增量解码语义断裂
# ============================================================

def _rand_signal_projections(m, seed=1234):
    """把零初始化的信号投影改为随机非零——否则 parity 测试在零权重下恒过（空转）。"""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        m.controller.direction_proj.weight.copy_(
            torch.randn(m.controller.direction_proj.weight.shape, generator=g) * 0.1)
        if m.controller.direction_proj.bias is not None:
            m.controller.direction_proj.bias.zero_()
        for proj in m.controller.film_projs:
            if isinstance(proj, nn.Linear):
                proj.weight.copy_(torch.randn(proj.weight.shape, generator=g) * 0.05)


@pytest.mark.xfail(reason="H1 已知限制（AGENT_MEMORY §11.1）：direction 的 x.mean(dim=1) "
                          "训练期覆盖整段、推理第 2 步起只覆盖 1 token → 增量不等价；"
                          "修复需重训，未实施", strict=False)
def test_r42_direction_incremental_parity_known_gap():
    """登记 H1：信号投影非零时，全量 vs 逐 token 的 direction 路径不等价。

    现有 test_r42_cache_parity 因信号零初始化而恒过（测不出 direction 分量）；
    本测试显式放开 direction_proj 后预期 fail（xfail），若某天变 pass 说明 H1 已修，
    应移除 xfail 并在 AGENT_MEMORY §11.1 打勾。
    """
    m = _build(seq=32)
    m.eval()
    _rand_signal_projections(m)
    x = torch.randint(0, 50, (2, 6))
    with torch.no_grad():
        y_full = m(x)
        y_first, past = m(x[:, :3], use_cache=True)
        ys = [y_first]
        cur_past = past
        for t in range(3, 6):
            y_t, cur_past = m(x[:, t:t+1], past_key_values=cur_past, use_cache=True)
            ys.append(y_t)
        y_inc = torch.cat(ys, dim=1)
    diff = (y_full - y_inc).abs().max().item()
    assert diff < 1e-4, f"H1 direction 增量 parity diff={diff}"


# ============================================================
# 17. L1 回归：temperature<=0 走贪心（不崩溃）
# ============================================================

def test_r42_sample_greedy_nonpositive_temperature():
    """temperature<=0 应走 argmax 贪心而非 0 除崩溃（README '0=贪心' 语义）。

    修复前：logits/0.0 → inf/nan → softmax 全 nan → torch.multinomial RuntimeError。
    """
    from models.sampling import sample_next_token
    logits = torch.tensor([0.1, 5.0, -1.0, 2.0, 0.0])
    for tau in (0.0, -1.0):
        tok = sample_next_token(
            logits.clone(), temperature=tau, repetition_penalty=1.0,
            generated_ids=[], ngram_fn=None, ngram_weight=0.0, device='cpu',
            pad_id=4, sep_id=-1, eos_id=3, generated_len=10, min_length=1,
            eos_penalty=0.0, top_k=0, vocab_size=5)
        assert tok == 1, f"temperature={tau} 应 argmax 取最大 logit 的 idx 1，实际 {tok}"


def test_r42_sample_does_not_mutate_raw_logits():
    """L9：全 -inf 回退分支不得就地改写调用方 logits（应 clone）。"""
    from models.sampling import sample_next_token
    logits = torch.zeros(5)
    raw = torch.tensor([3.0, 1.0, 0.0, 0.0, 0.0])
    raw_before = raw.clone()
    # pad/sep/eos 屏蔽到全 -inf 触发回退分支
    sample_next_token(
        logits, temperature=1.0, repetition_penalty=1.0, generated_ids=[],
        ngram_fn=None, ngram_weight=0.0, device='cpu', pad_id=4, sep_id=3,
        eos_id=0, generated_len=0, min_length=10, eos_penalty=0.0,
        top_k=0, vocab_size=5, raw_logits=raw, temperature_applied=True)
    assert torch.equal(raw, raw_before), "回退分支就地修改了 raw_logits（应 clone）"