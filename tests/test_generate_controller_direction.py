"""第七批：generate.py / chat.py 默认关 Controller direction 4 测试。

背景：第六批给 `baseline_eval.py` 加了独立 `--direction`；但 `generate.py` / `chat.py`
从不碰 `use_direction`，走 `load_model()` → 读 config 的 `controller_direction`
（r42/默认 = True = 旧行为）。本轮要的 r42 生成对照口径是 direction 关。

本轮：两个生成入口加 `--controller-direction {on,off}` **默认 off**（只有生成入口
默认关；model_config 默认值、checkpoint、baseline_eval 的默认值一个都不动）。

四个测试钉住四条线：
  1. CLI 默认值 = off，且在 load_model 之后、任何前向之前施加；
  2. on = 旧输出逐 token 相同（固定 seed），off 与 on 的 logits 必须不同（开关有效）；
  3. 实际生效值打进日志并写进 meta 文件；
  4. helper 只切运行时开关，不动 state_dict、不动 config 默认值。

运行：python -m pytest tests/test_generate_controller_direction.py -q
"""
import io
import os
import re

import pytest
import torch

from models.config_loader import build_model
from scripts.generate import apply_controller_direction

VOCAB = 37
DIM = 16
DEVICE = 'cpu'
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_SRC = io.open(os.path.join(_ROOT, 'scripts', 'generate.py'), encoding='utf-8').read()
CHAT_SRC = io.open(os.path.join(_ROOT, 'scripts', 'chat.py'), encoding='utf-8').read()


def _build(seed: int = 7):
    """小模型：Controller 开、direction 取 config 默认（True = 旧行为）。"""
    torch.manual_seed(seed)
    cfg = {'model': {
        'vocab_size': VOCAB, 'embedding_dim': DIM, 'num_heads': 4,
        'num_layers': 2, 'hidden_dim': 64, 'max_seq_length': 16,
        'dropout': 0, 'mixer': 'attn',
        'char_merge': False,
        'alibi': False, 'controller': True,
    }}
    m = build_model(cfg, device=DEVICE)
    m.eval()
    m.set_enhancements_active(True)
    return m


def _gen(m, prompt_ids, seed: int = 101):
    torch.manual_seed(seed)
    with torch.no_grad():
        return m.generate(prompt_ids, max_length=12, temperature=0.8, top_k=50,
                          device=DEVICE, repetition_penalty=1.4,
                          min_length=3, eos_penalty=-5.0)


def _logits(m, ids):
    tok = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = m(tok, use_cache=False)
    return out[0] if isinstance(out, tuple) else out


def test_1_cli_default_is_off_and_applied_after_load():
    """两个生成入口默认 off，且施加点在 load_model 之后、第一次前向之前。

    用源码断言而非跑 main()：main() 内联构造 argparse、还要真 checkpoint，
    跑起来既慢又把测试和具体权重绑死（同 tests/test_generate_charmerge_buffer.py）。
    """
    for src, name in ((GEN_SRC, 'generate.py'), (CHAT_SRC, 'chat.py')):
        m = re.search(
            r"add_argument\(\s*'--controller-direction',\s*choices=\['on',\s*'off'\],\s*default='(\w+)'",
            src)
        assert m, f'{name} 里找不到 `--controller-direction` 的 argparse 定义（或写法变了 → 请同步本测试）'
        assert m.group(1) == 'off', (
            f'{name} 的 Controller direction 默认值是 {m.group(1)!r}，必须是 off '
            f'（本轮 r42 生成对照口径；on = config 默认旧行为）')

    i_arg = GEN_SRC.index('--controller-direction')
    i_load = GEN_SRC.index('model, vocab = load_model(')
    i_apply = GEN_SRC.index('dir_on = apply_controller_direction(model,')
    i_first_fwd = GEN_SRC.index('generate_text(', i_load)
    assert i_arg < i_load < i_apply < i_first_fwd, (
        '开关必须在 load_model 之后（覆盖 config 读进来的 True）、且在第一次前向之前施加')

    # 施加动作本身必须真的改 use_direction，且不碰 config 默认值
    i_helper = GEN_SRC.index('def apply_controller_direction(')
    body = GEN_SRC[i_helper:GEN_SRC.index('@cli_guard', i_helper)]
    assert 'model.controller.use_direction' in body, 'helper 没有写入 use_direction'

    import inspect
    from models.model_config import ModelConfig
    from models.transformer import TransformerModel
    assert inspect.signature(TransformerModel.__init__).parameters['controller_direction'].default is True, \
        'TransformerModel 的 controller_direction 默认值被改成非 True（必须保持旧行为）'
    assert ModelConfig.__dataclass_fields__['controller_direction'].default is True, \
        'ModelConfig 的 controller_direction 默认值被改成非 True（必须保持旧行为）'


def test_2_on_matches_legacy_and_off_is_live():
    """on = 旧输出逐 token 相同；off 与 on 的 logits 必须不同（非恒真断言）。

    "旧输出" = 接线前的 generate.py —— 它从不碰 use_direction，而 config 默认
    controller_direction=True，所以"从未施加"就是旧路径。
    """
    prompt = [3, 5, 7, 9]
    m_legacy = _build(seed=13)   # 从未施加 = 旧路径
    # direction_proj 权重被 _apply_neutral_inits 清零 → 未训练模型 on/off 必然同值。
    # 这里把它随机化（= 训练后非零的模型），三个模型共用同一份 state_dict。
    torch.manual_seed(777)
    with torch.no_grad():
        m_legacy.controller.direction_proj.weight.normal_(0.0, 0.05)
    sd = m_legacy.state_dict()

    m_on = _build(seed=13)
    m_on.load_state_dict(sd)     # 同权重 → 只有 use_direction 一个差别
    m_off = _build(seed=13)
    m_off.load_state_dict(sd)

    assert apply_controller_direction(m_on, True) is True
    assert apply_controller_direction(m_off, False) is False
    assert m_legacy.controller.use_direction is True, 'config 默认应为 on（旧行为）'

    # 开关必须真的生效（否则下面的相等是恒真断言）
    assert not torch.equal(_logits(m_legacy, prompt), _logits(m_off, prompt)), (
        'direction on/off 的 logits 逐位相同 → 开关没接进前向，测试失效')
    assert (_logits(m_legacy, prompt) - _logits(m_off, prompt)).abs().max().item() > 1e-6

    assert _gen(m_legacy, prompt) == _gen(m_on, prompt), (
        '--controller-direction on 的生成 token 与旧输出不逐位相同 → 默认路径被改坏')

    # 施加点是 runtime 开关，来回切必须干净（on→off 不能留残渣）
    assert _gen(m_off, prompt) != _gen(m_on, prompt) or \
        not torch.equal(_logits(m_off, prompt), _logits(m_on, prompt)), \
        'on→off 往返后与 on 相同 → 开关切换留了残渣'


def test_3_effective_value_printed_and_written_to_meta():
    """实际生效值：打进 stdout 日志 + 写进 logs/generation_output.txt 的 meta 区。"""
    m = re.search(
        r'print\(f"Controller direction = \{\'on\' if dir_on else \'off\'\}'
        r'"\s*\n\s*f"（--controller-direction \{args\.controller_direction\}）"\)',
        GEN_SRC)
    assert m, 'generate.py 缺少「打印实际生效值 dir_on」的日志行（写法变了 → 请同步本测试）'

    assert 'of.write(f"controller_direction = {dir_on}\\n")' in GEN_SRC, \
        'logs/generation_output.txt 的 meta 区必须记录 controller_direction 实际值'

    m_chat = re.search(
        r"print\(f\"Controller direction = \{'on' if dir_on else 'off'\}\"", CHAT_SRC)
    assert m_chat, 'chat.py 缺少打印实际生效值的日志行'

    # helper 返回的就是实际生效值：controller 关闭时即使 CLI 传 on 也只能是 False
    class _NoCtrl:
        controller_enabled = False
    assert apply_controller_direction(_NoCtrl(), True) is False


def test_4_helper_only_touches_runtime_switch():
    """helper 只写 use_direction，不动任何参数/state_dict（config/checkpoint 不受影响）。"""
    m = _build(seed=5)
    state_before = {k: v.clone() for k, v in m.state_dict().items()}
    apply_controller_direction(m, False)
    assert m.controller.use_direction is False
    apply_controller_direction(m, True)
    assert m.controller.use_direction is True
    assert len(m.state_dict()) == len(state_before)
    for k, v in m.state_dict().items():
        assert torch.equal(v, state_before[k]), f'{k} 被 helper 改动了（应只切运行时开关）'


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
