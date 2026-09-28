# -*- coding: utf-8 -*-
"""H4：采样参数在 4 个分支全部接上 CLI。

背景：接线前 --temperature/--top-k/--max-length/--repetition-penalty 只在
"无 --igmcg 的单 prompt 分支"生效，interactive / igmcg / 默认示例三个分支
落到硬编码；min_length、eos_penalty 连 CLI 都没有。基准要求生成参数可控。

验收（用户拍板）：
  1. 各分支都接上 CLI，新增 --min-length / --eos-penalty；
  2. 所有默认值 = 现在硬编码的值 → 不传参时输出与改前逐字一致；
  3. interactive 的"一个 prompt 出两个温度"默认保留，传 --temperature 才用单温度。
"""
import inspect
import sys
from argparse import Namespace

import pytest

from scripts import generate as G


def _ns(**kw):
    """模拟 parse_args 结果：未显式传参的项不建属性（argparse.SUPPRESS 语义）。"""
    return Namespace(**kw)


# ---------------------------------------------------------------------------
# 1. 不传参 → 各分支回落到接线前的硬编码值
# ---------------------------------------------------------------------------

# 断言的是"H4 接线前 generate.py 里写死的那些数字"，不是新引入的常量。
PRE_WIRING = {
    'prompt':       dict(max_length=30, temperature=0.8, top_k=50,
                         repetition_penalty=1.4, min_length=3, eos_penalty=-5.0),
    'prompt_igmcg': dict(max_length=30, temperature=0.8, top_k=50,
                         repetition_penalty=2.0, min_length=3, eos_penalty=-5.0),
    'interactive':  dict(max_length=20, temperature=None, top_k=50,
                         repetition_penalty=2.0, min_length=3, eos_penalty=-5.0),
    'example':      dict(max_length=20, temperature=0.8, top_k=50,
                         repetition_penalty=2.0, min_length=3, eos_penalty=-5.0),
}


@pytest.mark.parametrize("branch", sorted(PRE_WIRING))
def test_h4_no_args_falls_back_to_pre_wiring_values(branch):
    """不传任何采样参数时，解析结果必须等于接线前该分支的硬编码值。"""
    got = G._resolve_gen_args(_ns(), branch)
    assert got == PRE_WIRING[branch], (
        f'{branch} 分支缺省值变化：{got} != {PRE_WIRING[branch]}（会破坏「不传参逐字一致」）')


@pytest.mark.parametrize("branch", sorted(PRE_WIRING))
def test_h4_explicit_args_override_branch_default(branch):
    """显式传参必须覆盖分支缺省（否则等于没接线）。"""
    args = _ns(max_length=7, temperature=1.23, top_k=9,
               repetition_penalty=3.4, min_length=6, eos_penalty=-1.5)
    got = G._resolve_gen_args(args, branch)
    assert got == dict(max_length=7, temperature=1.23, top_k=9,
                       repetition_penalty=3.4, min_length=6, eos_penalty=-1.5)


def test_h4_defaults_table_covers_all_dispatch_branches():
    """main() 的 4 条分发分支都必须在缺省表里（新增分支漏登记会 KeyError）。"""
    assert set(G._GEN_DEFAULTS) == {'prompt', 'prompt_igmcg', 'interactive', 'example'}


# ---------------------------------------------------------------------------
# 2. interactive 双温度：默认保留，传 --temperature 才单温度
# ---------------------------------------------------------------------------

def test_h4_interactive_temps_default_is_dual():
    """不传 --temperature → [0.7, 0.9]（接线前行为）。"""
    assert G._interactive_temps(_ns()) == [0.7, 0.9]


def test_h4_interactive_temps_single_when_given():
    """传了 --temperature → 单温度列表（用户拍板的语义）。"""
    assert G._interactive_temps(_ns(temperature=0.35)) == [0.35]


def test_h4_interactive_mode_signature_defaults_unchanged():
    """interactive_mode 的签名默认值必须仍等于接线前的硬编码值。"""
    sig = inspect.signature(G.interactive_mode)
    d = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}
    assert d['max_length'] == 20
    assert d['top_k'] == 50
    assert d['repetition_penalty'] == 2.0
    assert d['min_length'] == 3
    assert d['eos_penalty'] == -5.0
    assert d['temps'] is None          # None → 运行时填 [0.7, 0.9]


def _scripted_input(monkeypatch, prompt='今天天气'):
    """input() 先给一个 prompt（触发生成），再给 'quit' 退出。"""
    it = iter([prompt, 'quit'])
    monkeypatch.setattr('builtins.input', lambda *a: next(it))


def test_h4_interactive_mode_uses_dual_temps_by_default(monkeypatch):
    """不传 temps 时 interactive_mode 实际跑两个温度。"""
    seen = []
    monkeypatch.setattr(G, 'generate_text',
                        lambda *a, **k: seen.append(k) or 'x')
    _scripted_input(monkeypatch)
    G.interactive_mode(model=None, vocab=None)
    assert [k['temperature'] for k in seen] == [0.7, 0.9]
    # 其余采样参数也必须是接线前的硬编码值
    for k in seen:
        assert k['max_length'] == 20
        assert k['top_k'] == 50
        assert k['repetition_penalty'] == 2.0
        assert k['min_length'] == 3
        assert k['eos_penalty'] == -5.0


def test_h4_interactive_mode_single_temp_when_temps_passed(monkeypatch):
    seen = []
    monkeypatch.setattr(G, 'generate_text', lambda *a, **k: seen.append(k) or 'x')
    _scripted_input(monkeypatch)
    G.interactive_mode(model=None, vocab=None, temps=[0.42])
    assert [k['temperature'] for k in seen] == [0.42]


# ---------------------------------------------------------------------------
# 3. main() 端到端接线：每条分支真正把参数传给生成函数
# ---------------------------------------------------------------------------

class _FakeVocab:
    def __len__(self):
        return 5


class _Recorder:
    """替身生成函数，记录每次调用的关键字参数。"""

    def __init__(self, ret='X'):
        self.calls = []
        self.ret = ret

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self.ret


def _run_main(monkeypatch, argv_extra, prompt=True, interactive=False, igmcg=False):
    """跑 generate.main()，用替身挡掉模型加载与真实生成，返回记录器。"""
    import tempfile
    import os

    tmp = tempfile.mkdtemp()
    fake_model = os.path.join(tmp, 'm.pt')
    fake_vocab = os.path.join(tmp, 'v.json')
    open(fake_model, 'wb').close()
    open(fake_vocab, 'wb').close()

    txt = os.path.join(tmp, 'p.txt')
    with open(txt, 'w', encoding='utf-8') as f:
        f.write('今天天气')

    gtext = _Recorder('X')
    gigmcg = _Recorder(('', []))
    ginter = _Recorder(None)

    monkeypatch.setattr(G, 'load_model', lambda *a, **k: (object(), _FakeVocab()))
    monkeypatch.setattr(G, 'get_device', lambda *a, **k: __import__('torch').device('cpu'))
    monkeypatch.setattr(G, 'generate_text', gtext)
    monkeypatch.setattr(G, 'generate_igmcg', gigmcg)
    monkeypatch.setattr(G, 'interactive_mode', ginter)

    argv = ['generate.py', '--model', fake_model, '--vocab', fake_vocab,
            '--device', 'cpu', '--dtype', 'fp32']
    if interactive:
        argv.append('--interactive')
    elif prompt:
        argv += ['--prompt-file', txt]
    if igmcg:
        argv.append('--igmcg')
    argv += argv_extra
    monkeypatch.setattr(sys, 'argv', argv)

    G.main()
    return gtext, gigmcg, ginter


def test_h4_main_prompt_branch_wires_all_params(monkeypatch):
    """非 igmcg prompt 分支：6 个采样参数全部来自 CLI/分支缺省，不落硬编码。"""
    gtext, _, _ = _run_main(monkeypatch, ['--max-length', '11', '--temperature', '1.5',
                                          '--top-k', '13', '--repetition-penalty', '1.7',
                                          '--min-length', '4', '--eos-penalty', '-2.0'])
    assert len(gtext.calls) == 1
    k = gtext.calls[0]
    assert (k['max_length'], k['temperature'], k['top_k']) == (11, 1.5, 13)
    assert (k['repetition_penalty'], k['min_length'], k['eos_penalty']) == (1.7, 4, -2.0)


def test_h4_main_prompt_branch_defaults(monkeypatch):
    gtext, _, _ = _run_main(monkeypatch, [])
    k = gtext.calls[0]
    assert (k['max_length'], k['temperature'], k['top_k']) == (30, 0.8, 50)
    assert (k['repetition_penalty'], k['min_length'], k['eos_penalty']) == (1.4, 3, -5.0)


def test_h4_main_igmcg_branch_wires_repetition_penalty(monkeypatch):
    """接线前 igmcg 分支不传 repetition_penalty（回落到函数签名 2.0）。"""
    _, gigmcg, _ = _run_main(monkeypatch, ['--max-length', '21', '--temperature', '0.55',
                                           '--top-k', '23', '--repetition-penalty', '1.9',
                                           '--min-length', '5', '--eos-penalty', '-3.0'],
                             igmcg=True)
    k = gigmcg.calls[0]
    assert (k['max_length'], k['base_temp'], k['top_k']) == (21, 0.55, 23)
    assert (k['repetition_penalty'], k['min_length'], k['eos_penalty']) == (1.9, 5, -3.0)


def test_h4_main_igmcg_branch_defaults(monkeypatch):
    _, gigmcg, _ = _run_main(monkeypatch, [], igmcg=True)
    k = gigmcg.calls[0]
    assert (k['max_length'], k['base_temp'], k['top_k']) == (30, 0.8, 50)
    assert (k['repetition_penalty'], k['min_length'], k['eos_penalty']) == (2.0, 3, -5.0)


def test_h4_main_interactive_branch_defaults_dual_temp(monkeypatch):
    _, _, ginter = _run_main(monkeypatch, [], interactive=True)
    k = ginter.calls[0]
    assert k['temps'] == [0.7, 0.9]        # 默认保留双温度
    assert (k['max_length'], k['top_k']) == (20, 50)
    assert (k['repetition_penalty'], k['min_length'], k['eos_penalty']) == (2.0, 3, -5.0)


def test_h4_main_interactive_branch_explicit_temperature(monkeypatch):
    _, _, ginter = _run_main(monkeypatch, ['--temperature', '0.61',
                                           '--max-length', '9',
                                           '--top-k', '7',
                                           '--repetition-penalty', '1.1'],
                             interactive=True)
    k = ginter.calls[0]
    assert k['temps'] == [0.61]            # 传了 --temperature 才单温度
    assert (k['max_length'], k['top_k'], k['repetition_penalty']) == (9, 7, 1.1)


def test_h4_main_example_branch_wires_all_params(monkeypatch):
    """接线前默认示例分支硬编码 max_length=20/temperature=0.8/top_k=50。"""
    gtext, _, _ = _run_main(monkeypatch, ['--max-length', '6', '--temperature', '0.2',
                                          '--top-k', '3', '--repetition-penalty', '2.5',
                                          '--min-length', '1', '--eos-penalty', '-0.5'],
                            prompt=False)
    assert len(gtext.calls) == 4           # 4 条示例
    for k in gtext.calls:
        assert (k['max_length'], k['temperature'], k['top_k']) == (6, 0.2, 3)
        assert (k['repetition_penalty'], k['min_length'], k['eos_penalty']) == (2.5, 1, -0.5)


def test_h4_main_example_branch_defaults(monkeypatch):
    gtext, _, _ = _run_main(monkeypatch, [], prompt=False)
    assert len(gtext.calls) == 4
    for k in gtext.calls:
        assert (k['max_length'], k['temperature'], k['top_k']) == (20, 0.8, 50)
        assert (k['repetition_penalty'], k['min_length'], k['eos_penalty']) == (2.0, 3, -5.0)


def test_h4_new_cli_flags_exist(monkeypatch):
    """--min-length / --eos-penalty 必须被 argparse 接受（接线前无此参数）。"""
    gtext, _, _ = _run_main(monkeypatch, ['--min-length', '8', '--eos-penalty', '-0.25'])
    assert gtext.calls[0]['min_length'] == 8
    assert gtext.calls[0]['eos_penalty'] == -0.25
