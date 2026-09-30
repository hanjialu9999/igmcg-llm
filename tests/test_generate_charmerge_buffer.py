"""第六批：generate.py 默认打开 CharMerge 增量缓冲 3 测试。

背景：第五批把 CharMerge 增量滚动缓冲（char_merge_incremental_buffer）修好，
但只有 `baseline_eval.py` 有 `--char-merge-buffer`（默认 off = 逐位可比）；
`generate.py` / `chat.py` 走 `load_model()` → 读 r42 config 里没有这个键 →
落到 ModelConfig 默认 False = 旧行为，修好的读取路径在生成入口**从没生效过**。

本轮：`generate.py` 加 `--char-merge-buffer {on,off}` **默认 on**（只有生成入口
默认开；config/checkpoint/baseline_eval 的默认值一个都不动）。

三个测试钉住三条线：
  1. CLI 默认值 = on，且在 load_model 之后、任何前向之前施加；
  2. off = 旧输出逐 token 相同（含"先开过又关掉"不能留残渣）+ 开关确实有效；
  3. 连续两条 prompt 第二条不带第一条的尾巴。

运行：python -m pytest tests/test_generate_charmerge_buffer.py -q
"""
import io
import os
import re

import pytest
import torch

from models.config_loader import build_model

VOCAB = 37
DIM = 16
DEVICE = 'cpu'
GEN_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'scripts', 'generate.py')


def _build(seed: int = 7, buffer: bool = False):
    """小模型：direction 关（controller=False），只开 char_merge + 本开关。"""
    torch.manual_seed(seed)
    cfg = {'model': {
        'vocab_size': VOCAB, 'embedding_dim': DIM, 'num_heads': 4,
        'num_layers': 2, 'hidden_dim': 64, 'max_seq_length': 16,
        'dropout': 0, 'mixer': 'attn',
        'char_merge': True,
        'char_merge_incremental_buffer': buffer,
        'alibi': False, 'controller': False,
    }}
    m = build_model(cfg, device=DEVICE)
    m.eval()
    m.set_enhancements_active(True)
    return m


def _gen(m, prompt_ids, seed: int = 101):
    """固定 seed 的端到端生成（temperature>0 → 走 multinomial，必须定种子）。"""
    torch.manual_seed(seed)
    with torch.no_grad():
        return m.generate(prompt_ids, max_length=12, temperature=0.8, top_k=50,
                          device=DEVICE, repetition_penalty=1.4,
                          min_length=3, eos_penalty=-5.0)


def _full_logits(m, ids):
    """整段前向 logits（use_cache=False）。入口的 is_fresh 会顺手清缓冲。"""
    tok = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = m(tok, use_cache=False)
    return out[0] if isinstance(out, tuple) else out


def _tokenwise_logits(m, ids):
    """KV 缓存逐 token 前向——生成真正走的路径，缓冲在这里跨步滚起来。

    注意整段前向在序列起点是"空历史 + 零填充"，与旧码逐位相同，因此
    "开关是否生效"只能用这条路径来验（见 test_2 末尾的失败教训）。
    """
    past, steps = None, []
    with torch.no_grad():
        for t in range(len(ids)):
            inp = torch.tensor([[ids[t]]], dtype=torch.long)
            lg, past = m(inp, past_key_values=past, use_cache=True)
            steps.append(lg[:, -1:, :])
    return torch.cat(steps, dim=1)


def test_1_cli_default_is_on_and_applied_after_load():
    """generate.py 默认把开关设成 on，且施加点在 load_model 之后、前向之前。

    用源码断言而非跑 main()：main() 内联构造 argparse、还要真 checkpoint，
    跑起来既慢又把测试和具体权重绑死（同 tests/test_h4_wiring.py 的做法）。
    """
    src = io.open(GEN_SRC, encoding='utf-8').read()

    m = re.search(
        r"add_argument\(\s*'--char-merge-buffer',\s*choices=\['on',\s*'off'\],\s*default='(\w+)'",
        src)
    assert m, (
        'generate.py 里找不到 `--char-merge-buffer` 的 argparse 定义，'
        '或它的 choices/default 写法变了 → 请同步更新本测试')
    assert m.group(1) == 'on', (
        f'生成入口的 CharMerge 缓冲默认值是 {m.group(1)!r}，必须是 on '
        f'（修好的读取路径；off 走的是 r42 旧码零填充）')

    i_arg = src.index("--char-merge-buffer")
    i_load = src.index('model, vocab = load_model(')
    i_set = src.index('model.char_merge.incremental_buffer = cmb_on')
    i_first_fwd = src.index('generate_text(', i_load)   # 第一次真实前向调用
    assert i_arg < i_load < i_set < i_first_fwd, (
        '开关必须在 load_model 之后（覆盖 config 读进来的 False）、'
        '且在第一次前向之前施加')


def test_2_off_matches_legacy_bitwise_and_switch_is_live():
    """off = 旧输出逐 token 相同；且 on/off 的 logits 必须不同（非恒真断言）。

    "旧输出" = 接线前的 generate.py —— 它从不碰这个开关，而 r42 config 里没有
    此键 → ModelConfig 默认 False。所以"从未施加"就是旧路径。这里再补一条更强的：
    先开过又关掉（含 reset_buffer）也必须回到旧输出，否则关不干净会留残渣。
    """
    prompt = [3, 5, 7, 9]
    m_never = _build(seed=13, buffer=False)   # 从未施加 = 旧路径
    m_off = _build(seed=13, buffer=False)     # 同 seed → 权重逐位相同

    # 模拟 CLI：先 on 跑一次，再切回 off —— 切换后必须无残留
    _gen(m_off, prompt, seed=201)
    m_off.char_merge.incremental_buffer = True
    m_off.char_merge.reset_buffer()
    _gen(m_off, prompt, seed=202)
    m_off.char_merge.incremental_buffer = False
    m_off.char_merge.reset_buffer()

    assert _gen(m_never, prompt) == _gen(m_off, prompt), (
        '关开关后的生成 token 与旧路径不逐位相同 → 切换留下残渣或默认路径被改坏')

    # 开关必须真的生效（否则上面的相等是恒真断言）。
    # 用逐 token 路径验：整段前向在序列起点是"空历史 + 零填充"，与旧码逐位相同，
    # 拿它比会误判成"开关没接进前向"。
    off_lg = _tokenwise_logits(m_never, prompt)
    on_m = _build(seed=13, buffer=False)
    on_m.char_merge.incremental_buffer = True   # CLI 施加的那次覆盖
    on_m.char_merge.reset_buffer()
    on_lg = _tokenwise_logits(on_m, prompt)
    assert not torch.equal(off_lg, on_lg), (
        'on/off 的逐 token logits 逐位相同 → 开关没接进前向，测试失效')
    assert (off_lg - on_lg).abs().max().item() > 1e-6, (
        'on/off 的逐 token logits 差值过小，测试可能退化')


def test_3_two_prompts_back_to_back_through_generate():
    """连续两条 prompt：第二条必须与"干净状态单跑第二条"逐 token 相同。

    复位入口是 `model.generate()` 开头的 `reset_ngram_state()`
    （models/transformer.py:1756，其中 :1766-1767 顺手 `char_merge.reset_buffer()`），
    外加 forward 入口对新序列的 `is_fresh` 判定。
    末尾的"能力检查"把复位换成空操作 → 必须真的不同，否则本测试是恒真断言。
    """
    pa, pb = [3, 5, 7, 9], [11, 13]
    m = _build(seed=29, buffer=True)
    clean = _build(seed=29, buffer=True)   # 同 seed → 权重逐位相同

    b_clean = _gen(clean, pb, seed=301)    # 干净状态单跑 B
    _gen(m, pa, seed=300)                  # 先跑 A，让 m 变"脏"（不手动复位）
    b_after_a = _gen(m, pb, seed=301)
    assert b_after_a == b_clean, (
        f'连跑后的第二条 = {b_after_a}，干净单跑 = {b_clean} → 上一序列的尾巴串进了新序列')

    # 能力检查：复位失效时必须能抓到差异，证明上面的相等不是恒真断言。
    # 比 logits 而非生成 token——泄漏引起的 logits 差约 3e-2 量级，走 temperature=0.8
    # 的采样很可能落回同一批 token，比 token 抓不到。
    _gen(m, pa, seed=300)                  # 再留一次 A 的尾巴（1, pad, D）
    assert m.char_merge._cm_buffer is not None, 'A 跑完应留下滚动缓冲，否则本检查无意义'
    real_reset = m.reset_ngram_state
    m.reset_ngram_state = lambda: None     # 同时挡住 generate() 开头与 forward 的 is_fresh 复位
    try:
        leaked_lg = _full_logits(m, pb)
    finally:
        m.reset_ngram_state = real_reset
    clean_lg = _full_logits(_build(seed=29, buffer=True), pb)
    diff = (leaked_lg - clean_lg).abs().max().item()
    assert diff > 1e-3, (
        f'复位被摘掉后整段 logits 仍与干净状态相同（diff={diff:.3e}）→ '
        f'缓冲没有跨序列状态，本测试抓不到串扰')


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
