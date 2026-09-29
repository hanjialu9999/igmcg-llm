"""第四批：CharMergeLayer 增量滚动缓冲（char_merge_incremental_buffer）4 测试。

背景（详见 docs/ARCHITECTURE.md 附录 B char_merge 条目）：
CharMerge 是 kernel=3 的因果 depthwise 卷积，整段前向里位置 t 的窗口是
[x_{t-2}, x_{t-1}, x_t]；而 KV 缓存增量解码每步只喂 1 个 token，旧码
`F.pad(x_t, (self.pad, 0))` 把窗口补成 [0, 0, x_t]，丢掉真实的前 2 个 token。
后果 = 第二处训推不一致：同一模型全量 prefix=5.378725 / 增量 incr=5.907944。

本轮修法（`char_merge_incremental_buffer`，默认 False = r42 config 不动）：
层内保存最近 self.pad 个输入作左邻域，逐 token 用它替代零填充，只用 cat + 切片
（DML 禁 in-place / scatter）。因为只改**读取路径**，旧权重直接受益，不需要重训。

四个测试分别钉住四条回归线，改动这个层之前先跑本文件：
  1. 关（默认）= 旧码逐位一致 —— 开关关闭时不得改变任何现有行为；
  2. 开 + direction 关：整段 vs 逐 token 的 logits max diff < 1e-4；
  3. 连续两次不同 prompt 不串 —— 缓冲必须在新序列开头清空；
  4. batch 尺寸变化正确重建 —— 形状不同不能复用旧行。

运行：python -m pytest tests/test_charmerge_buffer.py -q
"""
import pytest
import torch
import torch.nn.functional as F

from models.config_loader import build_model
from models.layers import CharMergeLayer

VOCAB = 37
DIM = 16
DEVICE = 'cpu'


def _layer(buffer: bool, seed: int = 3):
    """独立 CharMergeLayer（dropout=0 → eval/train 无随机性，可逐位比较）。"""
    torch.manual_seed(seed)
    return CharMergeLayer(DIM, kernel_size=3, dropout=0.0,
                          incremental_buffer=buffer).eval()


def _legacy_forward(layer: CharMergeLayer, x: torch.Tensor) -> torch.Tensor:
    """旧码参考实现：F.pad 零填充 → conv1d → 门控 → RMSNorm → dropout。

    逐字复刻改动前的 forward，用作"关 = 旧码逐位一致"的对照基准
    （铁律 8 证据锚定：不能只断言"两条新路径互相相等"）。
    """
    xt = x.transpose(1, 2)
    x_padded = F.pad(xt, (layer.pad, 0))
    agg = F.conv1d(x_padded, layer.conv.weight, None, groups=DIM)
    agg = agg.transpose(1, 2)
    z = torch.sigmoid(layer.gate(x))
    out = x + z * (agg - x)
    return layer.drop(layer.norm(out))


def _build(buffer: bool, seed: int = 7):
    """小模型：direction 关（controller=False，无 Controller/方向扰动）。

    只开 char_merge + 本开关，好把整段 vs 逐 token 的差值全部归因到这一层。
    """
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


def _full_logits(m, ids: torch.Tensor) -> torch.Tensor:
    """整段前向（past=None）。入口的新序列判定会顺手清缓冲。

    use_cache=False 时 forward 直接返回裸 logits 张量，use_cache=True 才返回
    (logits, past) 元组 —— 两种返回形态都要兼容，否则解包会把 batch 维拆掉。
    """
    with torch.no_grad():
        out = m(ids, use_cache=False)
    return out[0] if isinstance(out, tuple) else out


def _tokenwise_logits(m, ids: torch.Tensor) -> torch.Tensor:
    """KV 缓存逐 token 前向：每步只喂 1 个 token，缓冲必须跨步滚起来。"""
    T = ids.size(1)
    past = None
    steps = []
    with torch.no_grad():
        for t in range(T):
            inp = ids[:, :1] if t == 0 else ids[:, t:t + 1]
            logits, past = m(inp, past_key_values=past, use_cache=True)
            # 取 (B, 1, V) 而非 (B, V)，才能沿 dim=1 拼成 (B, T, V)
            steps.append(logits[:, -1:, :])
    return torch.cat(steps, dim=1)


def test_1_off_matches_legacy_code_bitwise():
    """关（默认）= 旧码逐位一致；且关闭路径完全不留状态。

    覆盖两层：
    - 层内：`incremental_buffer=False` 的输出与旧码参考 `torch.equal`（不是
      allclose —— 用浮动容差就测不出"新写法改变了默认行为"）。
    - 层内：关闭时不产生 `_cm_buffer`（否则平白多占显存/引入跨步状态）。
    - 配置：新开关默认 False，r42 config 一个键都没动 → 旧 checkpoint 行为不变。
    """
    from dataclasses import fields
    from models.model_config import ModelConfig
    torch.manual_seed(11)
    layer = _layer(buffer=False, seed=11)
    x = torch.randn(3, 8, DIM)
    with torch.no_grad():
        got = layer(x)
        want = _legacy_forward(layer, x)
    assert torch.equal(got, want), (
        f'开关关闭时输出与旧码不逐位相等（max diff='
        f'{(got - want).abs().max().item():.3e}）→ 默认路径被改坏了')
    assert layer._cm_buffer is None, '开关关闭时不应产生滚动缓冲状态'
    # ModelConfig 带 __post_init__ 断言（vocab_size>0），不能空构造取默认值，
    # 只能查 dataclass 字段默认 —— r42 config 未加此键，走默认即旧行为。
    field = {f.name: f for f in fields(ModelConfig)}['char_merge_incremental_buffer']
    assert field.default is False, (
        '新开关默认值必须为 False（r42 config 未加此键，走默认 = 旧行为）')


def test_2_open_full_vs_tokenwise_max_diff():
    """开 + direction 关：整段前向 vs 逐 token 前向的 logits 最大绝对差 < 1e-4。

    这是本修复的核心断言 —— 修之前该差值是 0.96 量级（窗口退化成 [0,0,x_t]
    导致的数值差异，不是 fp 误差），修复后应只剩卷积分块带来的浮点舍入。
    用 logits 而非 loss 比较：loss 是标量聚合，会把"某些位置错了、某些位置对了"
    平均掉，测不出逐位置的偏差。
    """
    m = _build(buffer=True, seed=23)
    torch.manual_seed(101)
    ids = torch.randint(0, VOCAB, (2, 8))
    full = _full_logits(m, ids)
    stepwise = _tokenwise_logits(m, ids)
    assert full.shape == stepwise.shape
    diff = (full - stepwise).abs().max().item()
    assert diff < 1e-4, (
        f'开缓冲后整段 vs 逐 token logits max diff={diff:.3e} ≥ 1e-4 → '
        f'CharMerge 增量滚动缓冲没把窗口补对（或引入了别的训推不一致）')


def test_3_two_prompts_back_to_back_do_not_contaminate():
    """连续两次不同 prompt 不串：后一次必须与"干净状态跑同 prompt"逐位相同。

    缓冲是跨步状态，跑完 prompt A 会留下 A 的尾巴；若不在新序列开头清掉，
    prompt B 的前 pad 个 token 会把 A 的尾巴当左邻域 → 串扰（与 n-gram 滚动
    缓冲 / `_controller_past` 同类的 M9/N3 坑）。两层都钉住：
    - 层内：不 reset 会串（证明本测试有能力抓到串扰，不是恒真断言）；
    - 模型内：A→B 连跑 与 干净跑 B 逐位相同（证明 transformer 入口的清空生效）。
    """
    # 层内：证明 reset_buffer() 是串扰的唯一分界
    dirty = _layer(True, seed=31)
    clean = _layer(True, seed=31)          # 同 seed → 权重完全相同
    torch.manual_seed(301)
    prompt_a = torch.randn(1, 6, DIM)
    prompt_b = torch.randn(1, 6, DIM)
    with torch.no_grad():
        _ = dirty(prompt_a)                 # 留下 A 的尾巴
        b_after_a_dirty = dirty(prompt_b)   # 不 reset → B 看见 A 的尾巴
        b_clean = clean(prompt_b)           # 从未跑过 A → 零历史
        dirty.reset_buffer()                # 复位
        b_after_a_reset = dirty(prompt_b)
    assert not torch.equal(b_after_a_dirty, b_clean), (
        '不 reset 也与干净状态相同 → 本测试抓不到串扰（恒真断言），测试本身失效')
    assert torch.equal(b_after_a_reset, b_clean), (
        'reset_buffer() 之后仍不等于干净状态 → 复位没真正清掉缓冲')

    # 模型内：A→B 连跑必须等于干净跑 B
    torch.manual_seed(401)
    ids_a = torch.randint(0, VOCAB, (1, 7))
    ids_b = torch.randint(0, VOCAB, (1, 5))
    assert not torch.equal(ids_a, ids_b), '两个 prompt 必须不同，否则测不出串扰'
    m = _build(buffer=True, seed=37)
    with torch.no_grad():
        _ = _tokenwise_logits(m, ids_a)             # 脏状态：A 逐 token 跑完
        b_dirty = _tokenwise_logits(m, ids_b)       # 紧接着跑 B（不手动复位）
        m.reset_ngram_state()                       # 新一次生成开头的统一复位
        b_clean = _tokenwise_logits(m, ids_b)
    assert torch.equal(b_dirty, b_clean), (
        f'连跑 prompt B 与干净状态跑 B 不一致（max diff='
        f'{(b_dirty - b_clean).abs().max().item():.3e}）→ 上一序列的尾巴串进了新序列')


def test_4_batch_size_change_rebuilds_buffer():
    """batch 尺寸变化必须重建缓冲，不能复用形状不符的旧行。

    缓冲是 (B, pad, D)：batch 变了形状就对不上，复用旧行要么直接形状报错，
    要么（若盲目广播/切片）把别的样本的尾巴塞进当前样本。重建为零 = 回到
    "无历史"，与开关关闭时的零填充等价，是最保守也最正确的选择。
    """
    polluted = _layer(True, seed=53)
    fresh = _layer(True, seed=53)           # 同 seed → 权重完全相同
    torch.manual_seed(501)
    x_old = torch.randn(5, 4, DIM)          # batch=5，跑完留下 (5, 2, DIM)
    with torch.no_grad():
        _ = polluted(x_old)
        assert polluted._cm_buffer is not None and polluted._cm_buffer.shape[0] == 5, (
            f'跑完 batch=5 后缓冲形状应为 5 行，实际 '
            f'{None if polluted._cm_buffer is None else tuple(polluted._cm_buffer.shape)}')
        x_new = torch.randn(2, 1, DIM)      # batch 5→2，且 T=1（纯增量步）
        y_rebuilt = polluted(x_new)         # 形状不符 → 必须重建
        y_fresh = fresh(x_new)              # 从未用过 → None → 同样重建
    assert polluted._cm_buffer.shape == (2, 2, DIM), (
        f'batch=2 之后缓冲应重建为 (2, 2, {DIM})，实际 '
        f'{tuple(polluted._cm_buffer.shape)}')
    assert torch.equal(y_rebuilt, y_fresh), (
        'batch 变化后的输出与"干净缓冲"不逐位相同 → 旧行被错误复用了')
    assert torch.equal(y_rebuilt, _legacy_forward(polluted, x_new)), (
        'batch 变化重建为零后应与旧码零填充路径逐位一致')


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
