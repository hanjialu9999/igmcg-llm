"""10-20 潜伏问题修复：LR 调度跨 epoch、weight decay 分组、generate 长提示截断、
非 pack 长行开窗 + 按行切分、词表只用 train 建、早退 aux CE 忽略 pad。"""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from models.constants import BOS_IDX, PAD_IDX
from models.data_utils import CharTokenizer, TextDataset, load_packed_data, split_dataset
from models.transformer import TransformerModel
from models.utils import load_optimizer_state, optimizer_param_groups


def _build(**kw):
    torch.manual_seed(0)
    return TransformerModel(vocab_size=200, embedding_dim=64, num_heads=4, num_layers=2,
                            hidden_dim=128, max_seq_length=32, **kw)


class _DS(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, i):
        x = torch.randint(5, 50, (8,), generator=torch.Generator().manual_seed(i))
        return {'input_ids': x, 'target_ids': x}


# ---- B: LR 调度跨 epoch ----

def _final_lr(**kw):
    from scripts.train import train_epoch
    m = _build()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    train_epoch(m, DataLoader(_DS(), batch_size=2), opt, torch.nn.CrossEntropyLoss(), 'cpu', 1,
                base_lr=1e-3, warmup_steps=0, lr_schedule='cosine', eta_min=0.0,
                show_progress=False, grad_accum_steps=1, use_amp=False, **kw)
    return opt.param_groups[0]['lr']


def test_lr_schedule_spans_epochs():
    from scripts.train import compute_lr
    exp = lambda s, tot: compute_lr(s, tot, 0, 1e-3, 0.0, 'cosine', 0.1)
    # 4 个有效步 / epoch，2 个 epoch：第 1 轮末在半程，第 2 轮末才衰减到底
    assert _final_lr(lr_total_eff=8, initial_eff_step=0) == pytest.approx(exp(4, 8))
    assert _final_lr(lr_total_eff=8, initial_eff_step=4) == pytest.approx(exp(8, 8))
    assert exp(4, 8) > 2 * exp(8, 8)
    # 不传 = 旧行为（只按本 epoch 计）
    assert _final_lr() == pytest.approx(exp(4, 4))


# ---- D: weight decay 分组 ----

def test_param_groups_wd0_is_single_group():
    m = _build()
    opt = torch.optim.AdamW(optimizer_param_groups(m, 0.0), lr=1e-3, weight_decay=0.0)
    assert len(opt.param_groups) == 1
    assert len(opt.param_groups[0]['params']) == len(list(m.parameters()))


def test_param_groups_skip_bias_norm_embedding():
    m = _build()
    groups = optimizer_param_groups(m, 0.1)
    decay, no_decay = groups
    assert decay['weight_decay'] == 0.1 and no_decay['weight_decay'] == 0.0
    ids = {id(p) for p in no_decay['params']}
    emb = [mod.weight for mod in m.modules() if isinstance(mod, torch.nn.Embedding)]
    assert emb and all(id(w) in ids for w in emb)
    assert all(p.ndim >= 2 for p in decay['params'])
    assert len(decay['params']) + len(no_decay['params']) == len(list(m.parameters()))


def test_load_optimizer_state_group_mismatch_warns(capsys):
    m = _build()
    old = torch.optim.AdamW(optimizer_param_groups(m, 0.0), lr=1e-3)
    new = torch.optim.AdamW(optimizer_param_groups(m, 0.1), lr=1e-3, weight_decay=0.1)
    assert load_optimizer_state(new, old.state_dict(), 'cpu') is False
    assert '未恢复' in capsys.readouterr().out
    same = torch.optim.AdamW(optimizer_param_groups(m, 0.0), lr=1e-3)
    assert load_optimizer_state(same, old.state_dict(), 'cpu') is True


# ---- E: generate 长提示截断 ----

def _gen(m, prompt, max_length=8):
    return m.generate(list(prompt), max_length=max_length, temperature=1.0, top_k=1,
                      repetition_penalty=1.0, min_length=max_length, eos_id=-1)


def test_generate_long_prompt_truncates_and_keeps_prefix():
    m = _build()
    prompt = [BOS_IDX] + [10 + i % 30 for i in range(40)]  # 41 > max_seq_length=32
    out = _gen(m, prompt)
    assert out[:len(prompt)] == prompt          # 返回值仍以完整原提示开头
    assert len(out) == len(prompt) + 8          # 此前一个 token 也不生成
    # 等价于用截断后的提示（BOS + 末 23 个）生成：keep = 32 - min(8, 16) = 24
    short = [BOS_IDX] + prompt[-23:]
    assert out[len(prompt):] == _gen(m, short)[len(short):]


def test_generate_short_prompt_unchanged():
    m = _build()
    prompt = [BOS_IDX, 11, 12, 13]
    out = _gen(m, prompt)
    assert out[:4] == prompt and len(out) == 12


# ---- H: 非 pack 长行开窗 + 按行切分 ----

def _char_vocab(texts):
    v = CharTokenizer(vocab_size=500, pretok=2)
    v.train(texts)
    return v


def test_long_line_windows_cover_every_token():
    texts = ['a' * 5, 'b' * 20, 'c' * 7]  # max_seq_length=7 → L=8，步长 7
    v = _char_vocab(texts)
    old = TextDataset(texts, v, 7, preprocess=False)
    ds = TextDataset(texts, v, 7, preprocess=False, long_line_windows=True)
    assert old.line_ids is None and len(old) == 3
    enc = [v.encode(t) for t in texts]
    assert ds.line_ids.tolist() == [0, 1, 1, 1, 2, 2]
    for li, toks in enumerate(enc):
        rows = ds.tokens[torch.from_numpy(ds.line_ids == li)]
        targets = [int(t) for r in rows for t in r[1:] if t != PAD_IDX]
        assert targets == toks[1:]                 # 每个 token 恰好当一次目标，末窗含 EOS
    assert torch.equal(ds.tokens[0], old.tokens[0])  # 短行与旧行为相同


def test_split_dataset_by_line_keeps_windows_together():
    texts = [chr(0x4e00 + i) * (3 + i % 25) for i in range(60)]
    ds = TextDataset(texts, _char_vocab(texts), 7, preprocess=False, long_line_windows=True)
    tr, va = split_dataset(ds, train_ratio=0.8, seed=1)
    assert len(tr) + len(va) == len(ds)
    lt = set(ds.line_ids[tr.indices].tolist())
    lv = set(ds.line_ids[va.indices].tolist())
    assert lt and lv and not (lt & lv)
    assert len(lt) == 48


# ---- G: 词表只用 train 建 ----

def test_vocab_train_only(tmp_path):
    lines = ['共' + chr(0x4e00 + i) * 3 for i in range(40)]  # 每行一个独有字
    f = tmp_path / 'c.txt'
    f.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    kw = dict(vocab_size=500, max_seq_length=8, test_split=0.25, seed=3, tokenizer_pretok=2)
    _, _, v_all = load_packed_data(str(f), **kw)
    _, _, v_tr = load_packed_data(str(f), vocab_train_only=True, **kw)
    perm = np.random.RandomState(3).permutation(40)
    train_chars = {c for i in perm[:30] for c in lines[i]}
    all_chars = {c for s in lines for c in s}
    assert all_chars <= set(v_all.word2idx)
    assert train_chars <= set(v_tr.word2idx)
    assert not ((all_chars - train_chars) & set(v_tr.word2idx))
    assert all_chars - train_chars  # 这组数据里 val 确有独有字符


# ---- F: 早退 aux CE 忽略 pad ----

def test_early_exit_aux_ce_ignores_pad():
    from models.transformer import F as _F
    seen = {}
    orig = _F.cross_entropy

    def spy(*a, **k):
        seen.setdefault('ignore_index', k.get('ignore_index'))
        return orig(*a, **k)

    m = _build(early_exit=True, early_exit_layers=[0])
    m.train()
    x = torch.randint(5, 50, (2, 8))
    _F.cross_entropy = spy
    try:
        m(x, targets=x)
    finally:
        _F.cross_entropy = orig
    assert seen['ignore_index'] == PAD_IDX
    assert m._early_exit_aux_loss is not None
    m(x, targets=torch.full_like(x, PAD_IDX))  # 整批全是 pad：0 而不是 NaN
    assert m._early_exit_aux_loss.item() == 0.0
