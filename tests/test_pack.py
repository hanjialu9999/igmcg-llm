"""10-05 数据打包（data.pack）：先按行切 train/val 再各自打包成 L=max_seq_length+1 的块。

旧行为（pack 缺省关）每行截到 L，8k 语料每 epoch 只用 7.58%；打包后长行全文都进训练。
"""

import os
import tempfile

import numpy as np
import yaml

from models.data_utils import CharTokenizer, TextDataset, load_packed_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINES = ['今天天气真好我们去公园玩', '北京是中国的首都', '上海有很多高楼大厦和人',
         '测试文本数据一二三四五六七八九十', '短', '我在家里看书写字画画唱歌跳舞']


def _vocab():
    v = CharTokenizer()
    v.train(LINES, min_freq=1)
    return v


def _stream(v, lines):
    return np.concatenate([np.asarray(v.encode(t), dtype=np.int32) for t in lines])


def test_pack_chunks_tile_stream_with_one_token_overlap():
    v = _vocab()
    T = 8
    ds = TextDataset(LINES, v, max_seq_length=T, preprocess=False, pack=True)
    s = _stream(v, LINES)
    n = (len(s) - 1) // T
    assert len(ds) == n and ds.tokens.shape == (n, T + 1)
    for i in range(n):
        assert np.array_equal(ds.tokens[i].numpy(), s[i * T:i * T + T + 1])
        item = ds[i]
        assert item['input_ids'].tolist() == s[i * T:i * T + T].tolist()
        assert item['target_ids'].tolist() == s[i * T + 1:i * T + T + 1].tolist()
    # 目标覆盖流里 1..n*T 每个位置，块里没有 pad
    assert (ds.tokens != v.pad_idx).all()


def test_pack_max_chunks_fixed_sorted_subset():
    v = _vocab()
    full = TextDataset(LINES, v, max_seq_length=4, preprocess=False, pack=True)
    a = TextDataset(LINES, v, max_seq_length=4, preprocess=False, pack=True, max_chunks=5, seed=7)
    b = TextDataset(LINES, v, max_seq_length=4, preprocess=False, pack=True, max_chunks=5, seed=7)
    assert len(full) > 5 and len(a) == 5
    assert np.array_equal(a.tokens.numpy(), b.tokens.numpy())
    rows = [r.tolist() for r in full.tokens]
    idx = [rows.index(r.tolist()) for r in a.tokens]
    assert idx == sorted(idx)


def test_pack_off_keeps_truncate_pad_behaviour():
    v = _vocab()
    ds = TextDataset(LINES, v, max_seq_length=8, preprocess=False)
    assert ds.tokens.shape == (len(LINES), 9)
    assert ds.tokens[4].tolist()[:3] == v.encode('短')
    assert set(ds.tokens[4].tolist()[3:]) == {v.pad_idx}


def test_load_packed_data_splits_by_line_before_packing():
    lines = [f'第{i}行' + '天气真好' * (i % 5 + 1) for i in range(40)]
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'c.txt')
        with open(p, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines + lines[:3]) + '\n')  # 末尾 3 行重复，应被去重
        tr, va, v = load_packed_data(p, max_seq_length=8, test_split=0.25, seed=1)
        tr2, va2, _ = load_packed_data(p, max_seq_length=8, test_split=0.25, seed=1, vocab=v)
        _, none_val, _ = load_packed_data(p, max_seq_length=8, test_split=0.0, vocab=v)
    assert none_val is None
    assert np.array_equal(tr.tokens.numpy(), tr2.tokens.numpy())
    assert np.array_equal(va.tokens.numpy(), va2.tokens.numpy())
    # 去重后 40 行：train 30 行 / val 10 行，两边 token 总量 ≈ 各自行编码之和（只丢尾部不足一块的部分）
    total = len(_stream(v, lines))
    got = (len(tr) + len(va)) * 8
    assert total - 2 * 8 <= got <= total
    # 同一行的行号标记不会同时出现在 train 和 val
    dec = lambda ds: v.decode(ds.tokens.reshape(-1).tolist())
    tr_txt, va_txt = dec(tr), dec(va)
    import re
    tr_ids = set(re.findall(r'第(\d+)行', tr_txt))
    va_ids = set(re.findall(r'第(\d+)行', va_txt))
    assert va_ids and not (tr_ids & va_ids)


def test_configs_pack_switch():
    def data(name):
        with open(os.path.join(ROOT, 'configs', name), encoding='utf-8') as f:
            return yaml.safe_load(f)['data']
    d = data('config_train_8k_pack.yaml')
    assert d['pack'] is True and d['pack_max_train_chunks'] == 21600 and d['pack_max_val_chunks'] == 2400
    # 旧 config 不设 pack → 保持旧行为
    for name in ('config_train_8k.yaml', 'config_train_8k_r42.yaml'):
        assert not data(name).get('pack', False)
