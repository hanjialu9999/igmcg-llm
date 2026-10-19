"""10-20：CharTokenizer pretok=2（data.tokenizer_pretok）。

保留大小写与空格、只删非法字符本身、全角字母数字转半角；缺省 pretok=1 = 旧行为，
旧 vocab.json 无 pretok 键时按 1 加载（旧 ckpt 生成 / 评测不变）。
"""
import json

from models.config_loader import load_vocab
from models.data_utils import CharTokenizer, load_packed_data

TEXTS = ['Hello World, how are you? 你好，世界。', 'ＡＢ１２ 测试\t多个   空白', '坏字\ufffd符 在中间 x\ufffdy']


def _tok(pretok):
    t = CharTokenizer(vocab_size=500, pretok=pretok)
    t.train(TEXTS)
    return t


def _roundtrip(t, s):
    return t.decode(t.encode(s))


def test_default_is_old_behavior():
    t = CharTokenizer(vocab_size=500)
    assert t.pretok == 1
    t.train(TEXTS)
    assert _roundtrip(t, 'Hello World') == 'helloworld'
    assert t.tokenize('坏字\ufffd符 好') == ['好']  # 旧行为：含坏字符的整词丢弃


def test_v2_keeps_case_and_spaces():
    t = _tok(2)
    assert ' ' in t.word2idx and 'H' in t.word2idx
    s = 'Hello World, how are you? 你好，世界。'
    assert _roundtrip(t, s) == s


def test_v2_collapses_whitespace():
    t = _tok(2)
    assert _roundtrip(t, '  测试\t多个 \n  空白　好 ') == '测试 多个 空白 好'


def test_v2_drops_only_invalid_char():
    t = _tok(2)
    assert _roundtrip(t, '坏字\ufffd符 在中间') == '坏字符 在中间'
    assert _roundtrip(t, 'x \ufffd y') == 'x y'  # 删掉坏字符后不留双空格


def test_v2_fullwidth_alnum_to_halfwidth_keeps_cjk_punct():
    t = _tok(2)
    assert _roundtrip(t, 'ＡＢ１２ 测试') == 'AB12 测试'
    assert t.tokenize('你好，世界。') == list('你好，世界。')
    assert 'Ａ' not in t.word2idx and 'A' in t.word2idx


def test_v3_keeps_space_only_between_ascii():
    t = _tok(3)
    assert _roundtrip(t, 'Hello World, how are you? 你好，世界。') == 'Hello World, how are you?你好，世界。'
    assert _roundtrip(t, '使用 Python 编程  测试\t多个 空白') == '使用Python编程测试多个空白'
    assert _roundtrip(t, 'x � y 3 号') == 'x y 3号'
    assert _roundtrip(t, 'ＡＢ１２ 测试') == 'AB12测试'
    assert ' ' in t.word2idx


def test_save_load_keeps_pretok(tmp_path):
    t = _tok(2)
    p = tmp_path / 'vocab.json'
    t.save(str(p))
    v = load_vocab(str(p))
    assert v.pretok == 2
    s = 'Hello World 你好'
    assert v.encode(s) == t.encode(s)
    data = json.loads(p.read_text(encoding='utf-8'))
    del data['pretok']  # 旧 vocab.json 无此键
    p.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    assert load_vocab(str(p)).pretok == 1


def test_save_final_model_keeps_pretok(tmp_path):
    """save_final_model 写的 vocab.json 也要带 char/pretok（10-20 r53 读回成小写去空格）。"""
    import torch
    from models.utils import save_final_model
    t = _tok(2)
    _, vp = save_final_model(torch.nn.Linear(2, 2), t, str(tmp_path))
    v = load_vocab(vp)
    assert isinstance(v, CharTokenizer) and v.pretok == 2
    s = 'Hello World 你好'
    assert v.encode(s) == t.encode(s)


def test_load_packed_data_passes_pretok(tmp_path):
    f = tmp_path / 'c.txt'
    f.write_text('\n'.join(TEXTS * 4) + '\n', encoding='utf-8')
    _, _, v1 = load_packed_data(str(f), vocab_size=500, max_seq_length=8, test_split=0.25)
    _, _, v2 = load_packed_data(str(f), vocab_size=500, max_seq_length=8, test_split=0.25,
                                tokenizer_pretok=2)
    assert v1.pretok == 1 and ' ' not in v1.word2idx
    assert v2.pretok == 2 and ' ' in v2.word2idx
