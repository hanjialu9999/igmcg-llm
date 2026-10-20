import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import gen_cmp


class FakeVocab:
    """encode: 小写、去空格、加 eos；decode: 还原。"""

    eos_idx = 0

    def encode(self, s):
        return [ord(c) for c in s.lower() if c != " "] + [self.eos_idx]

    def decode(self, ids):
        return "".join(chr(i) for i in ids if i != self.eos_idx)


def test_strip_echo_raw():
    v = FakeVocab()
    assert (
        gen_cmp.strip_echo("Hello, how are you世界和平", "Hello, how are you", v)
        == "世界和平"
    )


def test_strip_echo_normalized():
    v = FakeVocab()
    assert (
        gen_cmp.strip_echo("hello,howareyou世界和平", "Hello, how are you", v)
        == "世界和平"
    )


def test_strip_echo_none():
    v = FakeVocab()
    assert gen_cmp.strip_echo("世界和平", "Hello, how are you", v) == "世界和平"
    assert (
        gen_cmp.strip_echo("今天天气真好，我们去公园", "今天天气真好，我们去", v)
        == "公园"
    )


def test_metrics():
    assert gen_cmp.metrics([{"cont": "aab"}, {"cont": "bc"}]) == {
        "n": 2,
        "avg_chars": 2.5,
        "distinct1": 0.6,
        "distinct2": 1.0,
    }
