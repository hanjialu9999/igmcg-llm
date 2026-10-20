import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import blind_eval

MODELS = ["r45", "r47", "r55"]
PROMPTS = ["今天天气真好，我们去", "Hello, how are you"]


def fake_gen(path, models=MODELS):
    samples = []
    for ctrl in ("on", "off"):
        for setting in ("T0.8", "greedy"):
            for seed in (0, 1):
                for p in PROMPTS:
                    for m in models:
                        samples.append(
                            {
                                "model": m,
                                "controller": ctrl,
                                "setting": setting,
                                "seed": seed,
                                "prompt": p,
                                "cont": f"{m}|{ctrl}|{setting}|{seed}|{p}",
                            }
                        )
    g = {
        "prompts": PROMPTS,
        "sets": [["T0.8", 0.8, [0, 1]], ["greedy", 0.0, [42]]],
        "samples": samples,
    }
    path.write_text(json.dumps(g, ensure_ascii=False), encoding="utf-8")
    return g


def test_make_counts_and_off(tmp_path):
    fake_gen(tmp_path / "gen.json")
    items, key, full = blind_eval.make(tmp_path / "gen.json", tmp_path, "_t", MODELS)
    assert len(items) == len(key) == len(full) == 2 * 2 * 2
    assert (tmp_path / "blind_items_t.txt").read_text(encoding="utf-8").count("#") == 8
    assert "|off|" not in (tmp_path / "blind_items_t.txt").read_text(encoding="utf-8")
    gen_all = (tmp_path / "gen_all_t.txt").read_text(encoding="utf-8-sig")
    assert "|off|" not in gen_all and gen_all.startswith("#1 ")


def test_make_deterministic(tmp_path):
    fake_gen(tmp_path / "gen.json")
    blind_eval.make(tmp_path / "gen.json", tmp_path, "_a", MODELS)
    blind_eval.make(tmp_path / "gen.json", tmp_path, "_b", MODELS)
    for fn in ("blind_items", "blind_key", "gen_all"):
        ext = ".json" if fn == "blind_key" else ".txt"
        assert (tmp_path / f"{fn}_a{ext}").read_bytes() == (
            tmp_path / f"{fn}_b{ext}"
        ).read_bytes()


def test_make_models_filter_and_too_many(tmp_path):
    fake_gen(tmp_path / "gen.json", MODELS + ["r56", "r57"])
    items, key, _ = blind_eval.make(
        tmp_path / "gen.json", tmp_path, "_f", ["r45", "r47"]
    )
    assert all(set(k["order"]) == {"r45", "r47"} for k in key.values())
    assert "r55|" not in "\n".join(items)
    import pytest

    with pytest.raises(SystemExit):
        blind_eval.make(tmp_path / "gen.json", tmp_path, "_x", MODELS + ["r56", "r57"])


def test_parse_judge():
    lines = [
        "```",
        "1 B A",
        "",
        "  ",
        "2 = =",
        "3 - C",
        "解释文字 不是题号",
        "4 D -",
        "```",
    ]
    j = blind_eval.parse_judge(lines)
    assert j[1] == ("B", "A")
    assert j[2] == (None, None)
    assert j[3] == (None, "C")
    assert j[4] == ("D", None)


def test_score_manual():
    key = {
        1: {
            "setting": "T0.8",
            "seed": 0,
            "prompt": "中文题",
            "order": ["r45", "r47", "r55"],
        },
        2: {
            "setting": "T0.8",
            "seed": 0,
            "prompt": "Hello",
            "order": ["r55", "r45", "r47"],
        },
        3: {
            "setting": "greedy",
            "seed": 0,
            "prompt": "中文题",
            "order": ["r47", "r55", "r45"],
        },
    }
    judge = {1: ("A", "C"), 2: ("A", "B"), 3: (None, None), 9: ("A", "B")}
    st = blind_eval.score(judge, key)
    assert st["best"] == {"r45": 1, "r55": 1}
    assert st["worst"] == {"r55": 1, "r45": 1}
    assert st["tie"] == 1
    assert st["groups"] == {
        "zh": {"r45": 1},
        "en": {"r55": 1},
        "T0.8": {"r45": 1, "r55": 1},
    }
    assert st["covered"] == 3 and st["skipped"] == 1
    # 3 个选项却判了 D：跳过不计，不抛 IndexError
    st = blind_eval.score({1: ("D", "A"), 2: ("A", "B")}, key)
    assert st["best"] == {"r55": 1} and st["worst"] == {"r45": 1}
    assert st["covered"] == 1 and st["skipped"] == 1


def test_agree_manual():
    j1 = {1: ("A", "B"), 2: ("A", None), 3: (None, "C"), 4: ("B", "A")}
    j2 = {1: ("A", "C"), 2: ("B", None), 3: ("C", None), 4: ("A", "B")}
    r = blind_eval.agree(j1, j2)
    assert r["best"] == (1, 3)
    assert r["worst"] == (0, 2)
    assert r["opposite"] == 2


def test_mark_and_task(tmp_path):
    fake_gen(tmp_path / "gen.json")
    last = tmp_path / "gen.json"
    _, key, _ = blind_eval.make(tmp_path / "gen.json", tmp_path, "_m", MODELS)
    for name, ws in (("a", "A"), ("b", "C")):
        (tmp_path / f"j_{name}.txt").write_text(
            "".join(f"{i} A {ws}\n" for i in key), encoding="utf-8"
        )
    out = tmp_path / "gen_all_m_judged.txt"
    rc = blind_eval.main(
        [
            "mark",
            str(last),
            "--key",
            str(tmp_path / "blind_key_m.json"),
            "--judge",
            f"a={tmp_path / 'j_a.txt'}",
            "--judge",
            f"b={tmp_path / 'j_b.txt'}",
            "--out",
            str(out),
            "--header",
            "说明行",
        ]
    )
    assert rc == 0
    text = out.read_text(encoding="utf-8-sig")
    assert text.startswith("说明行")
    assert "【a好】" in text and "【a差】" in text and "【b差】" in text
    assert text.count("【a好】") == len(key)
    first = next(iter(sorted(key)))
    assert f"#{first} " in text and "seed" in text

    items = tmp_path / "blind_items_m.txt"
    t1 = tmp_path / "task1.md"
    blind_eval.main(["task", "--items", str(items), "--out", str(t1)])
    s1 = t1.read_text(encoding="utf-8")
    assert str(items.resolve()) in s1 and "空格" not in s1 and "1 epoch" not in s1
    assert "8k 词表、训练轮数很少" in s1
    t2 = tmp_path / "task2.md"
    blind_eval.main(
        ["task", "--items", str(items), "--out", str(t2), "--note", "自定义说明"]
    )
    s2 = t2.read_text(encoding="utf-8")
    assert "自定义说明" in s2 and "空格" not in s2
