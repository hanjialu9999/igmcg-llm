import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_pipeline


def test_eval_jobs():
    jobs = eval_pipeline.eval_jobs("r55")
    assert len(jobs) == 4
    assert [n for n, _ in jobs] == [
        "r55_on_packval.json",
        "r55_off_packval.json",
        "r55_3mode_240.json",
        "r55off_3mode_240.json",
    ]
    on = jobs[0][1]
    assert on[:2] == [sys.executable, "scripts/baseline_eval.py"]  # Windows 不能直接执行 .py
    assert "configs/config_train_8k_r55.yaml" in on
    assert "checkpoints_train_8k_r55/final_model.pt" in on
    assert on[on.index("--controller") + 1] == "on"
    assert on[-2:] == ["--out", "baselines/pack_20261005/r55_on_packval.json"]
    assert on[on.index("--modes") + 1] == "tf"
    three = jobs[2][1]
    assert three[three.index("--max-seqs") + 1] == "240"
    assert three[three.index("--modes") + 1] == "tf,prefix,incremental"
    assert jobs[1][1][jobs[1][1].index("--controller") + 1] == "off"
    assert jobs[3][1][jobs[3][1].index("--controller") + 1] == "off"


def ns(**kw):
    d = dict(
        models=["r55"],
        compare=["r47"],
        work_dir="w",
        tag=None,
        gen_out=None,
        skip_eval=False,
        force=False,
        dry_run=False,
    )
    d.update(kw)
    return argparse.Namespace(**d)


def test_plan_tag_and_gen_out():
    pl = eval_pipeline.plan(ns())
    assert pl["tag"] == "_r55"
    assert pl["gen_out"] == "baselines/pack_20261005/r47_r55_gen.json"
    assert pl["gen_cmd"][1] == "scripts/gen_cmp.py"
    assert pl["gen_cmd"][2] == "baselines/pack_20261005/r47_r55_gen.json"
    assert pl["gen_cmd"][3:] == ["r47", "r55"]
    assert pl["blind_models"] == ["r45", "r47", "r55"]


def test_plan_drops_base_when_five():
    pl = eval_pipeline.plan(ns(models=["r55", "r54", "r56"], compare=["r47"]))
    assert pl["blind_models"] == ["r47", "r55", "r54", "r56"]


def test_dry_run_no_subprocess(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("--dry-run 不应起子进程")

    monkeypatch.setattr(eval_pipeline.subprocess, "run", boom)
    rc = eval_pipeline.main(
        ["r55", "--compare", "r47", "--work-dir", str(tmp_path / "w"), "--dry-run"]
    )
    assert rc == 0
    assert not (tmp_path / "w").exists()
