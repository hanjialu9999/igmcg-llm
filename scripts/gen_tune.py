"""生成端调参：默认 r47 开 Controller × 8 中文提示词 × seed 0..7，每组 4 个解码变体（单因素，基线 = baseline_eval.GEN_PARAMS）。
用法: python scripts/gen_tune.py out.json [模型名] [--ckpt-fmt checkpoints_train_8k_{}] [--ref ...]"""

import json, random, sys, time, itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARS = {
    "base": {},
    "top_p0.9": {"top_p": 0.9},
    "rep2.0": {"repetition_penalty": 2.0},
    "eos0": {"eos_penalty": 0.0},
}
SEEDS = list(range(8))


def gen(m, v, kw, seed, p):
    import numpy as np, torch
    from scripts.generate import generate_text

    if getattr(m, "char_merge_enabled", False):
        m.char_merge.reset_buffer()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with torch.no_grad():
        t = generate_text(m, v, p, device="cpu", **kw)
    return t[len(p) :] if t.startswith(p) else t


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="生成端解码参数单因素对照")
    ap.add_argument("out")
    ap.add_argument("model", nargs="?", default="r47")
    ap.add_argument("--ckpt-fmt", default="checkpoints_train_8k_{}")
    ap.add_argument("--ref", default="baselines/pack_20261005/r45_r46_gen.json")
    a = ap.parse_args(argv)
    for p in (str(ROOT), str(ROOT / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from models.checkpoint import load_model
    from baseline_eval import GEN_PARAMS, apply_value_relative_safe_pow

    rp = Path(a.ref)
    rp = rp if rp.is_absolute() else ROOT / rp
    ref = json.load(open(rp, encoding="utf-8"))
    PROMPTS = [p for p in ref["prompts"] if any("\u4e00" <= c <= "\u9fff" for c in p)]
    name = a.model
    ck = ROOT / a.ckpt_fmt.format(name)
    m, v = load_model(str(ck / "final_model.pt"), str(ck / "vocab.json"), device="cpu")
    m.eval()
    m.set_enhancements_active(True)
    if getattr(m, "char_merge_enabled", False):
        m.char_merge.incremental_buffer = True
    apply_value_relative_safe_pow(m, True)
    if getattr(m, "controller_enabled", False):
        m._rt_controller = True

    t0 = time.time()
    samples = []
    for vn, delta in VARS.items():
        kw = dict(GEN_PARAMS)
        kw.update(delta)
        for sd in SEEDS:
            for p in PROMPTS:
                c = gen(m, v, kw, sd, p)
                samples.append(
                    {
                        "model": name,
                        "controller": "on",
                        "setting": vn,
                        "seed": sd,
                        "prompt": p,
                        "cont": c,
                    }
                )
    print("生成", len(samples), "条", round(time.time() - t0), "s", flush=True)

    def bg(c):
        return {c[i : i + 2] for i in range(len(c) - 1)}

    met = {}
    for vn in VARS:
        ss = [s for s in samples if s["setting"] == vn]
        cs = [s["cont"] for s in ss]
        ch = "".join(cs)
        allbg = [c[i : i + 2] for c in cs for i in range(len(c) - 1)]
        # 同 seed 不同提示词之间的续写雷同度（bigram Jaccard 均值；越高越「不看提示词」）
        sims = []
        for sd in SEEDS:
            g = [bg(s["cont"]) for s in ss if s["seed"] == sd]
            sims += [
                len(x & y) / max(1, len(x | y)) for x, y in itertools.combinations(g, 2)
            ]
        met[vn] = {
            "n": len(cs),
            "avg_chars": round(sum(map(len, cs)) / len(cs), 1),
            "short_lt25": sum(len(c) < 25 for c in cs),
            "distinct1": round(len(set(ch)) / max(1, len(ch)), 4),
            "distinct2": round(len(set(allbg)) / max(1, len(allbg)), 4),
            "cross_prompt_sim": round(sum(sims) / len(sims), 4),
        }
        print(vn, met[vn], flush=True)
    json.dump(
        {
            "model": name,
            "base_params": GEN_PARAMS,
            "vars": VARS,
            "prompts": PROMPTS,
            "seeds": SEEDS,
            "metrics": met,
            "samples": samples,
        },
        open(a.out, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
