"""生成对比：各模型 Controller 开/关 × 12 提示词 × (greedy 42 + T0.8/T1.1 各 seed 42,0..3)，CPU。
用法: python scripts/gen_cmp.py out.json r47 r48 ...
      --base 模型先重生成 greedy 开 12 条，与 --ref 比对确定性。"""

import json, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def strip_echo(text, prompt, vocab):
    """截掉回显的提示词；英文提示会被分词器归一化（"Hello, how are you" → "hello,howareyou"），两种回显都截。"""
    tk = vocab.encode(prompt)
    if tk and tk[-1] == vocab.eos_idx:
        tk = tk[:-1]
    for pre in (prompt, vocab.decode(tk).strip()):
        if pre and text.startswith(pre):
            return text[len(pre) :]
    return text


def metrics(ss):
    cs = [s["cont"] for s in ss]
    ch = "".join(cs)
    bg = [c[i : i + 2] for c in cs for i in range(len(c) - 1)]
    return {
        "n": len(cs),
        "avg_chars": round(sum(map(len, cs)) / len(cs), 1),
        "distinct1": round(len(set(ch)) / max(1, len(ch)), 4),
        "distinct2": round(len(set(bg)) / max(1, len(bg)), 4),
    }


def setup(name, ckpt_fmt="checkpoints_train_8k_{}"):
    for p in (str(ROOT), str(ROOT / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from models.checkpoint import load_model
    from baseline_eval import apply_value_relative_safe_pow

    ck = ROOT / ckpt_fmt.format(name)
    m, v = load_model(str(ck / "final_model.pt"), str(ck / "vocab.json"), device="cpu")
    m.eval()
    m.set_enhancements_active(True)
    if getattr(m, "char_merge_enabled", False):
        m.char_merge.incremental_buffer = True
    apply_value_relative_safe_pow(m, True)
    return m, v


def gen(m, v, ctrl, setting, temp, seed, p):
    import numpy as np, torch
    from scripts.generate import generate_text
    from baseline_eval import GEN_PARAMS

    if getattr(m, "controller_enabled", False):
        m._rt_controller = ctrl == "on"
    if getattr(m, "char_merge_enabled", False):
        m.char_merge.reset_buffer()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    kw = dict(GEN_PARAMS)
    kw["temperature"] = temp
    with torch.no_grad():
        t = generate_text(m, v, p, device="cpu", **kw)
    return strip_echo(t, p, v)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="多模型生成对比")
    ap.add_argument("out")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--ref", default="baselines/pack_20261005/r45_r46_gen.json")
    ap.add_argument(
        "--base", default="r45", help="从 ref 取样本并重生成 greedy 开比对确定性"
    )
    ap.add_argument("--ckpt-fmt", default="checkpoints_train_8k_{}")
    a = ap.parse_args(argv)
    rp = Path(a.ref)
    rp = rp if rp.is_absolute() else ROOT / rp
    ref = json.load(open(rp, encoding="utf-8"))
    PROMPTS, SETS = ref["prompts"], ref["sets"]
    m, v = setup(a.base, a.ckpt_fmt)
    from baseline_eval import GEN_PARAMS

    old = {
        s["prompt"]: s["cont"]
        for s in ref["samples"]
        if s["model"] == a.base and s["controller"] == "on" and s["setting"] == "greedy"
    }
    same = sum(gen(m, v, "on", "greedy", 0.0, 42, p) == old[p] for p in PROMPTS)
    print(f"{a.base} greedy 重生成一致 {same}/{len(PROMPTS)}", flush=True)
    samples = [dict(s) for s in ref["samples"] if s["model"] == a.base]
    for n in a.models:
        t0 = time.time()
        m, v = setup(n, a.ckpt_fmt)
        for ctrl in ("on", "off"):
            for setting, temp, seeds in SETS:
                for sd in seeds:
                    for p in PROMPTS:
                        c = gen(m, v, ctrl, setting, temp, sd, p)
                        samples.append(
                            {
                                "model": n,
                                "controller": ctrl,
                                "setting": setting,
                                "seed": sd,
                                "prompt": p,
                                "text": p + c,
                                "cont": c,
                            }
                        )
        print(f"{n} 完成 {round(time.time() - t0)}s", flush=True)
    met = {}
    for s in samples:
        met.setdefault(f"{s['model']}-{s['controller']}/{s['setting']}", []).append(s)
    met = {k: metrics(x) for k, x in met.items()}
    json.dump(
        {
            "params": GEN_PARAMS,
            "prompts": PROMPTS,
            "sets": SETS,
            "r45_greedy_repro": f"{same}/{len(PROMPTS)}",
            "metrics": met,
            "samples": samples,
        },
        open(a.out, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=1,
    )
    for k, x in met.items():
        print(k, x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
