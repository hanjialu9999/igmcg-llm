"""盲评工具（只用标准库）。
make  gen.json --out-dir D [--tag _r55] [--models r45 r47 r55]
task  --items PATH --out PATH [--note 文本]
score JUDGE.txt --key KEY.json
agree J1 J2 --key KEY.json
mark  GEN.json --key KEY.json --judge 名=路径 [--judge ...] --out PATH [--header 文本]
"""

import argparse, collections, json, random
from pathlib import Path

LETTERS = "ABCD"


def build_items(gen, models, seed=20261019):
    """返回 (items 文本列表, key dict, 揭盲 full 文本列表)；只取 controller=on。"""
    keep = set(models)
    by = collections.defaultdict(dict)
    for s in gen["samples"]:
        if s["controller"] == "on" and s["model"] in keep:
            by[(s["setting"], s["seed"], s["prompt"])][s["model"]] = s["cont"]
    prompts = gen["prompts"]
    rng = random.Random(seed)
    items, key, full = [], {}, []
    for i, (k, v) in enumerate(
        sorted(
            by.items(), key=lambda kv: (kv[0][0], kv[0][1], prompts.index(kv[0][2]))
        ),
        1,
    ):
        order = models[:]
        rng.shuffle(order)
        key[i] = {"setting": k[0], "seed": k[1], "prompt": k[2], "order": order}
        items.append(
            f"#{i} 【{k[2]}】"
            + "".join(f"\n  {'ABCD'[j]}: {v[m]}" for j, m in enumerate(order))
        )
        full.append(
            f"#{i} {k[0]} seed{k[1]} 【{k[2]}】"
            + "".join(f"\n  {m}: {v[m]}" for m in models)
        )
    return items, key, full


def make(gen_path, out_dir, tag="", models=None):
    g = json.load(open(gen_path, encoding="utf-8"))
    models = list(models) if models else sorted({s["model"] for s in g["samples"]})
    if not 2 <= len(models) <= 4:
        raise SystemExit(f"模型数必须 2-4 个，实际 {len(models)}: {models}")
    items, key, full = build_items(g, models)
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"blind_items{tag}.txt").write_text("\n".join(items) + "\n", encoding="utf-8")
    (d / f"blind_key{tag}.json").write_text(
        json.dumps(key, ensure_ascii=False), encoding="utf-8"
    )
    (d / f"gen_all{tag}.txt").write_text("\n".join(full) + "\n", encoding="utf-8-sig")
    print(len(items), "题", models)
    return items, key, full


def parse_judge(lines):
    """{题号: (最好字母或 None, 最差字母或 None)}；`= =` -> (None, None)，`-` -> None。"""
    res = {}
    for line in lines:
        s = line.strip()
        if s.startswith("```"):
            continue
        p = s.split()
        if len(p) < 3 or not p[0].isdigit():
            continue
        b, w = p[1].upper(), p[2].upper()
        if not all(x in ("=", "-") or (len(x) == 1 and x in LETTERS) for x in (b, w)):
            continue
        res[int(p[0])] = (None if b in "=-" else b, None if w in "=-" else w)
    return res


def score(judge, key):
    """judge: parse_judge 的结果，key: {题号: {...}}；返回统计 dict。"""
    best, worst, tie = collections.Counter(), collections.Counter(), 0
    grp = collections.defaultdict(collections.Counter)
    covered = skipped = 0
    for i in sorted(judge):
        k = key.get(i)
        b, w = judge[i]
        n = len(k["order"]) if k else 0
        if k is None or any(x and LETTERS.index(x) >= n for x in (b, w)):
            skipped += 1  # 题号不在 key 里，或字母超出本题选项数
            continue
        covered += 1
        if b is None and w is None:
            tie += 1
            continue
        wm = None if w is None else k["order"][LETTERS.index(w)]
        if wm:
            worst[wm] += 1
        if b is None:
            continue
        bm = k["order"][LETTERS.index(b)]
        best[bm] += 1
        zh = "zh" if any("\u4e00" <= c <= "\u9fff" for c in k["prompt"]) else "en"
        for gname in (zh, k["setting"]):
            grp[gname][bm] += 1
    return {
        "best": dict(best),
        "worst": dict(worst),
        "tie": tie,
        "groups": {g: dict(c) for g, c in grp.items()},
        "covered": covered,
        "skipped": skipped,
    }


def agree(j1, j2):
    """两裁判一致率：最好一致 a/n1、最差一致 b/n2、相反数。"""
    a = n1 = b = n2 = opp = 0
    for i in set(j1) & set(j2):
        b1, w1 = j1[i]
        b2, w2 = j2[i]
        if b1 and b2:
            n1 += 1
            a += b1 == b2
        if w1 and w2:
            n2 += 1
            b += w1 == w2
        if (b1 and w2 and b1 == w2) or (w1 and b2 and w1 == b2):
            opp += 1
    return {"best": (a, n1), "worst": (b, n2), "opposite": opp}


def task_text(items_path, note=None):
    lines = [
        "# 任务：小语言模型续写盲评（只读，不写文件）",
        "",
        f"读取文件 `{Path(items_path).resolve()}`（UTF-8）。",
        "里面每题格式：`#题号 【提示词】`，下面若干行 `  A: 续写`、`  B: 续写`……（同一提示词、同一解码设置下，不同模型的续写，字母顺序已随机打乱）。",
        "这些都是很小的模型（8k 词表、训练轮数很少），整体都不连贯，属正常；请比较**相对**好坏。",
    ]
    if note:
        lines.append(note)
    lines += [
        "",
        "判据（按重要性）：",
        "1. 通顺：中文是否成词成句，有无乱码、无意义的拉丁字母 / 符号堆砌、同一字词机械重复；",
        "2. 切题：是否接着提示词往下说；",
        "3. 信息量：同样通顺时，内容更具体、不空洞的更好。",
        "",
        "对每一题选出**最好**和**最差**的一个字母；如果所有选项确实差不多分不出，写 `= =`。",
        "",
        "输出格式（严格遵守，只输出这些行，不要任何解释、表格或代码块）：每题一行 `题号 最好 最差`，例如",
        "```",
        "1 B A",
        "2 = =",
        "3 C B",
        "```",
        "必须覆盖文件里的全部题号，不能跳题；不要猜测选项来自哪个模型。",
        "",
    ]
    return "\n".join(lines)


def write_task(items_path, out_path, note=None):
    Path(out_path).write_text(task_text(items_path, note), encoding="utf-8")
    return out_path


def mark(gen_path, key_path, judges, out_path, header=None):
    """judges: [(名, 裁判文件路径)]；输出带标记原文（utf-8-sig）。"""
    g = json.load(open(gen_path, encoding="utf-8"))
    key = {int(k): v for k, v in json.load(open(key_path, encoding="utf-8")).items()}
    models = sorted(
        {m for k in key.values() for m in k["order"]}
    )  # 以 key 里的盲评模型为准
    cont = {
        (s["setting"], s["seed"], s["prompt"], s["model"]): s["cont"]
        for s in g["samples"]
        if s["controller"] == "on"
    }
    voted = [
        (name, parse_judge(Path(p).read_text(encoding="utf-8").splitlines()))
        for name, p in judges
    ]
    lines = []
    if header:
        lines.append(header)
    for name, j in voted:
        st = score(j, key)
        grp = "；".join(f"{g} 最好 {c}" for g, c in sorted(st["groups"].items()))
        lines.append(
            f"计数（{len(key)} 题）：{name} 最好 {st['best']}，最差 {st['worst']}，平 {st['tie']}；{grp}。"
        )
    if len(voted) == 2:
        r = agree(voted[0][1], voted[1][1])
        lines.append(
            f"两裁判：最好一致 {r['best'][0]}/{r['best'][1]}，最差一致 {r['worst'][0]}/{r['worst'][1]}，相反 {r['opposite']}。"
        )
    lines.append("")
    for i in sorted(key):
        k = key[i]
        lines.append(f"#{i} {k['setting']} seed{k['seed']} 【{k['prompt']}】")
        for m in models:
            tail = ""
            for name, j in voted:
                b, w = j.get(i, (None, None))
                letter = LETTERS[k["order"].index(m)]
                if b == letter:
                    tail += f"  【{name}好】"
                if w == letter:
                    tail += f"  【{name}差】"
            lines.append(
                f"  {m}: {cont[(k['setting'], k['seed'], k['prompt'], m)]}{tail}"
            )
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return lines


def load_key(path):
    return {int(k): v for k, v in json.load(open(path, encoding="utf-8")).items()}


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="blind_eval.py", description="盲评：出题 / 裁判任务 / 统计 / 一致率 / 标记"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make")
    p.add_argument("gen")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--models", nargs="+")
    p = sub.add_parser("task")
    p.add_argument("--items", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--note")
    p = sub.add_parser("score")
    p.add_argument("judge")
    p.add_argument("--key", required=True)
    p = sub.add_parser("agree")
    p.add_argument("j1")
    p.add_argument("j2")
    p.add_argument("--key", required=True)
    p = sub.add_parser("mark")
    p.add_argument("gen")
    p.add_argument("--key", required=True)
    p.add_argument("--judge", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--header")
    a = ap.parse_args(argv)

    if a.cmd == "make":
        make(a.gen, a.out_dir, a.tag, a.models)
    elif a.cmd == "task":
        write_task(a.items, a.out, a.note)
        print("写", a.out)
    elif a.cmd == "score":
        key = load_key(a.key)
        st = score(
            parse_judge(Path(a.judge).read_text(encoding="utf-8").splitlines()), key
        )
        print("最好", st["best"], "最差", st["worst"], "平", st["tie"])
        for gname, c in sorted(st["groups"].items()):
            print(gname, "最好", c)
        print(f"覆盖 {st['covered']}/{len(key)}")
        if st["skipped"]:
            print(f"跳过 {st['skipped']} 行")
    elif a.cmd == "agree":
        key = load_key(a.key)
        j = lambda p: {
            i: v
            for i, v in parse_judge(
                Path(p).read_text(encoding="utf-8").splitlines()
            ).items()
            if i in key
        }
        r = agree(j(a.j1), j(a.j2))
        print(
            f"最好一致 {r['best'][0]}/{r['best'][1]} 最差一致 {r['worst'][0]}/{r['worst'][1]} 相反 {r['opposite']}"
        )
    elif a.cmd == "mark":
        judges = []
        for spec in a.judge:
            name, _, path = spec.partition("=")
            if not path:
                ap.error(f"--judge 需要 名=路径，收到 {spec!r}")
            judges.append((name, path))
        mark(a.gen, a.key, judges, a.out, a.header)
        print("写", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
