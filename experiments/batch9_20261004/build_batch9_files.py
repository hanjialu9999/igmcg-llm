# -*- coding: utf-8 -*-
"""第九批【零】6：e8a/e8b 按组原样落 baselines/ + 生成 batch9 五件套中的表文件。

- baselines/<w>_seq<N>_ctrl<on|off>_dir<on|off>_cmb1_hook<on|off>_seed<S>.json：
  每组 {"meta","res"} 原样复制，不增删字段。
- experiments/batch9_20261004/e8a.md、e8b.md、loss20.csv、gen28.txt
用法：python experiments/batch9_20261004/build_batch9_files.py
"""

import io
import json
import os

REPO = "F:/Projects/igmcg-llm"
T = "C:/Users/hanji/AppData/Local/Temp/opencode"
B = os.path.join(REPO, "experiments", "batch9_20261004")


def load(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def short_model(m):
    for w in (
        "checkpoints_e_x1",
        "checkpoints_e_x2",
        "checkpoints_e_y1",
        "checkpoints_e_y2",
    ):
        if w in m:
            return w.split("_e_")[1]
    return m


def fname(w, n, ctrl, direction, cmb, hook, seed):
    return "%s_seq%d_ctrl%s_dir%s_cmb%d_hook%s_seed%d.json" % (
        w,
        n,
        ctrl,
        direction,
        cmb,
        hook,
        seed,
    )


def row(k, meta, res):
    r = res
    tf = r["tf"]["loss"]
    pf = r["prefix"]["loss"]
    ic = r["incremental"]["loss"]
    return [
        k,
        short_model(meta.get("model", "")),
        "cpu",
        str(meta.get("seed_config", "")),
        str(meta.get("max_seqs", res["tf"].get("tokens", ""))),
        "controller %s / direction %s / 缓冲 %s"
        % (meta.get("ctrl", ""), meta.get("direction", ""), meta.get("buffer", "")),
        "%.6f" % tf,
        "%.6f" % pf,
        "%.6f" % ic,
        "%+.6f" % (tf - pf),
        "%+.6f" % (pf - ic),
    ]


HEAD = [
    "组",
    "权重",
    "设备",
    "评估集 seed",
    "序列数",
    "开关",
    "tf",
    "prefix",
    "逐字",
    "tf−prefix",
    "prefix−逐字",
]


def md_table(rows):
    L = ["| " + " | ".join(HEAD) + " |", "|" + "|".join(["---"] * len(HEAD)) + "|"]
    for r in rows:
        L.append("| " + " | ".join(r) + " |")
    return "\n".join(L) + "\n"


def main():
    e8a = load(os.path.join(T, "e8a_20261002.json"))
    e8b = load(os.path.join(T, "e8b_20261002.json"))
    names = []
    for src, data in (("e8a_20261002.json", e8a), ("e8b_20261002.json", e8b)):
        for k, v in data.items():
            meta, res = v["meta"], v["res"]
            w = short_model(meta.get("model", ""))
            n = meta.get("max_seqs", 0)
            fn = fname(
                w,
                n,
                meta.get("ctrl", ""),
                meta.get("direction", ""),
                1 if meta.get("buffer") == "on" else 0,
                meta.get("hook", ""),
                meta.get("seed_config", 0),
            )
            with io.open(
                os.path.join(REPO, "baselines", fn), "w", encoding="utf-8"
            ) as f:
                json.dump(v, f, ensure_ascii=False, indent=2)
            names.append((src, k, fn))
    with io.open(os.path.join(B, "_copied_files.txt"), "w", encoding="utf-8") as f:
        for src, k, fn in names:
            f.write("%s <- %s [%s]\n" % (fn, src, k))

    ra = [row(k, v["meta"], v["res"]) for k, v in e8a.items()]
    rb = [row(k, v["meta"], v["res"]) for k, v in e8b.items()]
    note_a = (
        "# e8a：11 组（x2_off_nohook 未跑，故 11 不是 12），CPU，"
        "direction 关 / 缓冲开 / safe_pow 开，24 序列×64=1536 token\n"
        "# 评估集：x1/y1 用 seed42 config 的 val；x2/y2 用 seed43 config 的 val"
        "（是它们自己的训练 val，best val 锚点在此集上）\n"
        "# hook 列：x1 用 hook on（模拟 X 训练态 v 恒 0）+ nohook 对照；"
        "y1/y2 一律 hook off\n\n"
    )
    note_b = (
        "# e8b：6 组，CPU，direction 关 / 缓冲开 / safe_pow 开，"
        "评估集一律 seed42 config 的 val（r42 yaml）\n"
        "# ⚠ 污染：x2_seed42_on / y2_seed42_on 用 seed42 val 前 24 条，"
        "其中 20/24 落在 x2/y2 的 seed43 训练集里（VAL42=800/VAL43=800/"
        "INTER=74，SEED42_FRONT24_IN_TRAIN43=20/24），只贴不比\n"
        "# x1_800_on / y1_800_on 为干净配对（seed42 val，800 序列，51130 token）\n\n"
    )
    with io.open(os.path.join(B, "e8a.md"), "w", encoding="utf-8") as f:
        f.write(note_a + md_table(ra))
    with io.open(os.path.join(B, "e8b.md"), "w", encoding="utf-8") as f:
        f.write(note_b + md_table(rb))

    px = load(os.path.join(T, "prof_x1_20.json"))["pass1_losses"]
    py = load(os.path.join(T, "prof_y1_20.json"))["pass1_losses"]
    qx = load(os.path.join(T, "prof_x2_20.json"))["pass1_losses"]
    qy = load(os.path.join(T, "prof_y2_20.json"))["pass1_losses"]
    with io.open(os.path.join(B, "loss20.csv"), "w", encoding="utf-8") as f:
        f.write(
            "# 前 20 步逐个 loss（tf 口径）。x1/y1：此前曲线；"
            "x2/y2：2026-10-04 DML 重跑参照（见汇报【零】4 说明）\n"
        )
        f.write("step,x1,y1,x2,y2\n")
        for i in range(20):
            f.write("%d,%.6f,%.6f,%.6f,%.6f\n" % (i + 1, px[i], py[i], qx[i], qy[i]))

    ga = load(os.path.join(T, "b8_gen_a.json"))
    gb = load(os.path.join(T, "b8_gen_b.json"))
    with io.open(os.path.join(B, "gen28.txt"), "w", encoding="utf-8") as f:
        for g in (ga, gb):
            gp = g["gen_params"]
            f.write(
                "## tag=%s 权重=checkpoints_train_8k_r42 "
                "controller=%s direction=%s 缓冲=%s safe_pow=%s "
                "seed=%s max_length=%s temperature=%s top_k=%s "
                "repetition_penalty=%s\n"
                % (
                    g["tag"],
                    g["controller"],
                    g["direction"],
                    g["char_merge_incremental_buffer"],
                    g["safe_pow"],
                    g["seed"],
                    gp["max_length"],
                    gp["temperature"],
                    gp["top_k"],
                    gp["repetition_penalty"],
                )
            )
            for e in g["generation"]:
                f.write(
                    "[%s|seed%s] prompt=%s\n%s\n"
                    % (e["mode"], e["seed"], e["prompt"], e["text"])
                )
            f.write("\n")
    print(
        "baselines 新增 %d 个 JSON；e8a %d 行 / e8b %d 行；csv/gen 写完"
        % (len(names), len(ra), len(rb))
    )


if __name__ == "__main__":
    main()
