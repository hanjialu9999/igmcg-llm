# -*- coding: utf-8 -*-
"""第九批【二】1+2：Frobenius 范数。
1) y1 final vs seed42 重建未训练：q/k/v 并列（查 v 是否真没动）。
2) r42 e1->e2：embedding / FFN / norm / char_merge 各自的 ||ΔW||。
只读 state_dict，不推理不训练。
用法：python experiments/batch9_20261004/eval_dw.py
"""

import io
import json
import os
import random
import sys

sys.path.insert(0, "F:/Projects/igmcg-llm")
os.chdir("F:/Projects/igmcg-llm")
import numpy as np
import torch

from models.config_loader import load_config, build_model

OUT = "experiments/batch9_20261004/dw_norms.json"


def load_sd(p):
    ck = torch.load(p, map_location="cpu")
    return ck.get("model_state_dict", ck)


def seg_norm(a, b):
    return float((a.float() - b.float()).norm().item())


def main():
    out = {}
    # ---- 1) y1 vs 未训练 ----
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    cfg = load_config(
        "C:/Users/hanji/AppData/Local/Temp/opencode/e_cfg/config_e_y1.yaml"
    )
    m0 = build_model(cfg, device=torch.device("cpu"))
    sd0 = m0.state_dict()
    y1 = load_sd("checkpoints_e_y1/final_model.pt")
    g1 = {}
    for l in range(4):
        qk = "blocks.%d.attn.qkv.weight" % l
        for seg, sl in (
            ("q", slice(0, 256)),
            ("k", slice(256, 512)),
            ("v", slice(512, 768)),
        ):
            g1["L%d.%s" % (l, seg)] = seg_norm(y1[qk][sl], sd0[qk][sl])
        for nm, kn in (
            ("proj", "blocks.%d.attn.proj.weight" % l),
            ("og_w", "blocks.%d.attn.output_gate.weight" % l),
        ):
            g1["L%d.%s" % (l, nm)] = seg_norm(y1[kn], sd0[kn])
    out["y1_vs_init"] = g1
    for k, v in g1.items():
        print("y1-init %s %.6g" % (k, v), flush=True)
    # ---- 2) r42 e1->e2 按模块 ----
    e1 = load_sd("checkpoints_train_8k_r42/model_epoch_1.pt")
    e2 = load_sd("checkpoints_train_8k_r42/model_epoch_2.pt")
    groups = {"embedding": [], "ffn": [], "norm": [], "char_merge": [], "other": []}
    for k in e1:
        kl = k.lower()
        if ("embed" in kl or "wte" in kl or "token" in kl) and "control" not in kl:
            groups["embedding"].append(k)
        elif "char_merge" in kl or "merge" in kl:
            groups["char_merge"].append(k)
        elif (
            (
                "mlp" in kl
                or "ffn" in kl
                or "fc" in kl
                or "gate" in kl
                or "moe" in kl
                or "expert" in kl
            )
            and "attn" not in kl
            and "control" not in kl
            and "output_gate" not in kl
        ):
            groups["ffn"].append(k)
        elif "norm" in kl or "ln_" in kl or ".ln" in kl:
            groups["norm"].append(k)
        else:
            groups["other"].append(k)
    g2 = {}
    for g, keys in groups.items():
        tot = sum(seg_norm(e1[k], e2[k]) ** 2 for k in keys) ** 0.5
        g2[g] = {"n_keys": len(keys), "frob": tot, "keys": keys}
        print("e1->e2 %-10s n=%d frob=%.6g" % (g, len(keys), tot), flush=True)
    out["r42_e1_e2"] = g2
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("写入 %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
