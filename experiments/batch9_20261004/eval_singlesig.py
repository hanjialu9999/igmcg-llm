# -*- coding: utf-8 -*-
"""第九批【一】1：y1/x1 单信号开关四组（800 序列、CPU、seed42 val）。

只推理不训练。设开关的三行（controller.py:218-235：False 时信号为 None，
整条移除，不是置零）：
    model.controller.use_direction = False
    model.controller.use_memory_compress = (mem == 'on')
    model.controller.use_film = (film == 'on')
用法：
    python experiments/batch9_20261004/eval_singlesig.py
产物：experiments/batch9_20261004/singlesig_800.json
"""

import json
import os
import sys
import time

sys.path.insert(0, "F:/Projects/igmcg-llm")
os.chdir("F:/Projects/igmcg-llm")
sys.path.insert(0, "C:/Users/hanji/AppData/Local/Temp/opencode")
import torch

import b8_run as B
from models.checkpoint import load_model
from scripts.baseline_eval import (
    eval_tf,
    eval_prefix,
    eval_incremental,
    _finish,
    apply_value_relative_safe_pow,
)

R42 = "configs/config_train_8k_r42.yaml"
OUT = "experiments/batch9_20261004/singlesig_800.json"

GROUPS = [
    # (tag, ckpt, vocab, ctrl, mem, film)
    (
        "y1_800_ctrloff",
        "checkpoints_e_y1/final_model.pt",
        "checkpoints_e_y1/vocab.json",
        "off",
        "on",
        "on",
    ),
    (
        "y1_800_memoff",
        "checkpoints_e_y1/final_model.pt",
        "checkpoints_e_y1/vocab.json",
        "on",
        "off",
        "on",
    ),
    (
        "y1_800_filmoff",
        "checkpoints_e_y1/final_model.pt",
        "checkpoints_e_y1/vocab.json",
        "on",
        "on",
        "off",
    ),
    (
        "x1_800_ctrloff",
        "checkpoints_e_x1/final_model.pt",
        "checkpoints_e_x1/vocab.json",
        "off",
        "on",
        "on",
    ),
]


def run_one(tag, ckpt, vocab_p, ctrl, mem, film):
    device = torch.device("cpu")
    print("=== %s (ctrl=%s mem=%s film=%s) ===" % (tag, ctrl, mem, film), flush=True)
    model, vocab = load_model(ckpt, vocab_p, device=device)
    model.eval()
    model.set_enhancements_active(True)
    if getattr(model, "controller_enabled", False):
        model._rt_controller = ctrl == "on"
    if getattr(model, "controller", None) is not None:
        model.controller.use_direction = False
        model.controller.use_memory_compress = mem == "on"
        model.controller.use_film = film == "on"
    if getattr(model, "char_merge_enabled", False):
        model.char_merge.incremental_buffer = True
        model.char_merge.reset_buffer()
    n = apply_value_relative_safe_pow(model, True)
    print("      safe_pow 作用于 %d 个 mixer" % n, flush=True)
    cfg, loader, nseq = B.build_loader(R42, 800, 24)
    print("      val 前 %d 条 seed=%s" % (nseq, cfg["seed"]), flush=True)
    res = {}
    for name, fn in (
        ("tf", eval_tf),
        ("prefix", eval_prefix),
        ("incremental", eval_incremental),
    ):
        t = time.time()
        model.zero_grad(set_to_none=True)
        r = _finish(fn(model, loader, device, vocab.pad_idx))
        r["elapsed_s"] = round(time.time() - t, 2)
        res[name] = r
        print("      %s: loss=%s (%ss)" % (name, r["loss"], r["elapsed_s"]), flush=True)
    print(
        "  >> %s tf-prefix=%s prefix-inc=%s"
        % (
            tag,
            round(res["tf"]["loss"] - res["prefix"]["loss"], 6),
            round(res["prefix"]["loss"] - res["incremental"]["loss"], 6),
        ),
        flush=True,
    )
    meta = {
        "tag": tag,
        "model": ckpt,
        "config": R42,
        "ctrl": ctrl,
        "direction": "off",
        "mem": mem,
        "film": film,
        "buffer": "on",
        "pow": "on",
        "hook": "off",
        "max_seqs": nseq,
        "batch_size": 24,
        "seed_config": cfg["seed"],
        "device": "cpu",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return meta, res


def main():
    old = {}
    if os.path.exists(OUT):
        old = json.load(io_open(OUT))
    for tag, ckpt, vocab_p, ctrl, mem, film in GROUPS:
        if tag in old:
            print("跳过已存在 %s" % tag, flush=True)
            continue
        meta, res = run_one(tag, ckpt, vocab_p, ctrl, mem, film)
        old[tag] = {"meta": meta, "res": res}
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(old, f, ensure_ascii=False, indent=2)
        print("      写入 %s" % OUT, flush=True)


def io_open(p):
    import io

    return io.open(p, encoding="utf-8")


if __name__ == "__main__":
    main()
