# -*- coding: utf-8 -*-
"""训练单步计时（只读配置；不训练、不存 checkpoint、不改任何已有文件）。

目标：定位上下文 256 时（r58）训练慢在哪一段。

模型构造、设备选择、优化器/梯度裁剪全部复用 scripts/train.py 的实现：
  - models.config_loader.load_config / build_model（scripts/train.py:477、:572 同一入口）
  - models.device.get_device / apply_cpu_threads（scripts/train.py:483-484）
  - models.checkpoint.build_ngram_model（scripts/train.py:570-571）
  - scripts.train.clip_grad_norm_dml / set_seed（scripts/train.py:117、:75）
  - models.utils.optimizer_param_groups（scripts/train.py:659）

计时 A（整步）：forward + loss.item() + backward + 梯度裁剪 + optimizer.step()，
               step 末尾再同步一次设备队列，取中位数 ms；逐个变体对比。
计时 B（模块前向）：对顶层子模块 + 每个 block 内一层挂 forward pre/post hook，
               前后各做一次设备同步，统计前向 mean ms。

用法（workdir = 项目根）：
  python experiments/prof_20261020/prof_step.py --config configs/config_train_8k_r57.yaml --device dml
  python experiments/prof_20261020/prof_step.py --config configs/config_train_8k_r58.yaml --device dml

  # CPU 自测（不跑 DirectML）：
  python experiments/prof_20261020/prof_step.py --config configs/config_train_8k_r57.yaml \
      --device cpu --steps 2 --warmup 1 --out experiments/prof_20261020/selftest_cpu.json
"""

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast as torch_autocast, GradScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.checkpoint import build_ngram_model
from models.config_loader import build_model, load_config
from models.constants import PAD_IDX
from models.device import get_device, apply_cpu_threads
from models.utils import optimizer_param_groups
from scripts import train as T


def parse_args():
    ap = argparse.ArgumentParser(description="训练单步计时（只读，不训练不落盘）")
    ap.add_argument("--config", required=True, help="训练配置 yaml")
    ap.add_argument(
        "--bs",
        type=int,
        default=None,
        help="batch size（缺省读 config.training.batch_size）",
    )
    ap.add_argument("--steps", type=int, default=20, help="每变体计时步数（默认 20）")
    ap.add_argument("--warmup", type=int, default=5, help="每变体预热步数（默认 5）")
    ap.add_argument(
        "--device", choices=["dml", "cpu"], default="dml", help="设备（默认 dml）"
    )
    ap.add_argument("--out", default=None, help="结果 JSON 路径")
    return ap.parse_args()


def patch_dml_lerp(device):
    """与 scripts/train.py:644-657 完全一致的 DirectML AdamW lerp 替换。"""
    if device.type != "privateuseone":
        return

    def _dml_foreach_lerp(self_list, end_list, weight):
        torch._foreach_mul_(self_list, 1 - weight)
        torch._foreach_add_(self_list, [g * weight for g in end_list])
        return self_list

    torch._foreach_lerp_ = _dml_foreach_lerp

    def _dml_lerp(self, end, weight):
        return self.mul_(1 - weight).add_(end * weight)

    torch.Tensor.lerp_ = _dml_lerp


def enabled_enhancement(model, key):
    """判断某个运行时开关在当前配置下是否真的开启（决定是否单独测）。"""
    model_flags = {
        "controller": "controller_enabled",
        "char_merge": "char_merge_enabled",
        "controller_film": "controller_film_enabled",
        "layer_film": "layer_film_enabled",
        "input_highway": "input_highway_enabled",
    }
    if key in model_flags:
        return bool(getattr(model, model_flags[key], False))
    sub_flags = {
        "residual_gate": "residual_gate_enabled",
        "hybrid_gate": "hybrid_gate_enabled",
        "highway_gate": "highway_gate_enabled",
        "qk_norm": "qk_norm_enabled",
        "attn_temp": "temp_enabled",
        "value_relative_coding": "value_relative_coding_enabled",
    }
    attr = sub_flags.get(key)
    if attr is None:
        return False
    if key == "hybrid_gate":
        # block 级 hybrid_gate_enabled 恒等于 gate_cfg.hybrid_gate（与 block_type 无关），
        # 只有 hybrid 块才真的走这条门控，否则测出来是纯噪声。
        return any(
            getattr(blk, "block_type", "") == "hybrid"
            and getattr(blk, "hybrid_gate_enabled", False)
            for blk in model.blocks
        )
    for blk in model.blocks:
        for sub in (blk, getattr(blk, "attn", None)):
            if sub is not None and getattr(sub, attr, False):
                return True
    return False


def collect_modules(model):
    """顶层子模块 + 每个 block 内一层。返回 [(显示名, module)]。"""
    mods = []
    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList):
            for i, blk in enumerate(child):
                mods.append((f"{name}.{i}", blk))
                for sub_name, sub in blk.named_children():
                    mods.append((f"{name}.{i}.{sub_name}", sub))
        else:
            mods.append((name, child))
    return mods


def main():
    args = parse_args()
    config_path = args.config
    config = load_config(config_path)

    T.set_seed(config["seed"])
    device = get_device(args.device)
    apply_cpu_threads(config["training"].get("cpu_threads"))
    print(f"Using device: {device}")

    ngram_model = build_ngram_model(None, config["model"])
    model = build_model(config, device=device, ngram_model=ngram_model)
    model.train()  # 同 scripts/train.py:189

    vocab_size = int(config["model"]["vocab_size"])
    seq_len = int(config["model"]["max_seq_length"])
    bs = int(args.bs) if args.bs else int(config["training"]["batch_size"])

    print(
        f"Model: vocab={vocab_size} T={seq_len} layers={config['model']['num_layers']} "
        f"dim={config['model']['embedding_dim']}"
    )
    print(
        f"Batch size: {bs}  Steps: {args.steps} (warmup {args.warmup})  "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )

    # ---- 损失 / 优化器：照 scripts/train.py:621-688 ----
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    wd = config["training"]["weight_decay"]
    lr = float(config["training"]["learning_rate"])
    opt_name = str(config["training"].get("optimizer", "adamw")).lower()
    params = optimizer_param_groups(model, wd)
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=float(config["training"].get("sgd_learning_rate", lr)),
            momentum=float(config["training"].get("momentum", 0.9)),
            weight_decay=wd,
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            params, lr=lr, weight_decay=wd, betas=(0.9, 0.999), eps=1e-8
        )
    else:
        optimizer = torch.optim.AdamW(
            params, lr=lr, weight_decay=wd, betas=(0.9, 0.999), eps=1e-8
        )
    print(f"Optimizer: {opt_name}  lr={lr}  wd={wd}")
    if opt_name in ("adamw", "adam"):
        patch_dml_lerp(device)

    gradient_clip = float(config["training"].get("gradient_clip", 1.0))
    foreach_norm = bool(config["training"].get("use_foreach_norm_clip", False))

    # ---- 混合精度：照 scripts/train.py:742-765 ----
    precision = str(config["training"].get("precision", "fp32")).lower()
    use_amp = False
    autocast_dtype = torch.float32
    scaler = None
    amp_device = None
    if precision in ("fp16", "bf16"):
        if device.type == "cuda":
            use_amp = True
            amp_device = "cuda"
            autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
            scaler = GradScaler("cuda") if precision == "fp16" else None
        elif device.type == "cpu" and precision == "bf16":
            use_amp = True
            amp_device = "cpu"
            autocast_dtype = torch.bfloat16
    print(f"Precision: {precision} (AMP={use_amp})")

    # ---- 随机 token（形状 bs×(T+1)，与 data.pack 的块形状一致）----
    ids = torch.randint(0, vocab_size, (bs, seq_len + 1), device=device)
    src = ids[:, :-1].contiguous()
    tgt = ids[:, 1:].contiguous()

    sync_buf = torch.zeros(1, device=device)

    def sync():
        return sync_buf.item()

    def forward_loss():
        if use_amp and amp_device is not None:
            with torch_autocast(amp_device, dtype=autocast_dtype):
                logits = model(src, igmcg_force_off=False, targets=tgt).view(
                    -1, vocab_size
                )
                loss = criterion(logits, tgt.view(-1))
        else:
            logits = model(src, igmcg_force_off=False, targets=tgt).view(-1, vocab_size)
            loss = criterion(logits, tgt.view(-1))
        # 与 scripts/train.py:299-316 一致：把训练期累积的辅助损失并入主 loss
        _cl = getattr(model, "_contrastive_loss", None)
        if _cl is not None:
            loss = loss + 0.01 * _cl
        _ee = getattr(model, "_early_exit_aux_loss", None)
        if _ee is not None:
            loss = loss + model.early_exit_loss_weight * _ee
        _mlb = getattr(model, "_moe_load_balance_loss", None)
        if _mlb is not None:
            loss = loss + model.moe_load_balance_weight * _mlb
        _mzl = getattr(model, "_moe_z_loss", None)
        if _mzl is not None:
            loss = loss + model.moe_router_z_loss_weight * _mzl
        return loss

    def run_variant(name, setup):
        model.set_enhancements_active(True)
        setup()
        optimizer.zero_grad(set_to_none=True)
        times = []
        last_loss = float("nan")
        for i in range(args.warmup + args.steps):
            t0 = time.perf_counter()
            loss = forward_loss()
            last_loss = loss.item()  # 前向同步点（同 train.py 的 loss 读取）
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                T.clip_grad_norm_dml(
                    model.parameters(),
                    max_norm=gradient_clip,
                    foreach_norm=foreach_norm,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                T.clip_grad_norm_dml(
                    model.parameters(),
                    max_norm=gradient_clip,
                    foreach_norm=foreach_norm,
                )
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            sync()  # 排空异步队列，整步耗时才算完整
            dt = (time.perf_counter() - t0) * 1000.0
            if i >= args.warmup:
                times.append(dt)
        return {
            "median_ms": statistics.median(times),
            "mean_ms": statistics.fmean(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "loss": last_loss,
            "n": len(times),
        }

    print("\n== 计时 A：整步（forward + backward + clip + step）==")
    variants = []

    r = run_variant("baseline", lambda: None)
    variants.append({"name": "baseline", **r})
    base_med = r["median_ms"]

    r = run_variant(
        "off:controller", lambda: model.set_enhancements_active({"controller": False})
    )
    variants.append({"name": "off:controller", **r})

    single_keys = [
        k
        for k in model.ENHANCEMENT_KEYS
        if k != "controller" and enabled_enhancement(model, k)
    ]
    for key in single_keys:
        r = run_variant(
            f"off:{key}", lambda k=key: model.set_enhancements_active({k: False})
        )
        variants.append({"name": f"off:{key}", **r})

    model.set_enhancements_active(True)

    print(
        f"{'variant':<30}{'median ms':>12}{'delta ms':>12}{'delta %':>10}{'loss':>10}"
    )
    for v in variants:
        d = v["median_ms"] - base_med
        print(
            f"{v['name']:<30}{v['median_ms']:>12.2f}{d:>12.2f}{d / base_med * 100:>9.1f}%{v['loss']:>10.4f}"
        )

    # ---- 计时 B：模块前向 hook ----
    print("\n== 计时 B：模块前向（mean ms，含前后同步开销）==")
    mods = collect_modules(model)
    acc = defaultdict(float)
    cnt = defaultdict(int)
    t_start = {}
    handles = []

    def make_pre(n):
        def pre(_m, _inp):
            sync()
            t_start[n] = time.perf_counter()

        return pre

    def make_post(n):
        def post(_m, _inp, _out):
            sync()
            acc[n] += time.perf_counter() - t_start.pop(n)
            cnt[n] += 1

        return post

    for name, mod in mods:
        handles.append(mod.register_forward_pre_hook(make_pre(name)))
        handles.append(mod.register_forward_hook(make_post(name)))

    model.set_enhancements_active(True)
    try:
        with torch.no_grad():
            for _ in range(args.warmup):
                model(src)
            acc.clear()
            cnt.clear()
            for _ in range(args.steps):
                model(src)
    finally:
        for h in handles:
            h.remove()

    modules = []
    for name, _ in mods:
        if cnt[name]:
            mean_ms = acc[name] / cnt[name] * 1000.0
            modules.append(
                {
                    "name": name,
                    "mean_ms": mean_ms,
                    "total_ms": acc[name] * 1000.0,
                    "n": cnt[name],
                }
            )
    modules.sort(key=lambda x: x["mean_ms"], reverse=True)

    print(f"{'module':<38}{'mean ms':>10}{'total ms':>12}")
    for m in modules[:15]:
        print(f"{m['name']:<38}{m['mean_ms']:>10.2f}{m['total_ms']:>12.2f}")

    out_path = (
        Path(args.out)
        if args.out
        else Path(__file__).resolve().parent / "prof_step_result.json"
    )
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config_path,
        "device": str(device),
        "batch_size": bs,
        "seq_length": seq_len,
        "steps": args.steps,
        "warmup": args.warmup,
        "optimizer": opt_name,
        "precision": precision,
        "params": sum(p.numel() for p in model.parameters()),
        "variants": variants,
        "module_forward_ms": modules,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入: {out_path}")


if __name__ == "__main__":
    main()
