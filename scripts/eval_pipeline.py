"""一键评测流程：4 个 CPU 评测 -> 生成对比 -> 盲评出题 -> 裁判任务。
用法: python scripts/eval_pipeline.py r55 [r54s43 ...] --compare r47 [--compare ...] --work-dir D
      [--tag _r55] [--gen-out PATH] [--skip-eval] [--force] [--dry-run]"""

import argparse, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import blind_eval

PACK = "baselines/pack_20261005"
BASE = "r45"


def eval_jobs(name):
    """返回 [(输出文件名, 子进程 argv)]，共 4 个。"""
    ck = f"checkpoints_train_8k_{name}"
    common = [
        "--config",
        f"configs/config_train_8k_{name}.yaml",
        "--model",
        f"{ck}/final_model.pt",
        "--vocab",
        f"{ck}/vocab.json",
        "--device",
        "cpu",
    ]
    tail = [
        "--char-merge-buffer",
        "on",
        "--value-relative-safe-pow",
        "on",
        "--no-gen",
    ]
    specs = [
        (f"{name}_on_packval.json", "tf", "on", []),
        (f"{name}_off_packval.json", "tf", "off", []),
        (
            f"{name}_3mode_240.json",
            "tf,prefix,incremental",
            "on",
            ["--max-seqs", "240"],
        ),
        (
            f"{name}off_3mode_240.json",
            "tf,prefix,incremental",
            "off",
            ["--max-seqs", "240"],
        ),
    ]
    jobs = []
    for fn, modes, ctrl, extra in specs:
        argv = (
            [sys.executable, "scripts/baseline_eval.py"]
            + common
            + ["--modes", modes]
            + extra
            + ["--controller", ctrl]
            + tail
            + ["--out", f"{PACK}/{fn}"]
        )
        jobs.append((fn, argv))
    return jobs


def plan(a):
    """把命令行参数展开成流程计划（纯函数，不落盘、不起进程）。"""
    new = list(a.models)
    cmp_ = list(a.compare)
    tag = a.tag if a.tag is not None else "_" + "_".join(new)
    gen_out = a.gen_out or f"{PACK}/" + "_".join(cmp_ + new) + "_gen.json"
    gen_cmd = [sys.executable, "scripts/gen_cmp.py", gen_out] + cmp_ + new
    models = [BASE] + cmp_ + new
    if len(models) > 4:
        models = cmp_ + new
    if len(models) > 4:
        raise SystemExit(f"盲评模型超过 4 个（去掉 {BASE} 后仍超）：{models}")
    evals = [(n, fn, argv) for n in new for fn, argv in eval_jobs(n)]
    return {
        "new": new,
        "compare": cmp_,
        "tag": tag,
        "gen_out": gen_out,
        "gen_cmd": gen_cmd,
        "blind_models": models,
        "evals": evals,
        "work_dir": Path(a.work_dir),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="评测一键流程")
    ap.add_argument("models", nargs="+", help="新模型，如 r55 r54s43")
    ap.add_argument("--compare", action="append", default=[], help="对比模型，可多次")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--tag", help="盲评文件后缀，缺省 _<新模型>")
    ap.add_argument("--gen-out", help="gen.json 输出路径")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--force", action="store_true", help="已有输出也重跑")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--note", help="写进裁判任务的补充说明（如分词不同导致的空格 / 大小写差异）")
    a = ap.parse_args(argv)
    pl = plan(a)
    D = pl["work_dir"]
    if not a.dry_run:
        D.mkdir(parents=True, exist_ok=True)
    log_path = D / "pipeline.log"

    def log(msg):
        print(msg, flush=True)
        if not a.dry_run:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    def run(step, argv_):
        if a.dry_run:
            print(f"[dry-run] {step}: {' '.join(argv_)}", flush=True)
            return True
        t0 = time.time()
        # 子进程输出被重定向时默认按 GBK 编码，中文日志可能报错，统一 UTF-8
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        rc = subprocess.run(argv_, cwd=str(ROOT), env=env).returncode
        log(f"步骤 {step} rc={rc} 秒={round(time.time() - t0, 1)}")
        return rc == 0

    if a.skip_eval:
        log("步骤 eval 跳过（--skip-eval）")
    else:
        for n, fn, cmd in pl["evals"]:
            if (ROOT / PACK / fn).exists() and not a.force:
                log(f"步骤 eval:{n}/{fn} 跳过（已存在）")
                continue
            if not run(f"eval:{n}/{fn}", cmd):
                return 1

    gen_abs = Path(pl["gen_out"])
    gen_abs = gen_abs if gen_abs.is_absolute() else ROOT / gen_abs
    if gen_abs.exists() and not a.force:
        log(f"步骤 gen_cmp 跳过（已存在 {pl['gen_out']}）")
    elif not run("gen_cmp", pl["gen_cmd"]):
        return 1

    tag = pl["tag"]
    items_p = D / f"blind_items{tag}.txt"
    task_p = D / f"judge_task{tag}.md"
    if a.dry_run:
        print(
            f"[dry-run] blind_make: {pl['gen_out']} -> {D} models={pl['blind_models']}"
        )
        print(f"[dry-run] judge_task: {items_p} -> {task_p}")
    else:
        t0 = time.time()
        try:
            blind_eval.make(gen_abs, D, tag, pl["blind_models"])
        except Exception as e:
            log(f"步骤 blind_make rc=1 秒={round(time.time() - t0, 1)} {e}")
            return 1
        log(f"步骤 blind_make rc=0 秒={round(time.time() - t0, 1)}")
        t0 = time.time()
        blind_eval.write_task(items_p, task_p, a.note)
        log(f"步骤 judge_task rc=0 秒={round(time.time() - t0, 1)}")

    key_p = D / f"blind_key{tag}.json"
    jw, jo = D / f"judge_wb{tag}.txt", D / f"judge_oc{tag}.txt"
    print("下一步：")
    print(f"  1) 把 {task_p} 派给两个裁判，各自输出 {jw}、{jo}")
    print(f"  2) python scripts/blind_eval.py score {jw} --key {key_p}")
    print(f"  3) python scripts/blind_eval.py agree {jw} {jo} --key {key_p}")
    print(
        f"  4) python scripts/blind_eval.py mark {pl['gen_out']} --key {key_p} "
        f"--judge wb={jw} --judge oc={jo} --out {D / f'gen_all{tag}_judged.txt'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
