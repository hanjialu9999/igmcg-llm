import sys, time, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate import load_model, generate_text, generate_igmcg, get_device

DEVICE = get_device("dml")
PROMPTS = ["中国的首都是", "人工智能的发展", "学习编程需要", "科学技术是第一"]
INTUITION = [0.3, 0.8, 0.5, 0.2, 0.6, 0.4, 0.5]

base_m, base_v = load_model("checkpoints_cmp_base/final_model.pt", "checkpoints_cmp_base/vocab.json", device=DEVICE)
enh_m, enh_v = load_model("checkpoints_cmp_enh/final_model.pt", "checkpoints_cmp_enh/vocab.json", device=DEVICE)
alt_m, alt_v = load_model("checkpoints_cmp_alt/final_model.pt", "checkpoints_cmp_alt/vocab.json", device=DEVICE)
sel_m, sel_v = load_model("checkpoints_cmp_sel/final_model.pt", "checkpoints_cmp_sel/vocab.json", device=DEVICE)

lines = []
def log(s=""):
    print(s); lines.append(s)

log(f"设备: {DEVICE}")
log(f"提示词数: {len(PROMPTS)}，解码方式: 基线 top-k (generate_text) 与 IGMCG (generate_igmcg)")
log(f"四组：BASE(增强关) / ENH(常开) / ALT(整体50%关) / SEL(分段选择性交替)")
log("=" * 78)

models = [("BASE", base_m, base_v), ("ENH", enh_m, enh_v), ("ALT", alt_m, alt_v), ("SEL", sel_m, sel_v)]
stats = {f"{n}_topk": [] for n, _, _ in models}
stats.update({f"{n}_igmcg": [] for n, _, _ in models})
for i, p in enumerate(PROMPTS):
    log(f"\n提示词 {i+1}: {p}")
    for name, m, v in models:
        t0 = time.time(); out = generate_text(m, v, p, max_length=60, temperature=0.8, top_k=50, device=DEVICE); t1 = time.time()
        tok = max(1, len(v.encode(out)) - len(v.encode(p))); tps = tok/(t1-t0); stats[f"{name}_topk"].append(tps)
        log(f"  [{name} top-k ] {tok} tok/{t1-t0:.2f}s => {tps:.1f} tok/s")
        log(f"      {out}")
        t0 = time.time(); ig = generate_igmcg(m, v, p, intuition=INTUITION, num_candidates=4, max_length=60, device=DEVICE, base_temp=0.7, top_k=30)[0]; t1 = time.time()
        tok = max(1, len(v.encode(ig)) - len(v.encode(p))); tps = tok/(t1-t0); stats[f"{name}_igmcg"].append(tps)
        log(f"  [{name} IGMCG] {tok} tok/{t1-t0:.2f}s => {tps:.1f} tok/s")
        log(f"      {ig}")

def avg(x): return sum(x)/len(x)
log("\n" + "=" * 78)
log("【训练对比（8000 行×1 epoch, DML fp32, 含②检查点优化）】")
log("  BASE(关): Val 7.8300, ~3735 tok/s")
log("  ENH (常开): Val 7.1021, ~3424 tok/s")
log("  ALT (整体 off=0.5): Val 7.3543, ~3596 tok/s  <- 最坏交替方案，已弃用")
log("  SEL (分段选择性交替): Val 7.2254, ~3413 tok/s  <- 采用：质量接近 ENH，优于 ALT")
log("【生成速度 tok/s（均值）】")
for name, _, _ in models:
    log(f"  {name:4s}: top-k {avg(stats[name+'_topk']):.1f}  IGMCG {avg(stats[name+'_igmcg']):.1f}")
log(f"  ENH/BASE 速度比: top-k {avg(stats['ENH_topk'])/avg(stats['BASE_topk']):.2f}x, IGMCG {avg(stats['ENH_igmcg'])/avg(stats['BASE_igmcg']):.2f}x")
log(f"  SEL/BASE 速度比: top-k {avg(stats['SEL_topk'])/avg(stats['BASE_topk']):.2f}x, IGMCG {avg(stats['SEL_igmcg'])/avg(stats['BASE_igmcg']):.2f}x")

with open("experiments/cmp_4way.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
log("\n原始结果（含生成原文）已写入 experiments/cmp_4way.txt")
