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

lines = []
def log(s=""):
    print(s); lines.append(s)

log(f"设备: {DEVICE}")
log(f"提示词数: {len(PROMPTS)}，解码方式: 基线 top-k (generate_text) 与 IGMCG (generate_igmcg)")
log("=" * 78)

stats = {"base_topk": [], "enh_topk": [], "base_igmcg": [], "enh_igmcg": []}
for i, p in enumerate(PROMPTS):
    # 基线 top-k
    t0 = time.time(); b_topk = generate_text(base_m, base_v, p, max_length=60, temperature=0.8, top_k=50, device=DEVICE); t1 = time.time()
    b_topk_tok = max(1, len(base_v.encode(b_topk)) - len(base_v.encode(p))); b_topk_tps = b_topk_tok/(t1-t0); stats["base_topk"].append(b_topk_tps)
    # 增强 top-k
    t0 = time.time(); e_topk = generate_text(enh_m, enh_v, p, max_length=60, temperature=0.8, top_k=50, device=DEVICE); t1 = time.time()
    e_topk_tok = max(1, len(enh_v.encode(e_topk)) - len(enh_v.encode(p))); e_topk_tps = e_topk_tok/(t1-t0); stats["enh_topk"].append(e_topk_tps)
    # 基线 IGMCG
    t0 = time.time(); b_ig = generate_igmcg(base_m, base_v, p, intuition=INTUITION, num_candidates=4, max_length=60, device=DEVICE, base_temp=0.7, top_k=30)[0]; t1 = time.time()
    b_ig_tok = max(1, len(base_v.encode(b_ig)) - len(base_v.encode(p))); b_ig_tps = b_ig_tok/(t1-t0); stats["base_igmcg"].append(b_ig_tps)
    # 增强 IGMCG
    t0 = time.time(); e_ig = generate_igmcg(enh_m, enh_v, p, intuition=INTUITION, num_candidates=4, max_length=60, device=DEVICE, base_temp=0.7, top_k=30)[0]; t1 = time.time()
    e_ig_tok = max(1, len(enh_v.encode(e_ig)) - len(enh_v.encode(p))); e_ig_tps = e_ig_tok/(t1-t0); stats["enh_igmcg"].append(e_ig_tps)

    log(f"\n提示词 {i+1}: {p}")
    log(f"  [BASE top-k ] {b_topk_tok} tok/{t1-t0:.2f}s => {b_topk_tps:.1f} tok/s")
    log(f"      {b_topk}")
    log(f"  [ENH  top-k ] {e_topk_tok} tok/{t1-t0:.2f}s => {e_topk_tps:.1f} tok/s")
    log(f"      {e_topk}")
    log(f"  [BASE IGMCG] {b_ig_tok} tok/{t1-t0:.2f}s => {b_ig_tps:.1f} tok/s")
    log(f"      {b_ig}")
    log(f"  [ENH  IGMCG] {e_ig_tok} tok/{t1-t0:.2f}s => {e_ig_tps:.1f} tok/s")
    log(f"      {e_ig}")

def avg(x): return sum(x)/len(x)
log("\n" + "=" * 78)
log("【训练对比】 BASE: Val Loss 7.7210, ~3275 tok/s | ENH: Val Loss 7.2240, ~2374 tok/s")
log("【生成速度 tok/s（均值）】")
log(f"  top-k : BASE {avg(stats['base_topk']):.1f}  vs  ENH {avg(stats['enh_topk']):.1f}")
log(f"  IGMCG : BASE {avg(stats['base_igmcg']):.1f}  vs  ENH {avg(stats['enh_igmcg']):.1f}")
log(f"  ENH/BASE 速度比: top-k {avg(stats['enh_topk'])/avg(stats['base_topk']):.2f}x, IGMCG {avg(stats['enh_igmcg'])/avg(stats['base_igmcg']):.2f}x")

with open("experiments/cmp_enh_base.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
log("\n原始结果已写入 experiments/cmp_enh_base.txt")
