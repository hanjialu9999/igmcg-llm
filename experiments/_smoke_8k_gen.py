import sys, time, os, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 控制台为 GBK，部分生成字符无法编码；强制 stdout/stderr 走 UTF-8（errors=replace）避免崩溃
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from scripts.generate import load_model, generate_text, generate_igmcg, get_device

CKPT = "checkpoints_smoke_8k/final_model.pt"
VOCAB = "checkpoints_smoke_8k/vocab.json"
DEVICE = get_device("dml")  # 解析为 privateuseone:0

PROMPTS = [
    "中国的首都是",
    "人工智能的发展",
    "学习编程需要",
]

INTUITION = [0.3, 0.8, 0.5, 0.2, 0.6, 0.4, 0.5]  # 7 维直觉权重

model, vocab = load_model(CKPT, VOCAB, device=DEVICE)

lines = []
def log(s=""):
    print(s)
    lines.append(s)

log(f"模型: {CKPT}  设备: {DEVICE}")
log("=" * 70)

base_speeds, igmcg_speeds = [], []
for p in PROMPTS:
    # --- 基线：普通 top-k 生成（不开 IGMCG）---
    t0 = time.time()
    out_base = generate_text(model, vocab, p, max_length=60, temperature=0.8,
                             top_k=50, device=DEVICE)
    t1 = time.time()
    base_gen = max(1, len(vocab.encode(out_base)) - len(vocab.encode(p)))
    base_tps = base_gen / (t1 - t0)
    base_speeds.append(base_tps)

    # --- IGMCG 开 ---
    t0 = time.time()
    out_igmcg = generate_igmcg(model, vocab, p, intuition=INTUITION, num_candidates=4,
                               max_length=60, device=DEVICE, base_temp=0.7, top_k=30)[0]
    t1 = time.time()
    igmcg_gen = max(1, len(vocab.encode(out_igmcg)) - len(vocab.encode(p)))
    igmcg_tps = igmcg_gen / (t1 - t0)
    igmcg_speeds.append(igmcg_tps)

    log(f"\n提示词: {p}")
    log(f"  [基线 top-k]  {base_gen} tok / {t1-t0:.2f}s  => {base_tps:.1f} tok/s")
    log(f"    原文: {out_base}")
    log(f"  [IGMCG 开]    {igmcg_gen} tok / {t1-t0:.2f}s  => {igmcg_tps:.1f} tok/s")
    log(f"    原文: {out_igmcg}")

log("\n" + "=" * 70)
log(f"平均生成速度  基线 top-k : {sum(base_speeds)/len(base_speeds):.1f} tok/s")
log(f"平均生成速度  IGMCG 开  : {sum(igmcg_speeds)/len(igmcg_speeds):.1f} tok/s")
log(f"IGMCG 相对基线 速度比   : {(sum(igmcg_speeds)/len(igmcg_speeds))/(sum(base_speeds)/len(base_speeds)):.2f}x")

with open("experiments/smoke_8k_gen.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
log("\n原始结果已写入 experiments/smoke_8k_gen.txt")
