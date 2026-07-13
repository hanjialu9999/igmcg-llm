import sys, io, math, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

from scripts.generate import load_model, generate_text, generate_igmcg, get_device
import torch
import torch.nn as nn

DEVICE = get_device("dml")
PROMPTS = ["中国的首都是", "人工智能的发展", "学习编程需要", "科学技术是第一"]
INTUITION = [0.3, 0.8, 0.5, 0.2, 0.6, 0.4, 0.5]

enh_m, enh_v = load_model("checkpoints_cmp_enh_20k/final_model.pt", "checkpoints_cmp_enh_20k/vocab.json", device=DEVICE)
sel_m, sel_v = load_model("checkpoints_cmp_sel_20k/final_model.pt", "checkpoints_cmp_sel_20k/vocab.json", device=DEVICE)
MODELS = [("ENH", enh_m, enh_v), ("SEL", sel_m, sel_v)]
criterion = nn.CrossEntropyLoss()

lines = []
def log(s=""):
    print(s); lines.append(s)

def gen_ids(model, vocab, prompt, temperature, top_k, max_length=60):
    tokens = vocab.encode(prompt)
    if tokens and tokens[-1] == vocab.eos_idx:
        tokens = tokens[:-1]
    gen = model.generate(tokens, max_length=max_length, temperature=temperature, top_k=top_k,
                         device=DEVICE, repetition_penalty=1.4, min_length=3, eos_penalty=-5.0)
    return gen, tokens

def self_loss(model, gen_ids, prompt_len):
    cont = gen_ids[prompt_len:]
    if len(cont) < 1:
        return float("nan")
    with torch.no_grad():
        logits = model(torch.tensor([gen_ids], device=DEVICE))
        pred = logits[0, prompt_len-1 : prompt_len-1+len(cont)]
        tgt = torch.tensor(cont, device=DEVICE)
    return criterion(pred, tgt).item()

SECTIONS = [
    ("温度扫描 (top_k=50, rep=1.4)", [("T=0.5", dict(temperature=0.5, top_k=50)),
                                      ("T=0.8", dict(temperature=0.8, top_k=50)),
                                      ("T=1.1", dict(temperature=1.1, top_k=50)),
                                      ("T=1.4", dict(temperature=1.4, top_k=50))]),
    ("top_k 扫描 (T=0.8, rep=1.4)", [("K=10",  dict(temperature=0.8, top_k=10)),
                                     ("K=30",  dict(temperature=0.8, top_k=30)),
                                     ("K=100", dict(temperature=0.8, top_k=100))]),
]
summary = {name: {} for name, _, _ in MODELS}

# ===== 1) 生成对比（top-k / IGMCG，含原文 + 速度）=====
log(f"设备: {DEVICE}  |  20k 模型 ENH(常开) vs SEL(8段选择性交替)  | 提示词 {len(PROMPTS)}")
log("=" * 78)
gen_stats = {"ENH_topk": [], "SEL_topk": [], "ENH_igmcg": [], "SEL_igmcg": []}
for i, p in enumerate(PROMPTS):
    t0 = time.time(); e_topk = generate_text(enh_m, enh_v, p, max_length=60, temperature=0.8, top_k=50, device=DEVICE); t1 = time.time()
    e_tok = max(1, len(enh_v.encode(e_topk)) - len(enh_v.encode(p))); e_tps = e_tok/(t1-t0); gen_stats["ENH_topk"].append(e_tps)
    t0 = time.time(); s_topk = generate_text(sel_m, sel_v, p, max_length=60, temperature=0.8, top_k=50, device=DEVICE); t1 = time.time()
    s_tok = max(1, len(sel_v.encode(s_topk)) - len(sel_v.encode(p))); s_tps = s_tok/(t1-t0); gen_stats["SEL_topk"].append(s_tps)
    t0 = time.time(); e_ig = generate_igmcg(enh_m, enh_v, p, intuition=INTUITION, num_candidates=4, max_length=60, device=DEVICE, base_temp=0.7, top_k=30)[0]; t1 = time.time()
    e_itok = max(1, len(enh_v.encode(e_ig)) - len(enh_v.encode(p))); e_itps = e_itok/(t1-t0); gen_stats["ENH_igmcg"].append(e_itps)
    t0 = time.time(); s_ig = generate_igmcg(sel_m, sel_v, p, intuition=INTUITION, num_candidates=4, max_length=60, device=DEVICE, base_temp=0.7, top_k=30)[0]; t1 = time.time()
    s_itok = max(1, len(sel_v.encode(s_ig)) - len(sel_v.encode(p))); s_itps = s_itok/(t1-t0); gen_stats["SEL_igmcg"].append(s_itps)
    log(f"\n提示词 {i+1}: {p}")
    log(f"  [ENH top-k ] {e_tok} tok/{t1-t0:.2f}s => {e_tps:.1f} tok/s")
    log(f"      {e_topk}")
    log(f"  [SEL top-k ] {s_tok} tok/{t1-t0:.2f}s => {s_tps:.1f} tok/s")
    log(f"      {s_topk}")
    log(f"  [ENH IGMCG] {e_itok} tok/{t1-t0:.2f}s => {e_itps:.1f} tok/s")
    log(f"      {e_ig}")
    log(f"  [SEL IGMCG] {s_itok} tok/{t1-t0:.2f}s => {s_itps:.1f} tok/s")
    log(f"      {s_ig}")

def avg(x): return sum(x)/len(x)
log("\n" + "=" * 78)
log("【生成速度 tok/s（均值）】")
log(f"  top-k : ENH {avg(gen_stats['ENH_topk']):.1f}  SEL {avg(gen_stats['SEL_topk']):.1f}")
log(f"  IGMCG : ENH {avg(gen_stats['ENH_igmcg']):.1f}  SEL {avg(gen_stats['SEL_igmcg']):.1f}")
log(f"  SEL/BASE 速度比: top-k {avg(gen_stats['SEL_topk'])/avg(gen_stats['ENH_topk']):.2f}x, IGMCG {avg(gen_stats['SEL_igmcg'])/avg(gen_stats['ENH_igmcg']):.2f}x")

# ===== 2) 鲁棒性探针（温度/top_k 扫描，self-loss）=====
log("\n" + "=" * 78)
log("【鲁棒性探针：self_loss = 模型对自身续写的 cross-entropy（越低越自洽/稳健）】")
for sec_name, settings in SECTIONS:
    log(f"\n########## {sec_name} ##########")
    for name, m, v in MODELS:
        for setting, kw in settings:
            losses = []
            log(f"\n--- {name} | {setting} ---")
            for p in PROMPTS:
                gen, toks = gen_ids(m, v, p, kw["temperature"], kw["top_k"])
                loss = self_loss(m, gen, len(toks)); losses.append(loss)
                log(f"  [{p}] self_loss={loss:.2f}")
            summary[name][setting] = sum(losses)/len(losses)
for sec_name, settings in SECTIONS:
    log(f"\n{sec_name}")
    for setting, _ in settings:
        enh = summary["ENH"][setting]; sel = summary["SEL"][setting]
        log(f"  {setting:6s} | ENH loss={enh:.2f}   ||   SEL loss={sel:.2f}   -> {'SEL更鲁棒' if sel < enh else 'ENH更鲁棒'}")

log("\n【训练 Val Loss（20k, 1 epoch, DML fp32）】")
log("  ENH(常开): 6.2188 / ~3278 tok/s")
log("  SEL(8段选择性交替): 6.2808 / ~3471 tok/s  -> 质量接近 ENH，训练快约 7%")

with open("experiments/cmp_20k.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
log("\n原始结果（含生成原文）已写入 experiments/cmp_20k.txt")
