import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

from scripts.generate import load_model, generate_text, generate_igmcg, get_device
import torch
import torch.nn as nn

DEVICE = get_device("dml")
PROMPTS = ["中国的首都是", "人工智能的发展", "学习编程需要", "科学技术是第一"]
INTUITION = [0.3, 0.8, 0.5, 0.2, 0.6, 0.4, 0.5]

enh_m, enh_v = load_model("checkpoints_cmp_enh_full/final_model.pt", "checkpoints_cmp_enh_full/vocab.json", device=DEVICE)
old_m, old_v = load_model("checkpoints_cmp_sel_full/final_model.pt", "checkpoints_cmp_sel_full/vocab.json", device=DEVICE)
new_m, new_v = load_model("checkpoints_cmp_selv2_full/final_model.pt", "checkpoints_cmp_selv2_full/vocab.json", device=DEVICE)
MODELS = [("ENH(全开)", enh_m, enh_v), ("SEL旧(8段)", old_m, old_v), ("SELv2(全开+全关)", new_m, new_v)]
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

log(f"设备: {DEVICE}  |  全量对比：ENH(全开) vs SEL旧(8段无全关) vs SELv2(全开+全关极端)  [merged.txt 39700 行]")
log("=" * 78)
gen_stats = {name: {"topk": [], "igmcg": []} for name, _, _ in MODELS}
for i, p in enumerate(PROMPTS):
    for name, m, v in MODELS:
        t0 = time.time(); e = generate_text(m, v, p, max_length=60, temperature=0.8, top_k=50, device=DEVICE); t1 = time.time()
        tk = max(1, len(v.encode(e)) - len(v.encode(p))); gen_stats[name]["topk"].append(tk/(t1-t0))
        t0 = time.time(); g = generate_igmcg(m, v, p, intuition=INTUITION, num_candidates=4, max_length=60, device=DEVICE, base_temp=0.7, top_k=30)[0]; t1 = time.time()
        gk = max(1, len(v.encode(g)) - len(v.encode(p))); gen_stats[name]["igmcg"].append(gk/(t1-t0))
        log(f"\n提示词 {i+1}: {p}  [{name}]")
        log(f"  top-k : {e}")
        log(f"  IGMCG : {g}")
def avg(x): return sum(x)/len(x)
log("\n" + "=" * 78)
log("【生成速度 tok/s（均值）】")
for name, _, _ in MODELS:
    log(f"  {name}: top-k {avg(gen_stats[name]['topk']):.1f}  IGMCG {avg(gen_stats[name]['igmcg']):.1f}")

log("\n" + "=" * 78)
log("【鲁棒性探针：self_loss（越低越自洽/稳健）】")
for sec_name, settings in SECTIONS:
    log(f"\n########## {sec_name} ##########")
    for name, m, v in MODELS:
        for setting, kw in settings:
            losses = []
            for p in PROMPTS:
                gen, toks = gen_ids(m, v, p, kw["temperature"], kw["top_k"])
                losses.append(self_loss(m, gen, len(toks)))
            summary[name][setting] = sum(losses)/len(losses)
for sec_name, settings in SECTIONS:
    log(f"\n{sec_name}")
    for setting, _ in settings:
        row = "  ".join(f"{name.split('(')[0]}={summary[name][setting]:.2f}" for name, _, _ in MODELS)
        best = min(summary[name][setting] for name, _, _ in MODELS)
        winner = [name.split('(')[0] for name, _, _ in MODELS if abs(summary[name][setting]-best)<1e-6][0]
        log(f"  {setting:6s} | {row}  -> 最优 {winner}")

log("\n【训练 Val Loss（全量 merged.txt 39700 行, 1 epoch, DML fp32）】")
log("  ENH(全开): 5.3762")
log("  SEL旧(8段无全关): 5.4492")
log("  SELv2(全开+全关): 5.4969  -> 含全关极端后 Val 升、仍最差")

with open("experiments/cmp_sel_full.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
log("\n原始结果已写入 experiments/cmp_sel_full.txt")
