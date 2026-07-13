import sys, io, math
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, sys.path[0] if False else ".")

from scripts.generate import load_model, get_device
import torch
import torch.nn as nn

DEVICE = get_device("dml")
PROMPTS = ["中国的首都是", "人工智能的发展", "学习编程需要", "科学技术是第一",
           "今天天气很好", "我喜欢读书因为"]

enh_m, enh_v = load_model("checkpoints_cmp_enh/final_model.pt", "checkpoints_cmp_enh/vocab.json", device=DEVICE)
sel_m, sel_v = load_model("checkpoints_cmp_sel/final_model.pt", "checkpoints_cmp_sel/vocab.json", device=DEVICE)
MODELS = [("ENH", enh_m, enh_v), ("SEL", sel_m, sel_v)]
criterion = nn.CrossEntropyLoss()

lines = []
def log(s=""):
    print(s); lines.append(s)

def generate_ids(model, vocab, prompt, temperature, top_k, max_length=60):
    tokens = vocab.encode(prompt)
    if tokens and tokens[-1] == vocab.eos_idx:
        tokens = tokens[:-1]
    gen = model.generate(tokens, max_length=max_length, temperature=temperature, top_k=top_k,
                         device=DEVICE, repetition_penalty=1.4, min_length=3, eos_penalty=-5.0)
    return gen, tokens

def self_loss(model, gen_ids, prompt_len):
    cont = gen_ids[prompt_len:]
    if len(cont) < 1:
        return float("nan"), float("nan")
    with torch.no_grad():
        logits = model(torch.tensor([gen_ids], device=DEVICE))   # (1, T, V)
        pred = logits[0, prompt_len-1 : prompt_len-1+len(cont)]   # (Lc, V)
        tgt = torch.tensor(cont, device=DEVICE)
        loss = criterion(pred, tgt).item()
    rep = sum(1 for i in range(1, len(cont)) if cont[i] == cont[i-1]) / max(1, len(cont))
    uniq = len(set(cont)) / max(1, len(cont))
    return loss, rep, uniq

# 控制变量：固定其余，分别扫 temperature / top_k
SECTIONS = [
    ("温度扫描 (top_k=50, rep=1.4)", [("T=0.5", dict(temperature=0.5, top_k=50)),
                                      ("T=0.8", dict(temperature=0.8, top_k=50)),
                                      ("T=1.1", dict(temperature=1.1, top_k=50)),
                                      ("T=1.4", dict(temperature=1.4, top_k=50))]),
    ("top_k 扫描 (T=0.8, rep=1.4)", [("K=10",  dict(temperature=0.8, top_k=10)),
                                     ("K=30",  dict(temperature=0.8, top_k=30)),
                                     ("K=100", dict(temperature=0.8, top_k=100))]),
]

log(f"设备: {DEVICE}  | 模型: ENH(常开) vs SEL(分段选择性交替)  | 提示词 {len(PROMPTS)} 个")
log("指标: self_loss = 模型对自己生成续写的 cross-entropy（越低=越自洽/鲁棒）；rep = 连续重复占比（越低越好）；uniq = 唯一 token 比（越高越不退化）")
log("=" * 78)

summary = {name: {} for name, _, _ in MODELS}
for sec_name, settings in SECTIONS:
    log(f"\n########## {sec_name} ##########")
    for name, m, v in MODELS:
        for setting, kw in settings:
            losses, reps, uniqs = [], [], []
            log(f"\n--- {name} | {setting} ---")
            for p in PROMPTS:
                gen, toks = generate_ids(m, v, p, kw["temperature"], kw["top_k"])
                loss, rep, uniq = self_loss(m, gen, len(toks))
                text = v.decode(gen)
                losses.append(loss); reps.append(rep); uniqs.append(uniq)
                log(f"  [{p}] self_loss={loss:.2f} rep={rep:.2f} uniq={uniq:.2f}")
                log(f"      {text}")
            summary[name][setting] = (sum(losses)/len(losses), sum(reps)/len(reps), sum(uniqs)/len(uniqs))

log("\n" + "=" * 78)
log("【汇总：各设置下平均 self_loss / rep / uniq（越低 loss&rep 越好，uniq 越高越好）】")
for sec_name, settings in SECTIONS:
    log(f"\n{sec_name}")
    for setting, _ in settings:
        enh = summary["ENH"][setting]; sel = summary["SEL"][setting]
        log(f"  {setting:6s} | ENH loss={enh[0]:.2f} rep={enh[1]:.2f} uniq={enh[2]:.2f}"
            f"   ||   SEL loss={sel[0]:.2f} rep={sel[1]:.2f} uniq={sel[2]:.2f}"
            f"   -> {'SEL更鲁棒' if sel[0] < enh[0] else 'ENH更鲁棒'}")

with open("experiments/robust_enh_sel.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
log("\n原始结果（含生成原文）已写入 experiments/robust_enh_sel.txt")
