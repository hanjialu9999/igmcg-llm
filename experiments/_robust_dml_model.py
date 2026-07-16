"""
鲁棒性探针（单模型版）：针对 chat.py 当前调用的 checkpoints_dml/final_model.pt
做 self-loss 扫描，评估模型对自身续写的自洽性 / 分布外鲁棒性。

self-loss = 模型对自己生成的续写序列的 CE 损失（越低越自洽、越稳健）。
扫描两组：
  - 温度扫描 (top_k=50, rep=1.4)：T=0.5 / 0.8 / 1.1 / 1.4
  - top_k 扫描 (T=0.8, rep=1.4)：K=10 / 30 / 100

用法：
  python experiments/_robust_dml_model.py
结果写 experiments/robust_dml_model.txt
"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

from scripts.generate import load_model, get_device
import torch
import torch.nn as nn

DEVICE = get_device("dml")
MODEL_PATH = "checkpoints_dml/final_model.pt"
VOCAB_PATH = "checkpoints_dml/vocab.json"
PROMPTS = ["中国的首都是", "人工智能的发展", "学习编程需要", "科学技术是第一"]
INTUITION = [0.3, 0.8, 0.5, 0.2, 0.6, 0.4, 0.5]

model, vocab = load_model(MODEL_PATH, VOCAB_PATH, device=DEVICE)
model.eval()
criterion = nn.CrossEntropyLoss()

lines = []
def log(s=""):
    print(s)
    lines.append(s)

def gen_ids(prompt, temperature, top_k, max_length=60):
    tokens = vocab.encode(prompt)
    if tokens and tokens[-1] == vocab.eos_idx:
        tokens = tokens[:-1]
    gen = model.generate(tokens, max_length=max_length, temperature=temperature,
                         top_k=top_k, device=DEVICE, repetition_penalty=1.4,
                         min_length=3, eos_penalty=-5.0)
    return gen, tokens

def self_loss(gen_ids, prompt_len):
    cont = gen_ids[prompt_len:]
    if len(cont) < 1:
        return float("nan")
    with torch.no_grad():
        logits = model(torch.tensor([gen_ids], device=DEVICE))
        pred = logits[0, prompt_len - 1: prompt_len - 1 + len(cont)]
        tgt = torch.tensor(cont, device=DEVICE)
    return criterion(pred, tgt).item()

SECTIONS = [
    ("温度扫描 (top_k=50, rep=1.4)",
     [("T=0.5", dict(temperature=0.5, top_k=50)),
      ("T=0.8", dict(temperature=0.8, top_k=50)),
      ("T=1.1", dict(temperature=1.1, top_k=50)),
      ("T=1.4", dict(temperature=1.4, top_k=50))]),
    ("top_k 扫描 (T=0.8, rep=1.4)",
     [("K=10", dict(temperature=0.8, top_k=10)),
      ("K=30", dict(temperature=0.8, top_k=30)),
      ("K=100", dict(temperature=0.8, top_k=100))]),
]

log(f"设备: {DEVICE}  |  鲁棒性探针（单模型）：{MODEL_PATH}")
log("模型词表大小: %d" % len(vocab))
log("=" * 78)

# 生成样例 + 速度
gen_stats = []
log("【生成样例（top-k, T=0.8, K=50, rep=1.4）】")
for i, p in enumerate(PROMPTS):
    t0 = time.time()
    toks = vocab.encode(p)
    if toks and toks[-1] == vocab.eos_idx:
        toks = toks[:-1]
    gen = model.generate(toks, max_length=60, temperature=0.8, top_k=50,
                        device=DEVICE, repetition_penalty=1.4,
                        min_length=3, eos_penalty=-5.0)
    e = vocab.decode(gen, skip_special=True).strip()
    t1 = time.time()
    tk = max(1, len(vocab.encode(e)) - len(vocab.encode(p)))
    gen_stats.append(tk / (t1 - t0))
    log(f"\n提示{i+1}: {p}")
    log(f"  top-k : {e}")
log(f"\n生成速度(均值): {sum(gen_stats)/len(gen_stats):.1f} tok/s")

log("\n" + "=" * 78)
log("【鲁棒性探针：self_loss（越低越自洽/稳健）】")
summary = {}
for sec_name, settings in SECTIONS:
    log(f"\n########## {sec_name} ##########")
    for setting, kw in settings:
        losses = []
        for p in PROMPTS:
            gen, toks = gen_ids(p, kw["temperature"], kw["top_k"])
            losses.append(self_loss(gen, len(toks)))
        avg = sum(losses) / len(losses)
        summary[setting] = avg
        log(f"  {setting:6s} | self_loss={avg:.3f}  (各提示: {[round(x,2) for x in losses]})")

log("\n" + "=" * 78)
log("【结论】self_loss 越低代表模型对自身续写越自洽、在对应采样设置下越稳健；")
log("温度越高 / top_k 越大通常 self_loss 越高（生成更发散、与训练分布越远）。")

with open("experiments/robust_dml_model.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
log("\n原始结果已写 experiments/robust_dml_model.txt")
