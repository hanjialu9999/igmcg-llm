import sys
import time
import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models.device import get_device
from generate import load_model, generate_text, generate_igmcg

CKPT = ROOT / "checkpoints_smoke_4k"
DEVICE = get_device("auto")
torch.set_num_threads(4)

print(f"device = {DEVICE}")
model, vocab = load_model(str(CKPT / "final_model.pt"), str(CKPT / "vocab.json"), device=DEVICE)
model.eval()

PROMPTS = ["你好，", "The weather is", "机器学习是", "今天天气", "I love"]
INTUITION = [0.5] * 7
MAX_LEN = 40

print("\n" + "=" * 70)
print("生成对比：baseline(top-k)  vs  IGMCG(多候选)  —— 速度与效果")
print("=" * 70)

for p in PROMPTS:
    # baseline: top-k 采样
    t0 = time.perf_counter()
    base = generate_text(model, vocab, p, max_length=MAX_LEN,
                         temperature=0.8, top_k=50, device=DEVICE)
    t1 = time.perf_counter()

    # IGMCG: 5 候选并行生成 + 评分选优
    t2 = time.perf_counter()
    gen, cands = generate_igmcg(model, vocab, p, intuition=INTUITION, num_candidates=5,
                                max_length=MAX_LEN, base_temp=0.8, top_k=50, device=DEVICE)
    t3 = time.perf_counter()

    base_tok = max(len(vocab.encode(base)) - len(vocab.encode(p)), 1)
    igmcg_tok = max(len(vocab.encode(gen)) - len(vocab.encode(p)), 1)
    base_s = base_tok / max(t1 - t0, 1e-6)
    igmcg_s = igmcg_tok / max(t3 - t2, 1e-6)
    score = cands[0]["score"] if cands else 0.0
    rep = cands[0].get("rep", 0.0) if cands else 0.0

    print(f"\nPROMPT : {p}")
    print(f"  [base ] {base}")
    print(f"          {base_tok} tok, {base_s:.1f} tok/s")
    print(f"  [igmcg] {gen}")
    print(f"          {igmcg_tok} tok, {igmcg_s:.1f} tok/s, score={score:.3f}, rep={rep:.3f}")
