import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, 'scripts') not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import sys, os, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))
from generate import load_model, generate_text, generate_igmcg, NGramModel
from models.device import get_device

torch.set_num_threads(4)   # 降功耗
dev = get_device('cpu')
model, vocab = load_model('checkpoints_dml_b32/final_model.pt',
                          'checkpoints_dml_b32/vocab.json', device=dev)
print('loaded', flush=True)

# ---------- 1) KV-cache 正确性：增量解码应与全量 forward 逐 token 一致 ----------
model.eval()
seq = torch.tensor([[10, 20, 30, 40, 50]], device=dev)
with torch.no_grad():
    full = model.forward(seq)                              # (1,5,V)
    past = None
    for i in range(5):
        lg, past = model.forward(seq[:, i:i+1], past_key_values=past,
                                 use_cache=True)
    diff = (full[0, -1] - lg[0, -1]).abs().max().item()
print(f'KV-cache 一致性最大误差: {diff:.2e}', flush=True)

# ---------- 2) 速度与一致性对比（贪心，避免随机） ----------
def gen_full(m, ids, n):
    with torch.no_grad():
        for _ in range(n):
            if len(ids) >= m.max_seq_length:
                break
            ctx = ids[-m.max_seq_length:] if len(ids) > m.max_seq_length else ids
            inp = torch.tensor([ctx], dtype=torch.long, device=dev)
            nt = m.forward(inp)[0, -1].argmax().item()
            ids.append(nt)
    return ids

def gen_cache(m, ids, n):
    with torch.no_grad():
        past = None; cur = 0
        inp = torch.tensor([ids], dtype=torch.long, device=dev)
        lg, past = m.forward(inp, past_key_values=None, use_cache=True)
        cur = inp.size(1)
        for _ in range(n):
            if cur >= m.max_seq_length:
                break
            nt = lg[0, -1].argmax().item()
            ids.append(nt)
            inp = torch.tensor([[nt]], dtype=torch.long, device=dev)
            lg, past = m.forward(inp, past_key_values=past, use_cache=True)
            cur += 1
    return ids

prompt_ids = vocab.encode('今天天气怎么样', add_special_tokens=False)
N = 60
t0 = time.time(); r_full = gen_full(model, list(prompt_ids), N); t_full = time.time() - t0
t0 = time.time(); r_cache = gen_cache(model, list(prompt_ids), N); t_cache = time.time() - t0
print(f'旧(全量重算): {t_full:.2f}s, {N/t_full:.1f} tok/s', flush=True)
print(f'新(KV-cache): {t_cache:.2f}s, {N/t_cache:.1f} tok/s, 加速 {t_full/t_cache:.1f}x', flush=True)
print('贪心输出一致:', r_full == r_cache, flush=True)

# ---------- 3) 实际生成原文（供人工判断连贯性） ----------
ng = NGramModel(vocab, 'data/pretrain_corpus/merged_sample.txt', max_order=3, smoothing=1.0)
prompts = ['今天天气怎么样', '你好，请问你是谁', '中国的首都是哪里', '床前明月光']
intuition = [0.3, 0.8, 0.5, 0.2, 0.6, 0.4, 0.5]
lines = []
for p in prompts:
    base = generate_text(model, vocab, p, max_length=60, temperature=0.7, top_k=30,
                         device=dev)
    ng_out = generate_text(model, vocab, p, max_length=60, temperature=0.7, top_k=30,
                           device=dev, ngram=ng, ngram_weight=0.3)
    ig_out, _ = generate_igmcg(model, vocab, p, max_length=60, top_k=30, device=dev,
                               num_candidates=5, base_temp=0.7, intuition=intuition,
                               ngram_fn=ng.logprob_vector, ngram_weight=0.3)
    lines.append(f'P: {p}\n  基础:   {base}\n  n-gram: {ng_out}\n  IGMCG:  {ig_out}\n')
    print('gen', p, flush=True)
open('logs/gen_samples.txt', 'w', encoding='utf-8').write('\n'.join(lines))
print('done', flush=True)
