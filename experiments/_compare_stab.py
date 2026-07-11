import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, 'scripts') not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import sys, os, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))
from generate import (load_model, generate_text, generate_igmcg, NGramModel,
                      _repetition, _fluency)
from models.device import get_device

torch.set_num_threads(4)
dev = get_device('cpu')
vocab_path = 'checkpoints_dml_test/vocab.json'
model, vocab = load_model(r'archive_unused\checkpoints_backup\_stab_ckpt\final_model.pt',
                          vocab_path, device=dev)
print('loaded', sum(p.numel() for p in model.parameters())/1e6, 'M', flush=True)

# n-gram 先验（必须用同一个 vocab）
ng = NGramModel(vocab, 'data/pretrain_corpus/merged_sample.txt', max_order=3, smoothing=1.0)
# 连贯性参照：完整语料的 bigram 集合
corpus_bg = set()
with open('data/pretrain_corpus/merged.txt', 'r', encoding='utf-8') as f:
    for line in f:
        ids = vocab.encode(line, add_special_tokens=False)
        for i in range(1, len(ids)):
            corpus_bg.add((ids[i-1], ids[i]))
print('完整语料 bigram 数:', len(corpus_bg), flush=True)

def metrics(model, prompt, gen_ids):
    plen = len(vocab.encode(prompt, add_special_tokens=False))
    cont = gen_ids[plen:]
    bg = [(cont[i], cont[i+1]) for i in range(len(cont)-1)] or [(0, 0)]
    uniq_bg = len(set(bg))
    hit = sum(1 for b in bg if b in corpus_bg) / max(1, len(bg))
    d1 = len(set(cont)) / max(1, len(cont))
    d2 = uniq_bg / max(1, len(bg))
    text = vocab.decode(cont, skip_special=True)
    rep = _repetition(text)
    flu = _fluency(model, gen_ids, dev)
    return {'hit': hit, 'd1': d1, 'd2': d2, 'rep': rep, 'flu': flu}

prompts = ['今天天气怎么样', '你好，请问你是谁', '中国的首都是哪里', '床前明月光']
intuition = [0.3, 0.8, 0.5, 0.2, 0.6, 0.4, 0.5]
samples = 3
agg = {'base': [], 'ngram': [], 'igmcg': []}
for p in prompts:
    ids = vocab.encode(p, add_special_tokens=False)
    for s in range(samples):
        g_base = model.generate(ids, max_length=60, temperature=0.7, top_k=30,
                                repetition_penalty=1.4, device=dev)
        g_ng = model.generate(ids, max_length=60, temperature=0.7, top_k=30,
                              repetition_penalty=1.4, device=dev,
                              ngram_fn=ng.logprob_vector, ngram_weight=0.3)
        t_ig, _ = generate_igmcg(model, vocab, p, max_length=60, top_k=30, device=dev,
                                 num_candidates=5, base_temp=0.7, intuition=intuition,
                                 ngram_fn=ng.logprob_vector, ngram_weight=0.3)
        g_ig = vocab.encode(p + t_ig, add_special_tokens=False)
        agg['base'].append(metrics(model, p, g_base))
        agg['ngram'].append(metrics(model, p, g_ng))
        agg['igmcg'].append(metrics(model, p, g_ig))
    print('done', p, flush=True)

def avg(lst, k):
    return sum(x[k] for x in lst) / len(lst)

# 速度
t0 = time.time()
model.generate(vocab.encode('今天天气怎么样', add_special_tokens=False),
               max_length=60, temperature=0.7, top_k=30, device=dev)
spd = 60 / (time.time() - t0)

lines = []
lines.append(f'模型: _stab_ckpt (~22M, 纯attn, KV-cache) | 速度 ~{spd:.1f} tok/s (CPU, 4线程)')
lines.append('指标: hit=生成bigram命中语料(连贯↑), d1/d2=多样性(↑), rep=重复率(↓), flu=模型流畅度(↑)')
lines.append(f'{"方法":8} | {"hit":>6} | {"d1":>5} | {"d2":>5} | {"rep":>6} | {"flu":>7}')
for name in ['base', 'ngram', 'igmcg']:
    a = agg[name]
    lines.append(f'{name:8} | {avg(a,"hit"):6.3f} | {avg(a,"d1"):5.3f} | '
                 f'{avg(a,"d2"):5.3f} | {avg(a,"rep"):6.3f} | {avg(a,"flu"):7.3f}')
open('logs/stab_metrics.txt', 'w', encoding='utf-8').write('\n'.join(lines))
print('\n'.join(lines), flush=True)
