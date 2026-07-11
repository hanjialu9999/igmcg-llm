import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, 'scripts') not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))
from generate import (load_model, generate_text, generate_igmcg, NGramModel,
                      _repetition, _fluency)
from models.config_loader import load_config, build_model
from models.device import get_device

dev = get_device('cpu')

# ---------- 参数统计 ----------
model, vocab = load_model('checkpoints_dml_b32/final_model.pt',
                          'checkpoints_dml_b32/vocab.json', device=dev)
n_base = sum(p.numel() for p in model.parameters())
print(f'当前完整模型(纯attn, d512/6层) 参数量: {n_base/1e6:.2f}M', flush=True)

cfg_small = load_config('configs/config_hybrid_small.yaml')
small = build_model(cfg_small)
n_small = sum(p.numel() for p in small.parameters())
print(f'小混合模型(d256/4层, attn,ssm,attn,ssm) 参数量: {n_small/1e6:.2f}M', flush=True)

# ---------- 语料 bigram 集合（连贯性参照，用完整语料） ----------
ng = NGramModel(vocab, 'data/pretrain_corpus/merged_sample.txt', max_order=3, smoothing=1.0)
corpus_bg = set()
with open('data/pretrain_corpus/merged.txt', 'r', encoding='utf-8') as f:
    for line in f:
        ids = vocab.encode(line, add_special_tokens=False)
        for i in range(1, len(ids)):
            corpus_bg.add((ids[i-1], ids[i]))
print(f'完整语料 bigram 数: {len(corpus_bg)}', flush=True)

def metrics(model, vocab, prompt, gen_ids, device):
    plen = len(vocab.encode(prompt, add_special_tokens=False))
    cont = gen_ids[plen:]
    bg = [(cont[i], cont[i+1]) for i in range(len(cont)-1)] or [(0, 0)]
    uniq_bg = len(set(bg))
    hit = sum(1 for b in bg if b in corpus_bg) / max(1, len(bg))
    d1 = len(set(cont)) / max(1, len(cont))
    d2 = uniq_bg / max(1, len(bg))
    text = vocab.decode(cont, skip_special=True)
    rep = _repetition(text)
    flu = _fluency(model, gen_ids, device)  # 越高越流畅(负loss)
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
        agg['base'].append(metrics(model, vocab, p, g_base, dev))
        agg['ngram'].append(metrics(model, vocab, p, g_ng, dev))
        agg['igmcg'].append(metrics(model, vocab, p, g_ig, dev))
    print('done', p, flush=True)

def avg(lst, k):
    return sum(x[k] for x in lst) / len(lst)

lines = []
lines.append(f'当前模型参数量: {n_base/1e6:.2f}M | 小混合模型: {n_small/1e6:.2f}M')
lines.append('指标含义: hit=生成bigram命中语料比例(连贯性↑), d1/d2=字符/词多样性(↑), rep=重复率(↓), flu=模型自身流畅度(↑)')
lines.append(f'{"方法":8} | {"hit":>6} | {"d1":>5} | {"d2":>5} | {"rep":>6} | {"flu":>7}')
for name in ['base', 'ngram', 'igmcg']:
    a = agg[name]
    lines.append(f'{name:8} | {avg(a,"hit"):6.3f} | {avg(a,"d1"):5.3f} | '
                 f'{avg(a,"d2"):5.3f} | {avg(a,"rep"):6.3f} | {avg(a,"flu"):7.3f}')
open('logs/decode_metrics.txt', 'w', encoding='utf-8').write('\n'.join(lines))
print('\n'.join(lines), flush=True)
