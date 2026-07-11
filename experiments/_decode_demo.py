import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, 'scripts') not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))
from generate import load_model, generate_text, generate_igmcg, NGramModel
from models.device import get_device

dev = get_device('cpu')
model, vocab = load_model('checkpoints_dml_b32/final_model.pt',
                          'checkpoints_dml_b32/vocab.json', device=dev)
print('model loaded', flush=True)
ng = NGramModel(vocab, 'data/pretrain_corpus/merged_sample.txt', max_order=3, smoothing=1.0)

prompts = ['今天天气怎么样', '你好，请问你是谁', '中国的首都是哪里', '床前明月光']
lines = []
intuition = [0.3, 0.8, 0.5, 0.2, 0.6, 0.4, 0.5]
for p in prompts:
    basic = generate_text(model, vocab, p, max_length=40, temperature=0.7, top_k=30, device=dev)
    ng_out = generate_text(model, vocab, p, max_length=40, temperature=0.7, top_k=30,
                           device=dev, ngram=ng, ngram_weight=0.3)
    igmcg_out, cands = generate_igmcg(model, vocab, p, max_length=40, top_k=30,
                                      device=dev, num_candidates=5, base_temp=0.7,
                                      intuition=intuition, ngram_fn=ng.logprob_vector,
                                      ngram_weight=0.3)
    best = cands[0] if cands else None
    lines.append(
        f'P: {p}\n'
        f'  基础:   {basic}\n'
        f'  n-gram: {ng_out}\n'
        f'  IGMCG:  {igmcg_out}\n'
        f'    [IGMCG候选数={len(cands)}, 最优分={best["score"]:.3f}, 重复度={best["rep"]:.3f}]\n'
    )
    print('done', p, flush=True)

open('logs/decode_demo.txt', 'w', encoding='utf-8').write('\n'.join(lines))
print('Decode demo done', flush=True)
