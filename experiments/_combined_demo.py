import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, 'scripts') not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import sys, os, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))
from generate import load_model, generate_igmcg, NGramModel
from models.device import get_device

torch.set_num_threads(4)   # 低功耗
dev = get_device('cpu')
vocab_path = 'checkpoints_dml_test/vocab.json'
mp = r'archive_unused\checkpoints_backup\_stab_ckpt\final_model.pt'
model, vocab = load_model(mp, vocab_path, device=dev)
ng = NGramModel(vocab, 'data/pretrain_corpus/merged_sample.txt', max_order=3,
                smoothing=1.0, l1=0.1, l2=0.3, l3=0.6)
print('ngram tri 上下文数:', len(ng.tri), 'bi 上下文数:', len(ng.bi), flush=True)

intuition = [0.3, 0.8, 0.5, 0.2, 0.6, 0.4, 0.5]
prompts = ['今天天气怎么样', '中国的首都是哪里', '床前明月光']
lines = []
t0 = time.time()
total_tok = 0
for p in prompts:
    out, cands = generate_igmcg(model, vocab, p, max_length=60, top_k=30, device=dev,
                                num_candidates=5, base_temp=0.7, intuition=intuition,
                                ngram_fn=ng.logprob_vector, ngram_weight=0.3)
    total_tok += 60
    best = cands[0] if cands else None
    lines.append(f'P: {p}\n  [n-gram+IGMCG 联合] {out}\n   (候选={len(cands)}, 最优分={best["score"]:.3f}, 重复度={best["rep"]:.3f})\n')
    print('done', p, flush=True)
elapsed = time.time() - t0
lines.append(f'\n速度: {total_tok/elapsed:.1f} tok/s | CPU 线程=4 (低功耗) | KV-cache 增量解码')
open('logs/combined_demo.txt', 'w', encoding='utf-8').write('\n'.join(lines))
print(f'\n速度: {total_tok/elapsed:.1f} tok/s', flush=True)
