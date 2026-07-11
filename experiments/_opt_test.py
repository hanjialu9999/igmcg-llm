import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, 'scripts') not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import time, sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
from generate import load_model, generate_igmcg, NGramModel

CKPT = r'archive_unused\checkpoints_backup\_stab_ckpt\final_model.pt'
VOCAB = 'checkpoints_dml_test/vocab.json'
CORPUS = 'data/pretrain_corpus/merged_sample.txt'
DEV = 'cpu'

torch = __import__('torch')
torch.set_num_threads(4)

model, vocab = load_model(CKPT, VOCAB, device=DEV)
ng = NGramModel(vocab, CORPUS, max_order=3, smoothing=1.0, l1=0.1, l2=0.3, l3=0.6)
ngram_fn = (lambda g, d: ng.logprob_vector(g, d))

prompts = ['今天天气怎么样', '中国的首都是哪里', '我觉得这个项目很有意思']
N, W = 5, 0.4
intuition = [0.3, 0.8, 0.5, 0.2, 0.6, 0.4, 0.5]

t0 = time.perf_counter()
for p in prompts:
    generate_igmcg(model, vocab, p, intuition=intuition, num_candidates=N,
                   max_length=60, device=DEV, base_temp=0.7, top_k=30,
                   ngram_fn=ngram_fn, ngram_weight=W)
dt = time.perf_counter() - t0
print('=== 优化后(批量化IGMCG + 加速n-gram) ===')
print(f'总耗时 {dt:.2f}s  3提示x{N}候选  约 {3*N*60/max(dt,1e-9):.1f} tok/s(有效)')
for p in prompts:
    best, cands = generate_igmcg(model, vocab, p, intuition=intuition,
                                 num_candidates=N, max_length=60, device=DEV,
                                 base_temp=0.7, top_k=30,
                                 ngram_fn=ngram_fn, ngram_weight=W)
    print(f'\n提示: {p}')
    for i, c in enumerate(cands):
        print(f"  #{i} [分={c['score']:.3f} 流畅={c['flu']:.2f} 重复={c['rep']:.3f} 风格={c['style']:.2f} T={c['temp']}]: {c['text'][:48]}")
    print(f"  >>> 选中: {best[:48]}")
