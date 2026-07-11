import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, 'scripts') not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
from generate import load_model, generate_igmcg, NGramModel
torch = __import__('torch')
torch.set_num_threads(4)
dev = 'cpu'
model, vocab = load_model(r'archive_unused\checkpoints_backup\_stab_ckpt\final_model.pt',
                          'checkpoints_dml_test/vocab.json', device=dev)
ng = NGramModel(vocab, 'data/pretrain_corpus/merged_sample.txt', max_order=3,
                smoothing=1.0, l1=0.1, l2=0.3, l3=0.6)
fn = (lambda g, d: ng.logprob_vector(g, d))
out = []
for p in ['今天天气怎么样', '中国的首都是哪里', '我觉得这个项目很有意思']:
    best, cands = generate_igmcg(model, vocab, p, max_length=60, top_k=30, device=dev,
                                 num_candidates=5, base_temp=0.7,
                                 intuition=[0.3, 0.8, 0.5, 0.2, 0.6, 0.4, 0.5],
                                 ngram_fn=fn, ngram_weight=0.3)
    out.append('提示: ' + p)
    for i, c in enumerate(cands):
        out.append('  #%d T=%s 重复=%.3f: %s' % (i, c['temp'], c['rep'], c['text']))
    out.append('  >>> 选中: ' + best)
    out.append('')
open('logs/show.txt', 'w', encoding='utf-8').write('\n'.join(out))
