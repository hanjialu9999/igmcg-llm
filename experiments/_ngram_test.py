import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, 'scripts') not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))
from generate import load_model, generate_text, NGramModel
from models.device import get_device

dev = get_device('cpu')
model, vocab = load_model('checkpoints_dml_b32/final_model.pt',
                          'checkpoints_dml_b32/vocab.json', device=dev)
print('model loaded', flush=True)
ng = NGramModel(vocab, 'data/pretrain_corpus/merged_sample.txt', max_order=3, smoothing=1.0)
print('ngram built', flush=True)

prompts = ['今天天气怎么样', '你好，请问你是谁', '中国的首都是哪里', '床前明月光']
lines = []
for p in prompts:
    a = generate_text(model, vocab, p, max_length=60, temperature=0.7, top_k=30,
                      device=dev)
    b = generate_text(model, vocab, p, max_length=60, temperature=0.7, top_k=30,
                      device=dev, ngram=ng, ngram_weight=0.3)
    lines.append(f'P: {p}\n  无ngram: {a}\n  有ngram: {b}\n')
    print('done', p, flush=True)

open('logs/ngram_test.txt', 'w', encoding='utf-8').write('\n'.join(lines))
print('NGram test done', flush=True)
