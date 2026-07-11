import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, 'scripts') not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import sys, os, json, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))
from generate import load_model, generate_text
from models.data_utils import Vocabulary
from models.device import get_device

dev = get_device('cpu')

def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    v = Vocabulary()
    v.word2idx = d['word2idx']
    v.idx2word = {int(k): val for k, val in d['idx2word'].items()}
    v.vocab_size = len(v.word2idx)
    return v

mp = r'archive_unused\checkpoints_backup\_stab_ckpt\final_model.pt'
# 加载一次权重（vocab 仅用于取对象，真正解码用候选 vocab）
model, _ = load_model(mp, 'checkpoints/vocab.json', device=dev)
print('model loaded, params=', sum(p.numel() for p in model.parameters())/1e6, 'M', flush=True)

cands = ['checkpoints/vocab.json', 'checkpoints_dml_b32/vocab.json',
         'checkpoints_dml_test/vocab.json']
prompts = ['今天天气怎么样', '中国的首都是哪里']
out_lines = []
for cp in cands:
    v = load_vocab(cp)
    out_lines.append('=' * 60)
    out_lines.append(f'VOCAB: {cp} size {v.vocab_size}')
    for p in prompts:
        out = generate_text(model, v, p, max_length=50, temperature=0.7, top_k=30, device=dev)
        out_lines.append(f'  [{p}] -> {out}')
open('logs/vocab_detect.txt', 'w', encoding='utf-8').write('\n'.join(out_lines))
print('done', flush=True)
