import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, 'scripts') not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))
from generate import load_model, generate_igmcg, NGramModel
from models.device import get_device

torch.set_num_threads(4)
dev = get_device('cpu')
vocab_path = 'checkpoints_dml_test/vocab.json'
mp = r'archive_unused\checkpoints_backup\_stab_ckpt\final_model.pt'
model, vocab = load_model(mp, vocab_path, device=dev)
ng = NGramModel(vocab, 'data/pretrain_corpus/merged_sample.txt', max_order=3,
                smoothing=1.0, l1=0.1, l2=0.3, l3=0.6)

intuition = [0.3, 0.8, 0.5, 0.2, 0.6, 0.4, 0.5]
prompts = ['今天天气怎么样', '中国的首都是哪里']
lines = []
for p in prompts:
    lines.append('=' * 70)
    lines.append(f'提示: {p}')
    lines.append('联合解码 = 神经LM + n-gram(uni/bi/tri插值)先验叠加 + IGMCG多候选(7维直觉)筛选')
    out, cands = generate_igmcg(model, vocab, p, max_length=60, top_k=30, device=dev,
                                num_candidates=5, base_temp=0.7, intuition=intuition,
                                ngram_fn=ng.logprob_vector, ngram_weight=0.3)
    # 按分数排序展示所有候选
    for i, c in enumerate(cands):
        lines.append(f'  候选#{i} [temp={c["temp"]} 分={c["score"]:.3f} '
                     f'流畅={c["flu"]:.2f} 重复={c["rep"]:.3f} 风格={c["style"]:.2f}]: {c["text"]}')
    lines.append(f'  >>> 最终选中(分数最高): {out}')
    lines.append('')
open('logs/combined_full.txt', 'w', encoding='utf-8').write('\n'.join(lines))
print('done', flush=True)
