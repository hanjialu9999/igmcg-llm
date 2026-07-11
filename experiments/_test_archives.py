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

torch.set_num_threads(4)
dev = get_device('cpu')
vocab_path = 'checkpoints_dml_test/vocab.json'
intuition = [0.3, 0.8, 0.5, 0.2, 0.6, 0.4, 0.5]
prompts = ['今天天气怎么样', '你好，请问你是谁', '中国的首都是哪里', '床前明月光']

models = {
    '_stab_ckpt(28M)': r'archive_unused\checkpoints_backup\_stab_ckpt\final_model.pt',
    'checkpoints_4(8.8M)': r'archive_unused\checkpoints_backup\checkpoints_4\final_model.pt',
}
lines = []
ng = None
for name, mp in models.items():
    print('loading', name, flush=True)
    try:
        model, vocab = load_model(mp, vocab_path, device=dev)
    except Exception as e:
        lines.append(f'### {name}: LOAD FAILED -> {e}\n')
        print('  load failed', e, flush=True)
        continue
    if ng is None:
        ng = NGramModel(vocab, 'data/pretrain_corpus/merged_sample.txt', max_order=3, smoothing=1.0)
    n = sum(p.numel() for p in model.parameters())
    lines.append(f'### {name}  参数量={n/1e6:.2f}M  架构=纯attn(KV-cache可用)\n')
    for p in prompts:
        base = generate_text(model, vocab, p, max_length=60, temperature=0.7, top_k=30, device=dev)
        ng_out = generate_text(model, vocab, p, max_length=60, temperature=0.7, top_k=30,
                               device=dev, ngram=ng, ngram_weight=0.3)
        ig_out, _ = generate_igmcg(model, vocab, p, max_length=60, top_k=30, device=dev,
                                   num_candidates=5, base_temp=0.7, intuition=intuition,
                                   ngram_fn=ng.logprob_vector, ngram_weight=0.3)
        lines.append(f'P: {p}\n  基础:   {base}\n  n-gram: {ng_out}\n  IGMCG:  {ig_out}\n')
    print('done', name, flush=True)

open('logs/archive_samples.txt', 'w', encoding='utf-8').write('\n'.join(lines))
print('done all', flush=True)
