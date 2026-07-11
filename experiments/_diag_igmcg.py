import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, 'scripts') not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
from generate import load_model, generate_text, generate_igmcg, NGramModel
torch = __import__('torch')
torch.set_num_threads(4)
dev = 'cpu'
model, vocab = load_model(r'archive_unused\checkpoints_backup\_stab_ckpt\final_model.pt',
                          'checkpoints_dml_test/vocab.json', device=dev)
ng = NGramModel(vocab, 'data/pretrain_corpus/merged_sample.txt', max_order=3,
                smoothing=1.0, l1=0.1, l2=0.3, l3=0.6)
ngram_fn = (lambda g, d: ng.logprob_vector(g, d))


def model_fluency(ids):
    if len(ids) < 2:
        return 0.0
    with torch.inference_mode():
        logits = model.forward(torch.tensor([ids], dtype=torch.long, device=dev))
        lp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        tgt = torch.tensor(ids[1:], dtype=torch.long, device=dev).unsqueeze(1)
        return lp.gather(1, tgt).mean().item()


def ngram_coherence(ids):
    """平均 n-gram 模型对序列的预测 log-prob：越高=越连贯(相连)，越低=越碎片化。"""
    if len(ids) < 2:
        return 0.0
    tot, n = 0.0, 0
    for i in range(1, len(ids)):
        lp = ng.logprob_vector(ids[:i], dev)
        tot += lp[ids[i]].item()
        n += 1
    return tot / max(1, n)


prompts = ['今天天气怎么样', '中国的首都是哪里', '我觉得这个项目很有意思',
           '人工智能对未来有什么影响', '如何学习一门新语言']
lines = []
def L(s):
    lines.append(s)
L('%-2s %-14s %10s %12s' % ('#', 'mode', 'flu(model)', 'coh(ngram)'))
strong = [0.1, 0.95, 0.95, 0.1, 0.95, 0.1, 0.9]
for pi, p in enumerate(prompts):
    ids0 = vocab.encode(p, add_special_tokens=False)
    base = model.generate(ids0, max_length=60, temperature=0.7, top_k=30,
                          repetition_penalty=1.7, device=dev)
    ngc = model.generate(ids0, max_length=60, temperature=0.7, top_k=30,
                         repetition_penalty=1.7, device=dev,
                         ngram_fn=ngram_fn, ngram_weight=0.3)
    bestN, candsN = generate_igmcg(model, vocab, p, max_length=60, top_k=30, device=dev,
                                   num_candidates=5, base_temp=0.7,
                                   intuition=[0.5] * 7, ngram_fn=ngram_fn, ngram_weight=0.3)
    bestS, candsS = generate_igmcg(model, vocab, p, max_length=60, top_k=30, device=dev,
                                   num_candidates=5, base_temp=0.7,
                                   intuition=strong, ngram_fn=ngram_fn, ngram_weight=0.3)
    itN = vocab.encode(bestN, add_special_tokens=False)
    itS = vocab.encode(bestS, add_special_tokens=False)
    L('%-2d %-14s %10.3f %12.3f' % (pi, 'base', model_fluency(base), ngram_coherence(base)))
    L('%-2d %-14s %10.3f %12.3f' % (pi, 'ngram', model_fluency(ngc), ngram_coherence(ngc)))
    L('%-2d %-14s %10.3f %12.3f' % (pi, 'igmcg-neutral', model_fluency(itN), ngram_coherence(itN)))
    L('%-2d %-14s %10.3f %12.3f' % (pi, 'igmcg-strong', model_fluency(itS), ngram_coherence(itS)))
    allc = candsN + candsS
    oracle = max(ngram_coherence(vocab.encode(c['text'], add_special_tokens=False)) for c in allc)
    L('   oracle best coh = %.3f (sel-neutral=%.3f, sel-strong=%.3f)'
      % (oracle, ngram_coherence(itN), ngram_coherence(itS)))
    L('')
open('logs/diag_igmcg.txt', 'w', encoding='utf-8').write('\n'.join(lines))
