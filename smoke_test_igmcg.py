import sys
sys.path.insert(0, r'F:\Projects\新项目')
import torch
from models.config_loader import load_config, build_model
from models.data_utils import Vocabulary
from scripts.generate import _generate_candidates_batch

print('Testing IGMCG _generate_candidates_batch...')
config = load_config('configs/config_hybrid.yaml')
model = build_model(config, device='cpu')
model.eval()

vocab = Vocabulary()
vocab.build_vocab(['hello world', 'test prompt'])

ids = vocab.encode('hello world', add_special_tokens=False)
ids = [vocab.bos_idx] + ids

# Test _generate_candidates_batch
generated = _generate_candidates_batch(
    model, ids, temps=[0.7, 0.9], max_length=20, top_k=30,
    rep_penalty=1.4, device='cpu',
    ngram_fn=None, ngram_weight=0.0,
    pad_id=vocab.pad_idx, sep_id=vocab.sep_idx, eos_id=vocab.eos_idx
)
print(f'Generated {len(generated)} candidates')
for i, g in enumerate(generated):
    text = vocab.decode(g, skip_special=True)
    print(f'  Candidate {i}: {text[:50]}')

print('IGMCG test passed!')