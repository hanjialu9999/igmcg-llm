import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch
from models.config_loader import load_config, build_model, load_vocab

config = load_config('configs/pretrain.yaml')
model = build_model(config, device='cpu')
ckpt = torch.load('archive_unused/checkpoints_backup/checkpoints_4k/final_model.pt', map_location='cpu', weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
vocab = load_vocab('archive_unused/checkpoints_backup/checkpoints_4k/vocab.json')
print('Model loaded from checkpoints_4k')

prompts = ['你好', '今天天气怎么样', '什么是人工智能']
for prompt in prompts:
    tokens = vocab.encode(prompt, add_special_tokens=False)
    tokens = [vocab.bos_idx] + tokens
    with torch.no_grad():
        generated = model.generate(tokens, max_length=30, device='cpu', temperature=0.7, top_k=30, repetition_penalty=2.0, min_length=10)
        text = vocab.decode(generated, skip_special=True)
        print(f'Q: {prompt}')
        print(f'A: {text}')
        print()
