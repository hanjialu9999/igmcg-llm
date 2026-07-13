import sys
sys.path.insert(0, r'F:\Projects\igmcg-llm')
import torch
from models.config_loader import load_config, build_model, load_vocab

config = load_config('configs/pretrain.yaml')
model = build_model(config, device='cpu')
vocab = load_vocab('checkpoints/vocab.json')
print(f'Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params')
print(f'Vocab: {len(vocab)} tokens')

model.eval()
prompts = ['hello', 'how are you', 'machine learning']
for prompt in prompts:
    tokens = vocab.encode(prompt, add_special_tokens=False)
    tokens = [vocab.bos_idx] + tokens
    with torch.no_grad():
        generated = model.generate(tokens, max_length=30, device='cpu', temperature=0.7, top_k=30)
        text = vocab.decode(generated, skip_special=True)
        print(f'Q: {prompt}')
        print(f'A: {text}')
        print()