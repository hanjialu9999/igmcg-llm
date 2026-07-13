import sys
sys.path.insert(0, r'F:\Projects\igmcg-llm')
import torch
from models.config_loader import load_config, build_model, load_vocab

config = load_config('configs/pretrain.yaml')
model = build_model(config, device='cpu')
checkpoint = torch.load('checkpoints/final_model.pt', map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint)
vocab = load_vocab('checkpoints/vocab.json')
print(f'Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params')
print(f'Vocab: {len(vocab)} tokens')

model.eval()
prompts = ['你好', '今天天气怎么样', '什么是人工智能']
for prompt in prompts:
    tokens = vocab.encode(prompt, add_special_tokens=False)
    tokens = [vocab.bos_idx] + tokens
    with torch.no_grad():
        generated = model.generate(tokens, max_length=60, device='cpu', temperature=0.5, top_k=20, repetition_penalty=2.5, min_length=15)
    text = vocab.decode(generated, skip_special=True)
    print(f'Q: {prompt}')
    print(f'A: {text}')
    print()