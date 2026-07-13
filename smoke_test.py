import sys
sys.path.insert(0, r'F:\Projects\新项目')
import torch
from models.config_loader import load_config, build_model
from models.data_utils import Vocabulary

print('Testing basic model load...')
config = load_config('configs/pretrain.yaml')
model = build_model(config, device='cpu')
print(f'Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

print('Testing hybrid model...')
hybrid_config = load_config('configs/config_hybrid.yaml')
hybrid_model = build_model(hybrid_config, device='cpu')
print(f'Hybrid model params: {sum(p.numel() for p in hybrid_model.parameters())/1e6:.2f}M')

print('Testing generation...')
vocab = Vocabulary()
vocab.build_vocab(['hello world', 'test'])
tokens = [2] + vocab.encode('hello')  # bos + prompt
with torch.no_grad():
    generated = hybrid_model.generate(tokens, max_length=10, device='cpu')
print(f'Generated: {generated}')
print('All smoke tests passed!')