import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch
from models.config_loader import load_config, build_model
from models.data_utils import load_data

config = load_config('configs/pretrain.yaml')
print('Train file:', config['data']['train_file'])
print('Max seq len:', config['data']['max_seq_length'])
print('Vocab size:', config['data']['vocab_size'])

dataset, vocab = load_data(config['data']['train_file'], config['data']['vocab_size'], config['data']['max_seq_length'])
print(f'Dataset size: {len(dataset)}')
print(f'Vocab size: {len(vocab)}')
print(f'Vocab pad_idx: {vocab.pad_idx}')
print(f'Vocab bos_idx: {vocab.bos_idx}, eos_idx: {vocab.eos_idx}')

# Check a sample
sample = dataset[0]
print(f'Sample input_ids shape: {sample["input_ids"].shape}')
print(f'Sample target_ids shape: {sample["target_ids"].shape}')
print(f'Sample input_ids: {sample["input_ids"][:20]}')
print(f'Sample target_ids: {sample["target_ids"][:20]}')

# Decode first few tokens
decoded = vocab.decode(sample['input_ids'][:50].tolist(), skip_special=True)
print(f'Decoded sample: {decoded[:100]}')
