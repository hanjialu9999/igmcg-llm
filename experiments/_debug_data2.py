import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch
from models.config_loader import load_config, build_model
from models.data_utils import load_data
from collections import Counter

config = load_config('configs/pretrain.yaml')
dataset, vocab = load_data(config['data']['train_file'], config['data']['vocab_size'], config['data']['max_seq_length'])

# Check the actual tokens in the first sample
sample = dataset[0]
input_ids = sample['input_ids']
target_ids = sample['target_ids']

print("First 20 input_ids:", input_ids[:20].tolist())
print("First 20 target_ids:", target_ids[:20].tolist())

# Check if target_ids are actually shifted by 1
print("Are targets shifted by 1?", torch.equal(input_ids[1:], target_ids[:-1]))

# Check unique tokens in the sample
unique_input = torch.unique(input_ids)
print(f"Unique tokens in sample: {len(unique_input)}")
print(f"Min token: {unique_input.min()}, Max token: {unique_input.max()}")

# Check if there are many repeated tokens
counter = Counter(input_ids.tolist())
print(f"Most common tokens: {counter.most_common(20)}")

# Check the decoded text with different approaches
decoded = vocab.decode(input_ids.tolist(), skip_special=True)
print(f"Decoded (skip_special=True): {decoded[:200]}")
