import sys
sys.path.insert(0, r'F:\Projects\新项目')
import torch
from models.config_loader import load_config, build_model
from models.data_utils import load_data, create_dataloader, split_dataset, Vocabulary
from models.device import get_device, apply_cpu_threads

print('Testing train.py flow...')
config = load_config('configs/pretrain.yaml')
device = torch.device('cpu')  # Force CPU to avoid DML issues
model = build_model(config, device=device)
print(f'Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params')

# Test data loading
dataset, vocab = load_data(
    config['data']['train_file'],
    vocab_size=config['data']['vocab_size'],
    max_seq_length=config['data']['max_seq_length']
)
print(f'Dataset size: {len(dataset)}, vocab size: {len(vocab)}')

# Test forward pass
model.eval()
x = torch.randint(0, config['model']['vocab_size'], (2, 32))
with torch.no_grad():
    out = model(x)
print(f'Forward pass output shape: {out.shape}')
print('Train flow works!')