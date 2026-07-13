import sys
sys.path.insert(0, r'F:\Projects\新项目')
import torch
from models.config_loader import load_config, build_model
from models.data_utils import load_data, create_dataloader, split_dataset, Vocabulary
from models.device import get_device, apply_cpu_threads
from scripts.train import train_epoch

print('Testing train.py flow...')
config = load_config('configs/config_test.yaml')
device = torch.device('cpu')
model = build_model(config, device=device)
print(f'Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params')

# Test data loading
dataset, vocab = load_data(
    config['data']['train_file'],
    vocab_size=config['data']['vocab_size'],
    max_seq_length=config['data']['max_seq_length']
)
print(f'Dataset size: {len(dataset)}, vocab size: {len(vocab)}')

# Test train_epoch (with minimal setup)
from torch.optim import AdamW
import torch.nn as nn

optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])
criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

dataloader = create_dataloader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

# Test one epoch
model.train()
loss = train_epoch(
    model, dataloader, optimizer, criterion, device, epoch=1,
    warmup_steps=0, base_lr=config['training']['learning_rate'],
    gradient_clip=config['training']['gradient_clip'], scaler=None,
    use_amp=False, autocast_dtype=torch.float32, grad_accum_steps=1,
    lr_schedule='cosine', eta_min=0.0, wsd_decay_frac=0.1,
    show_progress=False
)
print(f'Train epoch completed, loss: {loss:.4f}')
print('Train flow works!')