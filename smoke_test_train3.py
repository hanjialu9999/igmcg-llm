import sys
sys.path.insert(0, r'F:\Projects\新项目')
import torch
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from models.config_loader import load_config, build_model
from models.data_utils import load_data, create_dataloader
from scripts.train import train_epoch
from models.data_utils import Vocabulary
from torch.optim import AdamW
import torch.nn as nn

config = load_config('configs/config_test.yaml')
device = torch.device('cpu')
model = build_model(config, device=device)
print(f'Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

dataset, vocab = load_data(config['data']['train_file'], config['data']['vocab_size'], config['data']['max_seq_length'])
print(f'Dataset: {len(dataset)}, vocab: {len(vocab)}')

model.train()
optimizer = AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

dataloader = create_dataloader(dataset, batch_size=2, shuffle=True, num_workers=0)

from scripts.train import train_epoch

loss = train_epoch(
    model, dataloader, AdamW(model.parameters(), lr=1e-3), 
    nn.CrossEntropyLoss(ignore_index=vocab.pad_idx), 
    torch.device('cpu'), epoch=1,
    warmup_steps=0, base_lr=1e-3, gradient_clip=1.0, scaler=None,
    use_amp=False, autocast_dtype=torch.float32, grad_accum_steps=1,
    lr_schedule='cosine', eta_min=0.0, wsd_decay_frac=0.1,
    show_progress=False
)
print(f'Train epoch completed, loss: {loss:.4f}')
print('Train flow works!')