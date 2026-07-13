#!/usr/bin/env python3
import torch
import json

ckpt = torch.load('checkpoints/model_epoch_99.pt', map_location='cpu', weights_only=True)
print('检查点中的键:')
for key in ckpt.keys():
    if isinstance(ckpt[key], dict):
        print(f'  - {key} (dict, {len(ckpt[key])} items)')
    elif isinstance(ckpt[key], torch.Tensor):
        print(f'  - {key} (tensor, {ckpt[key].shape})')
    else:
        print(f'  - {key} ({type(ckpt[key]).__name__})')
