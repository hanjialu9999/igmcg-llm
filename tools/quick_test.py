#!/usr/bin/env python3
import torch
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config_loader import load_config, build_model, load_vocab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = load_config()
vocab = load_vocab('checkpoints/vocab.json')

model = build_model(config)

checkpoint_dir = Path('checkpoints')
epoch_files = sorted(checkpoint_dir.glob('model_epoch_*.pt'))

if epoch_files:
    latest = epoch_files[-1]
    checkpoint = torch.load(latest, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    epoch_num = latest.name.split('_')[2].split('.')[0]
    
    prompts = [
        "What is artificial intelligence",
        "How does neural network work",
        "Tell me about programming"
    ]
    
    print(f"\n=== Model Epoch {epoch_num} ===\n")
    
    for prompt in prompts:
        tokens = vocab.encode(prompt, add_special_tokens=False)
        tokens = [vocab.bos_idx] + tokens
        
        with torch.no_grad():
            output_ids = model.generate(tokens, max_length=30, temperature=0.8, top_k=50, device=device)
        
        response = vocab.decode(output_ids, skip_special=True)
        input_text = vocab.decode(tokens, skip_special=True)
        
        if response.startswith(input_text):
            response = response[len(input_text):].strip()
        
        print(f"Q: {prompt}")
        print(f"A: {response[:100]}")
        print()
else:
    print("No checkpoints found")
