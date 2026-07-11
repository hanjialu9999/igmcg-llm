#!/usr/bin/env python3
"""快速测试最新模型的生成效果"""

import torch
import json
import yaml
from pathlib import Path
from models.transformer import TransformerModel
from models.data_utils import Vocabulary

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load config
with open('configs/pretrain.yaml') as f:
    cfg = yaml.safe_load(f)

# Load vocab
with open('checkpoints/vocab.json') as f:
    vdata = json.load(f)

vocab = Vocabulary(vocab_size=cfg['model']['vocab_size'])
vocab.word2idx = vdata['word2idx']
vocab.idx2word = vdata['idx2word']

# Load model
model = TransformerModel(
    vocab_size=cfg['model']['vocab_size'],
    embedding_dim=cfg['model']['embedding_dim'],
    num_heads=cfg['model']['num_heads'],
    num_layers=cfg['model']['num_layers'],
    hidden_dim=cfg['model']['hidden_dim'],
    max_seq_length=cfg['data']['max_seq_length'],
    dropout=cfg['model']['dropout']
)

# Find latest checkpoint
checkpoint_dir = Path('checkpoints')
epoch_files = sorted(checkpoint_dir.glob('model_epoch_*.pt'))

if epoch_files:
    latest = epoch_files[-1]
    cp = torch.load(latest, map_location=device)
    
    # 从checkpoint中获取词汇表大小
    vocab_size_in_ckpt = cp.get('vocab_size', cfg['model']['vocab_size'])
    
    # 创建匹配checkpoint大小的模型
    model = TransformerModel(
        vocab_size=vocab_size_in_ckpt,
        embedding_dim=cfg['model']['embedding_dim'],
        num_heads=cfg['model']['num_heads'],
        num_layers=cfg['model']['num_layers'],
        hidden_dim=cfg['model']['hidden_dim'],
        max_seq_length=cfg['data']['max_seq_length'],
        dropout=cfg['model']['dropout']
    )
    
    model.load_state_dict(cp['model_state_dict'])
    model.to(device).eval()
    
    epoch_num = latest.name.split('_')[2].split('.')[0]
    
    print(f"\n{'='*75}")
    print(f"🧪 改进模型生成效果测试 - Epoch {epoch_num}")
    print(f"{'='*75}\n")
    
    test_prompts = [
        "What is artificial intelligence",
        "How does neural network work",
        "Machine learning is",
        "Python programming helps with",
        "The difference between AI and machine learning",
        "Can you explain deep learning",
        "Tell me about computer science"
    ]
    
    for prompt in test_prompts:
        tokens = vocab.encode(prompt, add_special_tokens=False)
        tokens = [vocab.bos_idx] + tokens
        
        with torch.no_grad():
            output_ids = model.generate(
                tokens,
                max_length=40,
                temperature=0.8,
                top_k=60,
                device=device,
                repetition_penalty=2.5
            )
        
        response = vocab.decode(output_ids, skip_special=True)
        input_text = vocab.decode(tokens, skip_special=True)
        
        # Remove input from response
        if response.startswith(input_text):
            response = response[len(input_text):].strip()
        
        # Remove trailing punctuation for cleaner display
        response = response.strip('.!?,;')
        
        print(f"Q: {prompt}")
        print(f"A: {response[:110]}")
        print()
    
    print(f"{'='*75}")
    print(f"✅ 测试完成 (词汇表大小: {len(vocab.word2idx)}, Batch size: 16)")
    print(f"{'='*75}\n")
else:
    print("❌ 没有找到checkpoint")
