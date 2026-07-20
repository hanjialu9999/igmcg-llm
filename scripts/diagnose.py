"""
诊断脚本 - 检查模型输出
"""

import torch
import json
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config_loader import load_vocab
from models.device import get_device
from scripts.generate import load_model

# Load model and vocab
parser = argparse.ArgumentParser(description='诊断模型输出')
parser.add_argument('--model', default='checkpoints/final_model.pt', help='模型权重路径')
parser.add_argument('--vocab', default='checkpoints/vocab.json', help='词表路径')
parser.add_argument('--device', default=None, help='推理设备（默认自动选择，如 cpu / cuda / dml）')
parser.add_argument('--prompt', default='Hello world', help='测试输入文本')
args = parser.parse_args()

device = get_device(args.device)
print(f"Device: {device}")

model_path = args.model
vocab_path = args.vocab

# Load vocab（统一走 BaseTokenizer 系，与推理 load_vocab 一致）
vocab = load_vocab(vocab_path)

# Load model（复用 generate.load_model：统一安全加载 weights_only=True +
# 白名单全局放行，且自动从 *_config.yaml 透传 qk_norm/attn_temp/residual_gate/
# hybrid_gate 增强开关，避免增强权重 state_dict 不匹配）
model, _ = load_model(model_path, vocab_path, device=device, quantize=False, compile_model=False)
model.eval()

print("Model loaded successfully!")
print(f"Model vocab size: {model.vocab_size}")
print(f"Vocab size: {len(vocab)}")
print()

# Test
prompt = args.prompt
tokens = vocab.encode(prompt)
print(f"Prompt: {prompt}")
print(f"Token IDs: {tokens}")
print(f"Decoded back: {vocab.decode(tokens)}")
print()

# Manual generation step
print("Manual generation test:")
print("="*50)

with torch.no_grad():
    # First step
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    print(f"Input shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids}")
    
    logits = model(input_ids)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits min/max: {logits.min():.4f} / {logits.max():.4f}")
    
    # Get next token
    next_logits = logits[0, -1, :]
    probs = torch.softmax(next_logits, dim=-1)
    
    print(f"\nTop 10 prob tokens:")
    top_probs, top_indices = torch.topk(probs, 10)
    for prob, idx in zip(top_probs, top_indices):
        word = vocab.idx2word.get(idx.item(), '<unk>')
        print(f"  {word:<20} {prob:.4f}")
    
    # Sample next token
    next_token = torch.multinomial(probs, 1).item()
    print(f"\nSampled next token: {next_token} ({vocab.idx2word.get(next_token, '<unk>')})")
    print()

print("="*50)
print("\nConclusion:")
print("If the model generates mostly <pad> or <eos> tokens, it hasn't learned well.")
print("If it generates relevant words, the model is learning correctly.")
