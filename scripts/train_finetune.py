import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
import glob
import sys
from pathlib import Path

# 注入项目根目录，确保可 import models（脚本位于 scripts/，上一级即根）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.transformer import TransformerModel
from models.config_loader import load_config, build_model, load_vocab
from models.device import get_device, apply_cpu_threads
from models.utils import save_checkpoint, cli_guard


def load_vocab_from_json(vocab_path):
    """从 vocab.json 加载词表（复用 config_loader.load_vocab，正确处理 BPE/char 词表的 merges 等字段）。"""
    return load_vocab(vocab_path)

# ===== 2. 处理问答数据 (强制对齐长度) =====
class QADataset(Dataset):
    def __init__(self, data_folder, vocab, max_length=64):
        self.vocab = vocab
        self.max_length = max_length
        self.qa_pairs = []
        
        file_pattern = os.path.join(data_folder, "*.txt")
        files = glob.glob(file_pattern)
        
        print(f"找到 {len(files)} 个数据文件")
        
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines()]
            
            for i in range(0, len(lines)-1, 2):
                question = lines[i]
                answer = lines[i+1]
                if question and answer:
                    self.qa_pairs.append((question, answer))
        
        print(f"总共加载了 {len(self.qa_pairs)} 对问答")
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        question, answer = self.qa_pairs[idx]
        # 拼成单条序列 [BOS] 问题 [SEP] 答案 [EOS]，做标准 next-token 语言模型训练：
        # 原实现 input=问题 / target=答案 在因果 LM 下位置错位（output[t] 预测的是问题[t+1] 而非答案[t]），
        # 这里用整条序列错位一位得到输入/目标，与 data_utils.TextDataset 的构造方式一致
        seq = ([self.vocab.bos_idx] + self.vocab.encode(question, add_special_tokens=False)
               + [self.vocab.sep_idx]
               + self.vocab.encode(answer, add_special_tokens=False) + [self.vocab.eos_idx])
        # 截断到 max_length+1，再拆成输入/目标（错位一位）
        if len(seq) > self.max_length + 1:
            seq = seq[:self.max_length + 1]
        elif len(seq) < self.max_length + 1:
            seq = seq + [self.vocab.pad_idx] * (self.max_length + 1 - len(seq))

        return {
            'input_ids': torch.tensor(seq[:-1], dtype=torch.long),
            'target_ids': torch.tensor(seq[1:], dtype=torch.long)
        }

# ===== 3. 微调训练 =====
def train():
    # 路径设置（统一存入 checkpoints/，避免模型文件散落各处）
    model_path = "checkpoints/best_finetuned_model.pt"
    vocab_path = "checkpoints/vocab.json"
    data_folder = "data/datasets"

    print("加载tokenizer...")
    vocab = load_vocab_from_json(vocab_path)

    # 模型结构统一从 config.yaml 读取，避免与训练脚本不一致
    print("初始化 Transformer 结构...")
    config = load_config()
    device = get_device()  # 自动适配 CUDA / DirectML(AMD) / CPU
    model = build_model(config, device=device)

    # 加载权重：优先用微调后的模型；不存在则回退到预训练底座 final_model.pt
    # （支持「先预训练 base 再微调」的两阶段流程）；都缺失则从随机初始化开始
    if not os.path.exists(model_path):
        fallback = "checkpoints/final_model.pt"
        if os.path.exists(fallback):
            print(f"未找到 {model_path}，改用预训练底座 {fallback}")
            model_path = fallback
        else:
            print(f"WARNING 权重文件不存在（{model_path} / {fallback}），将从随机初始化开始训练")

    if os.path.exists(model_path):
        print("加载权重参数...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("OK 从 checkpoint 成功加载权重")
        else:
            model.load_state_dict(checkpoint)
            print("OK 直接加载权重成功")

    apply_cpu_threads(config['training'].get('cpu_threads'))
    model.train()
    print(f"使用设备: {device}")

    # 准备数据 (max_length 设为 64，与 config 一致)
    dataset = QADataset(data_folder, vocab, max_length=64)
    # batch_size 设为 16，显存如果炸了就改小
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    # 忽略 padding 部分的 loss
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    num_epochs = 10  # 原 200 轮过多，改为 10 轮
    print(f"\n开始微调，共 {num_epochs} 个epoch...\n")

    best_loss = float('inf')

    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)

            optimizer.zero_grad()

            # 前向传播 (batch, 64, vocab_size)
            outputs = model(input_ids)

            # 计算loss: 需要把 outputs 展平为 (batch*64, vocab_size)
            # target_ids 展平为 (batch*64)
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        print(f"Epoch {epoch+1} 完成，平均Loss: {avg_loss:.4f}")

        # 保存最佳权重
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch+1, best_loss, "checkpoints", len(vocab), config['model'])
            print(f"✨ 已更新最佳权重文件: checkpoints/best_finetuned_model.pt")

    print("\n🎉 微调任务圆满成功！")

if __name__ == "__main__":
    try:
        train()
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)