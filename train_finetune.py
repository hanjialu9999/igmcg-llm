import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
import glob

# 尝试导入模型类
try:
    from models.transformer import TransformerModel
    from models.config_loader import load_config, build_model
    from models.device import get_device, apply_cpu_threads
    from models.data_utils import Vocabulary
except ImportError:
    print("❌ 错误：在当前目录下找不到 models/transformer.py，请确保在项目根目录下运行脚本")
    exit()


def load_vocab_from_json(vocab_path):
    """从 vocab.json 加载词表并构建 Vocabulary 对象。
    
    使用项目自带的 Vocabulary 类（支持 CJK 逐字切分），替代原 SimpleTokenizer 的按空格分词。
    """
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    word2idx = vocab_data['word2idx']
    idx2word = {v: k for k, v in word2idx.items()}
    
    # 实例化 Vocabulary，再手动注入已有映射（避免重新 build_vocab）
    vocab = Vocabulary(vocab_size=len(word2idx))
    vocab.word2idx = word2idx
    vocab.idx2word = idx2word
    
    # 特殊 token 索引对齐（Vocabulary 默认顺序：pad=0, unk=1, bos=2, eos=3, sep=4）
    vocab.pad_idx = word2idx.get('<pad>', 0)
    vocab.unk_idx = word2idx.get('<unk>', 1)
    vocab.bos_idx = word2idx.get('<bos>', 2)
    vocab.eos_idx = word2idx.get('<eos>', 3)
    vocab.sep_idx = word2idx.get('[SEP]', 4)
    
    return vocab

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
        input_ids = self.vocab.encode(question)
        target_ids = self.vocab.encode(answer)
        
        # --- 核心修复：强制对齐输入和目标的长度 ---
        # 1. 对输入进行填充/截断
        if len(input_ids) < self.max_length:
            input_ids += [self.vocab.pad_idx] * (self.max_length - len(input_ids))
        else:
            input_ids = input_ids[:self.max_length]
            
        # 2. 对目标进行填充/截断 (必须和输入一样长，否则 Loss 计算会报错)
        if len(target_ids) < self.max_length:
            target_ids += [self.vocab.pad_idx] * (self.max_length - len(target_ids))
        else:
            target_ids = target_ids[:self.max_length]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long)
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
            print(f"⚠️ 权重文件不存在（{model_path} / {fallback}），将从随机初始化开始训练")

    if os.path.exists(model_path):
        print("加载权重参数...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ 从 checkpoint 成功加载权重")
        else:
            model.load_state_dict(checkpoint)
            print("✅ 直接加载权重成功")

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

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 完成，平均Loss: {avg_loss:.4f}")

        # 保存最佳权重
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/best_finetuned_model.pt")
            print(f"✨ 已更新最佳权重文件: checkpoints/best_finetuned_model.pt")

    print("\n🎉 微调任务圆满成功！")

if __name__ == "__main__":
    train()