import torch
import json
import re
import os
from models.transformer import TransformerModel
from models.config_loader import load_config as load_model_config, build_model, load_vocab

# ===== 1. 环境配置 =====
MODEL_PATH = "best_finetuned_model.pt" 
VOCAB_PATH = "checkpoints/vocab.json"
CONFIG_PATH = "chat_config.json"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {"temperature": 0.8, "top_k": 50, "repetition_penalty": 1.3, 
            "min_new_tokens": 10, "max_new_tokens": 40, "context_rounds": 3}

# ===== 2. 核心逻辑 =====
def clean_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    return re.sub(r"\s+", " ", text).strip()

def load_all():
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    w2i = vocab_data['word2idx']
    i2w = {v: k for k, v in w2i.items()}

    # 模型结构统一从 config/config.yaml 读取，避免与训练脚本不一致
    model = build_model(load_model_config()).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model, w2i, i2w

def generate_answer(model, w2i, i2w, history):
    cfg = load_config()
    
    # 组合上下文
    context_str = " [SEP] ".join(history[-cfg['context_rounds']*2:])
    tokens = [w2i.get(w, w2i.get('<unk>')) for w in clean_text(context_str).split()]
    
    # 编码并对齐 64 位
    input_ids = [w2i.get('<bos>')] + tokens + [w2i.get('<eos>')]
    if len(input_ids) < 64:
        input_ids += [w2i.get('<pad>')] * (64 - len(input_ids))
    else:
        input_ids = input_ids[-64:]

    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
    generated = []
    
    eos_id = w2i.get('<eos>')
    unk_id = w2i.get('<unk>', 1)
    pad_id = w2i.get('<pad>', 0)

    with torch.no_grad():
        # 获取所有位置的 logits (batch, seq_len, vocab)
        outputs = model(input_tensor) 
        logits_seq = outputs[0]

        # 逐个 token 采样，但受 max_new_tokens 限制
        for i in range(min(cfg['max_new_tokens'], 64)):
            logits = logits_seq[i] / cfg['temperature']
            
            # 重复惩罚
            for prev_id in set(generated):
                logits[prev_id] /= cfg['repetition_penalty'] if logits[prev_id] > 0 else (1/cfg['repetition_penalty'])

            # 屏蔽无用词
            logits[unk_id] = -1e9
            logits[pad_id] = -1e9
            
            # 最小长度限制：没到 min_new_tokens 时禁止输出结束符
            if len(generated) < cfg['min_new_tokens']:
                logits[eos_id] = -1e9
            
            # Top-K 采样
            v, _ = torch.topk(logits, cfg['top_k'])
            logits[logits < v[-1]] = -1e9
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token == eos_id and len(generated) >= cfg['min_new_tokens']:
                break
            generated.append(next_token)
    
    res = [i2w.get(idx, '') for idx in generated if i2w.get(idx, '') not in ['<pad>', '<bos>', '<eos>', '<unk>']]
    return " ".join(res)

# ===== 3. 运行界面 =====
if __name__ == "__main__":
    model, w2i, i2w = load_all()
    history = []
    
    print("\n" + "★"*20)
    print("  布丁的 Transformer 终极版")
    print("  输入 'reset' 清空记忆 | 'quit' 退出")
    print("★"*20)

    while True:
        user_input = input("\n布丁: ")
        
        if user_input.lower() == 'quit': break
        if user_input.lower() == 'reset':
            history = []
            print("🧹 记忆已重置！")
            continue
        
        history.append(user_input)
        
        # 打印上下文预览
        context_preview = " -> ".join(history[-3:])
        print(f"\033[90m🔍 记忆栈: {context_preview}\033[0m")
        
        reply = generate_answer(model, w2i, i2w, history)
        history.append(reply)
        print(f"AI: {reply}")