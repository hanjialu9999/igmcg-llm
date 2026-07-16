import sys, os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))

# 只重设 stdout 为 UTF-8（避免中文在 GBK 控制台打印时崩溃）。
# stdin 保留控制台原生编码：用户在 GBK 或 UTF-8 终端直接打字都能正确解码，无需改。
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import torch
from scripts.generate import load_model

DEFAULT_MODEL = str(project_root / 'checkpoints_dml' / 'final_model.pt')
DEFAULT_VOCAB = str(project_root / 'checkpoints_dml' / 'vocab.json')


def de_space(s):
    """去掉字符级模型输出里每字之间的空格，还原成正常中文（便于阅读）。"""
    return s.replace(' ', '')


def chat_generate(model, vocab, prompt, max_length, temperature, top_k, device):
    tokens = vocab.encode(prompt)
    if tokens and tokens[-1] == vocab.eos_idx:
        tokens = tokens[:-1]
    with torch.no_grad():
        generated = model.generate(tokens, max_length=max_length,
                                   temperature=temperature, top_k=top_k,
                                   device=device, repetition_penalty=1.4,
                                   min_length=8, eos_penalty=-5.0)
    new_ids = generated[len(tokens):]
    return vocab.decode(new_ids).strip()


def main():
    import argparse
    ap = argparse.ArgumentParser(description='与训练好的基础模型对话（字符级，64 上下文）')
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--vocab', default=DEFAULT_VOCAB)
    ap.add_argument('--device', default='cpu', help='cpu / cuda / dml')
    ap.add_argument('--max-length', type=int, default=64, help='总计上下文长度（含输入）')
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--top-k', type=int, default=40)
    args = ap.parse_args()

    device = str(args.device)
    print('加载模型中…')
    model, vocab = load_model(args.model, args.vocab, device=device)
    model.eval()
    print('模型已加载。在「你> 」后输入中文即可对话（输入 exit / quit 退出）。')
    print('提示：这是基础语言模型（非聊天微调），它做的是「续写」而非真正理解；上下文仅 64 字。\n')

    log_path = project_root / 'logs' / 'chat_log.txt'
    log_path.parent.mkdir(exist_ok=True)

    try:
        while True:
            try:
                user = input('你> ').strip()
            except EOFError:
                break
            if not user:
                continue
            if user.lower() in ('exit', 'quit', 'q'):
                print('再见！')
                break
            if len(user) > 40:
                user = user[:40]
            prompt = f'用户：{user}\n助手：'
            raw = chat_generate(model, vocab, prompt, args.max_length,
                                args.temperature, args.top_k, device)
            clean = de_space(raw)
            print(f'模型> {clean}')
            print(f'     (原文: {raw})\n')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f'你> {user}\n模型(原文)> {raw}\n模型(净)> {clean}\n\n')
    except KeyboardInterrupt:
        print('\n再见！')


if __name__ == '__main__':
    main()
