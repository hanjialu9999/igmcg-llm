import os
import sys
import re
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 复用 generate.py 的模型加载与生成逻辑
from generate import load_model, generate_text
from models.data_utils import Vocabulary

CJK = r'[\u3400-\u9fff\uf900-\ufaff]'

def clean_text(s):
    """去掉两个 CJK 字符之间的空格，让中文更自然；英文词之间保留空格。"""
    return re.sub(r'(?<=' + CJK + r')\s+(?=' + CJK + r')', '', s)

def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    parser = argparse.ArgumentParser(description='与训练好的模型对话（续写式）')
    parser.add_argument('--model', default='checkpoints_test/final_model.pt')
    parser.add_argument('--vocab', default='checkpoints_test/vocab.json')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--max-length', type=int, default=60)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-k', type=int, default=30)
    # 以下为参数扫描(粗扫+细扫+连贯性评估)得到的最优生成参数，已自动写入默认值
    parser.add_argument('--repetition-penalty', type=float, default=1.7)
    parser.add_argument('--history', default='logs/chat_history.txt')
    parser.add_argument('--script', default=None,
                        help='UTF-8 文件，每行一个 prompt，非交互式跑完一轮对话（便于测试/避免控制台编码问题）')
    args = parser.parse_args()

    print("加载模型中...")
    device = __import__('models.device', fromlist=['get_device']).get_device(args.device)
    model, vocab = load_model(args.model, args.vocab, device=device)
    print(f"模型加载完成。词表大小 {len(vocab)}。输入 'quit' 或 '退出' 结束。")

    os.makedirs(os.path.dirname(args.history), exist_ok=True)
    hist_f = open(args.history, 'w', encoding='utf-8')

    def reply(prompt):
        ids = vocab.encode(prompt, add_special_tokens=False)
        gen = model.generate(ids, max_length=args.max_length,
                             temperature=args.temperature, top_k=args.top_k,
                             repetition_penalty=args.repetition_penalty, device=device)
        # 只取 prompt 之后新生成的部分作为"回复"
        new_ids = gen[len(ids):]
        text = vocab.decode(new_ids, skip_special=True)
        return clean_text(text.strip())

    # 非交互式：从 UTF-8 文件逐行对话
    if args.script:
        with open(args.script, 'r', encoding='utf-8-sig') as sf:
            prompts = [ln.strip() for ln in sf if ln.strip()]
        for prompt in prompts:
            if prompt.lower() in ('quit', 'exit', '退出', 'q'):
                print("再见！")
                break
            resp = reply(prompt)
            print(f"你: {prompt}")
            print(f"模型: {resp}")
            hist_f.write(f"你: {prompt}\n模型: {resp}\n\n")
            hist_f.flush()
        hist_f.close()
        print(f"\n对话已写入 {args.history}")
        return

    try:
        while True:
            try:
                prompt = input("你: ").strip()
            except EOFError:
                break
            if not prompt:
                continue
            if prompt.lower() in ('quit', 'exit', '退出', 'q'):
                print("再见！")
                break
            resp = reply(prompt)
            print("模型:", resp)
            hist_f.write(f"你: {prompt}\n模型: {resp}\n\n")
            hist_f.flush()
    finally:
        hist_f.close()

if __name__ == '__main__':
    main()
