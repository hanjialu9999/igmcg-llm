import os
import sys
import re
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 复用 generate.py 的模型加载与生成逻辑
from generate import load_model, generate_text, NGramModel
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
    parser.add_argument('--model', default='checkpoints/final_model.pt')
    parser.add_argument('--vocab', default='checkpoints/vocab.json')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--cpu-threads', type=int, default=4,
                        help='CPU 生成时使用的线程数（降功耗）')
    parser.add_argument('--max-length', type=int, default=60)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-k', type=int, default=30)
    # 以下为参数扫描(粗扫+细扫+连贯性评估)得到的最优生成参数，已自动写入默认值
    parser.add_argument('--repetition-penalty', type=float, default=1.7)
    parser.add_argument('--ngram', action='store_true',
                        help='解码期融合 Bigram/Trigram 统计先验（神经+统计双轨）')
    parser.add_argument('--ngram-corpus', default='data/pretrain_corpus/merged.txt',
                        help='构建 n-gram 统计所用的语料文件')
    parser.add_argument('--ngram-weight', type=float, default=0.3,
                        help='n-gram 先验叠加权重（0=关闭）')
    parser.add_argument('--igmcg', action='store_true',
                        help='启用 IGMCG 直觉引导解码；与 --ngram 同开即为 n-gram+IGMCG 联合解码')
    parser.add_argument('--igmcg-candidates', type=int, default=5)
    parser.add_argument('--intuition', type=str, default='0.5,0.5,0.5,0.5,0.5,0.5,0.5',
                        help='IGMCG 7 维直觉向量(逗号分隔, 0~1)')
    parser.add_argument('--history', default='logs/chat_history.txt')
    parser.add_argument('--script', default=None,
                        help='UTF-8 文件，每行一个 prompt，非交互式跑完一轮对话（便于测试/避免控制台编码问题）')
    args = parser.parse_args()

    # CPU 生成时限制线程数以降功耗
    import torch as _torch
    if args.cpu_threads and args.cpu_threads > 0:
        _torch.set_num_threads(max(1, args.cpu_threads))

    print("加载模型中...")
    device = __import__('models.device', fromlist=['get_device']).get_device(args.device)
    model, vocab = load_model(args.model, args.vocab, device=device)
    ngram = None
    if args.ngram:
        print(f"构建 n-gram 模型（{args.ngram_corpus}）...")
        ngram = NGramModel(vocab, args.ngram_corpus, max_order=3, smoothing=1.0)
        print(f"n-gram 就绪（权重 {args.ngram_weight}）。")
    print(f"模型加载完成。词表大小 {len(vocab)}。输入 'quit' 或 '退出' 结束。")

    os.makedirs(os.path.dirname(args.history), exist_ok=True)
    hist_f = open(args.history, 'w', encoding='utf-8')

    def reply(prompt):
        ids = vocab.encode(prompt, add_special_tokens=False)
        if args.igmcg:
            gen, _ = generate_igmcg(model, vocab, prompt, max_length=args.max_length,
                                    temperature=args.temperature, top_k=args.top_k,
                                    device=device, num_candidates=args.igmcg_candidates,
                                    intuition=[float(x) for x in args.intuition.split(',')],
                                    ngram_fn=(ngram.logprob_vector if ngram else None),
                                    ngram_weight=args.ngram_weight,
                                    repetition_penalty=args.repetition_penalty)
            new_ids = vocab.encode(prompt, add_special_tokens=False)
            new_ids = vocab.encode(gen, add_special_tokens=False)[len(new_ids):]
            text = vocab.decode(new_ids, skip_special=True)
            return clean_text(text.strip())
        gen = model.generate(ids, max_length=args.max_length,
                             temperature=args.temperature, top_k=args.top_k,
                             repetition_penalty=args.repetition_penalty, device=device,
                             ngram_fn=(ngram.logprob_vector if ngram else None),
                             ngram_weight=args.ngram_weight)
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
