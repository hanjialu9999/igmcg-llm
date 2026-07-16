"""50MB DML 训练产物的生成验证脚本：打印原始 token id 与解码文本。

用法:
    python scripts/gen_50mb.py --prompt "人工" --temp 0.8 --topk 30 --max 60
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, '.')

from models.config_loader import build_model, load_config, load_vocab
from models.device import get_device
from scripts.generate import _safe_torch_load


def main() -> None:
    ap = argparse.ArgumentParser(description='50MB 训练产物生成验证')
    ap.add_argument('--config', default='configs/config_char_50mb_dml.yaml')
    ap.add_argument('--ckpt', default='checkpoints_50mb_dml/final_model.pt')
    ap.add_argument('--vocab', default='checkpoints_50mb_dml/vocab.json')
    ap.add_argument('--prompt', default='人工')
    ap.add_argument('--temp', type=float, default=0.8)
    ap.add_argument('--topk', type=int, default=30)
    ap.add_argument('--max', type=int, default=60)
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    cfg = load_config(args.config)
    device = get_device('auto')
    model = build_model(cfg, device=device)
    sd = _safe_torch_load(args.ckpt)
    model.load_state_dict(sd['model_state_dict'])
    model.eval()
    vocab = load_vocab(args.vocab)

    ids = [vocab.bos_idx] + vocab.encode(args.prompt, add_special_tokens=False)
    if len(ids) <= 1:
        ids = [vocab.bos_idx]
    out = model.generate(ids, max_length=args.max, temperature=args.temp,
                         top_k=args.topk, device=device.type,
                         repetition_penalty=1.7, min_length=3, eos_penalty=-5.0)
    gen_ids = out[len(ids):]
    text = vocab.decode(out, skip_special=True)
    # 直接按 idx2word 取每个生成 token 的字符，绕开控制台 GBK 显示 ? 的问题
    chars = []
    for t in gen_ids:
        w = vocab.idx2word.get(int(t), '?')
        chars.append(w if isinstance(w, str) else str(w))
    # 写 UTF-8 文件，避免 PowerShell GBK 终端把中文渲染成 ?
    with open('logs/gen_50mb_out.txt', 'w', encoding='utf-8') as gf:
        gf.write('PROMPT            : ' + args.prompt + '\n')
        gf.write('PROMPT_TOKEN_IDS  : ' + str(ids) + '\n')
        gf.write('GEN_TOKEN_IDS     : ' + str(gen_ids) + '\n')
        gf.write('GEN_CHARS         : ' + ''.join(chars) + '\n')
        gf.write('DECODED_TEXT      : ' + text.strip() + '\n')
    print('PROMPT            :', args.prompt)
    print('GEN_TOKEN_IDS     :', gen_ids)
    print('DECODED_TEXT      :', text.strip())
    print('(结果已写入 logs/gen_50mb_out.txt，UTF-8 可读)')


if __name__ == '__main__':
    main()
