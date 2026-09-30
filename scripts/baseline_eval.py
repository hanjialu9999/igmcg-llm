# -*- coding: utf-8 -*-
"""r42 质量基准：三条口径的 ppl + 固定提示词生成文本，落一份 JSON。

为什么必须三种口径（只跑一种会得出错的结论）：
  tf            全序列一次前向（validate/训练同一算法）。注意力虽有因果掩码，但
                整段序列一次性喂进 memory/controller 记忆槽 → logits(t) 可能读到
                t+1..T-1 的记忆内容（H2 泄漏），ppl 偏乐观。
  prefix        逐 t 只喂前缀 x[0:t+1]（use_cache=False）。任何"未来"信息都不存在
                → 无泄漏；卷积/注意力看到的上下文与 tf 同为前缀 → 干净参照。
  incremental   use_cache=True KV 缓存逐 token 推理（真实生成路径）。无泄漏，但每步
                只喂 1 个 token，char_merge 卷积 / ALiBi 距离 / 注意力边界与训练不同
                → 含"路径差异"。

tf − prefix        ≈ 泄漏让指标虚低了多少
prefix − incremental ≈ 增量路径本身的边界差异

⚠ 基准必须在 H1/H3 改动之前跑完：那两处直接改 logits，改完再测就无法归因。
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.checkpoint import load_model
from models.data_utils import load_data, split_dataset
from models.device import get_device, apply_cpu_threads
from models.config_loader import load_config
from scripts.generate import generate_text

# 生成基准的固定提示词（中英各半，覆盖"续写 / 补全 / 对话起头"三种用法）
DEFAULT_PROMPTS = [
    '今天天气真好，我们去',
    '人工智能的未来是',
    'Hello, how are you',
    'The weather is',
    'I love',
    'Machine learning',
    # 追加（2026-09-28）：前 6 条全是"续写"起头，缺最基础的"自我介绍"句式，
    # 字符级模型在此处最容易露馅（称呼 + 冒号 + 名字的固定搭配）。追加在末尾，
    # 保持前 6 条索引不变，与既有 baselines/r42_baseline*.json 可逐条对齐。
    '我叫',
]
# 生成参数写死在 JSON 里（与 H4 的 prompt 分支缺省一致），避免"基准没记参数"这种低级返工
GEN_PARAMS = {
    'max_length': 30, 'temperature': 0.8, 'top_k': 50,
    'repetition_penalty': 1.4, 'min_length': 3, 'eos_penalty': -5.0,
}


def _sha256_head(path, n=16):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:n]


def load_val_set(config, max_seqs=0):
    """复刻训练期的 val 划分（同 train_file / test_split / seed），保证与 val_loss 同分布。"""
    dataset, vocab = load_data(
        config['data']['train_file'],
        vocab_size=config['data']['vocab_size'],
        max_seq_length=config['data']['max_seq_length'],
    )
    test_split = config['data'].get('test_split', 0.0)
    if test_split <= 0:
        raise ValueError('config 的 data.test_split <= 0，没有 val 集可评测')
    _, val_dataset = split_dataset(dataset, train_ratio=1.0 - test_split,
                                   seed=config['seed'])
    n = max_seqs if (max_seqs and max_seqs > 0) else len(val_dataset)
    n = min(n, len(val_dataset))
    if n < len(val_dataset):
        val_dataset = torch.utils.data.Subset(
            val_dataset, list(range(n)))   # 固定前 n 个：顺序确定，可复现
    return val_dataset, vocab


def _new_acc():
    return {'nll_sum': 0.0, 'tokens': 0}


def _finish(acc):
    loss = acc['nll_sum'] / max(1, acc['tokens'])
    return {'nll_sum': round(acc['nll_sum'], 6), 'tokens': acc['tokens'],
            'loss': round(loss, 6), 'ppl': round(float(np.exp(min(loss, 50.0))), 6)}


def _ce(logits_2d, target_1d, ignore_index):
    """reduction='sum'：分子只累有效 token，分母自己数。"""
    n_valid = int((target_1d != ignore_index).sum().item())
    if n_valid == 0:
        return 0.0, 0
    return float(F.cross_entropy(logits_2d, target_1d, ignore_index=ignore_index,
                                 reduction='sum').item()), n_valid


def eval_tf(model, loader, device, ignore_index):
    """全序列一次前向（validate 同路径）。"""
    acc = _new_acc()
    with torch.no_grad():
        for batch in loader:
            x = batch['input_ids'].to(device)
            y = batch['target_ids'].to(device)
            logits = model(x)
            loss, n = _ce(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index)
            acc['nll_sum'] += loss
            acc['tokens'] += n
    return acc


def eval_prefix(model, loader, device, ignore_index):
    """逐 t 只喂前缀 x[0:t+1]：无未来信息 → 无泄漏的干净参照（O(T^2)，慢）。"""
    acc = _new_acc()
    with torch.no_grad():
        for batch in loader:
            x = batch['input_ids'].to(device)
            y = batch['target_ids'].to(device)
            T = x.size(1)
            for t in range(T):
                logits = model(x[:, :t + 1])
                loss, n = _ce(logits[:, t, :].reshape(-1, logits.size(-1)),
                              y[:, t], ignore_index)
                acc['nll_sum'] += loss
                acc['tokens'] += n
    return acc


def eval_incremental(model, loader, device, ignore_index):
    """use_cache=True KV 缓存逐 token 推理：真实生成路径。"""
    acc = _new_acc()
    with torch.no_grad():
        for batch in loader:
            x = batch['input_ids'].to(device)
            y = batch['target_ids'].to(device)
            T = x.size(1)
            past = None
            for t in range(T):
                inp = x[:, :1] if t == 0 else x[:, t:t + 1]
                logits, past = model(inp, past_key_values=past, use_cache=True)
                loss, n = _ce(logits[:, -1, :].reshape(-1, logits.size(-1)),
                              y[:, t], ignore_index)
                acc['nll_sum'] += loss
                acc['tokens'] += n
    return acc


MODES = {'tf': eval_tf, 'prefix': eval_prefix, 'incremental': eval_incremental}


def run_generation(model, vocab, device, prompts, seed):
    """固定 seed + 固定参数逐条生成，把原文整个存下来（ppl 之外的主观质量锚点）。"""
    out = []
    for p in prompts:
        for name, fn_seed in (('sample', seed), ('greedy', seed)):
            random.seed(fn_seed)
            np.random.seed(fn_seed)
            torch.manual_seed(fn_seed)
            kwargs = dict(GEN_PARAMS)
            if name == 'greedy':
                kwargs['temperature'] = 0.0
            text = generate_text(model, vocab, p, device=device, **kwargs)
            out.append({'prompt': p, 'mode': name, 'seed': fn_seed, 'text': text})
    return out


def main():
    ap = argparse.ArgumentParser(description='质量基准（TF / prefix / 增量 ppl + 生成）')
    ap.add_argument('--config', default='configs/config_train_8k_r42.yaml')
    ap.add_argument('--model', default='checkpoints_train_8k_r42/final_model.pt')
    ap.add_argument('--vocab', default='checkpoints_train_8k_r42/vocab.json')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--modes', default='tf,prefix,incremental',
                    help='逗号分隔：tf / prefix / incremental')
    ap.add_argument('--max-seqs', type=int, default=0, help='0=全部 val 序列，>0 取前 N 条')
    ap.add_argument('--batch-size', type=int, default=0, help='0=用 config 的 batch_size')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--controller', choices=['on', 'off'], default='on',
                    help='Controller 总开关。off=与 validate 的 controller_active=False 同一条路'
                         '（三信号 mem/film/direction **全关**），用来把"H2 泄漏"和'
                         '"Controller 增量退化"两个效应拆开')
    ap.add_argument('--direction', choices=['on', 'off'], default='on',
                    help='Controller 的 direction 信号③开关（model.controller.use_direction），'
                         '与 --controller **独立**：--controller off 会把 mem/film 一起关掉，'
                         '本开关只关 direction、film/mem 仍在（= 第三轮"关 direction"消融的设法）')
    ap.add_argument('--char-merge-buffer', choices=['on', 'off'], default='off',
                    help='CharMerge 增量滚动缓冲开关（char_merge_incremental_buffer）。'
                         '默认 off = r42 原行为，逐位可比；on=读取路径改用真实前 pad 个输入，'
                         '不重训。只改层内运行时开关，不动 config/checkpoint。')
    ap.add_argument('--out', default='baselines/r42_baseline.json')
    ap.add_argument('--no-gen', action='store_true', help='跳过生成基准')
    args = ap.parse_args()

    t0 = time.time()
    config = load_config(args.config)
    device = get_device(args.device)
    apply_cpu_threads(int(config.get('training', {}).get('cpu_threads', 0) or 0))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    for m in modes:
        if m not in MODES:
            raise SystemExit(f'未知 mode: {m}（可选 {sorted(MODES)}）')

    print(f'[1/4] 加载模型 {args.model} @ {device}')
    model, vocab = load_model(args.model, args.vocab, device=device)
    model.eval()
    model.set_enhancements_active(True)   # 与 validate 一致，保证和 val_loss 可比
    ctrl_on = args.controller == 'on'
    if getattr(model, 'controller_enabled', False):
        # 必须在 set_enhancements_active 之后设：后者会把 _rt_controller 重置为 True
        model._rt_controller = ctrl_on
    dir_on = args.direction == 'on'
    if getattr(model, 'controller', None) is not None:
        # set_enhancements_active 不碰 use_direction（它只管 _rt_layer_film/_input_highway/
        # _rt_controller 三个模型级开关），所以这里按需覆盖；Controller 总关时本开关无效果，
        # 但仍照实记录，避免把"总关"误读成"只关 direction"。
        model.controller.use_direction = dir_on
    print(f'      Controller = {"on" if ctrl_on else "off"}'
          f' / direction = {"on" if dir_on else "off"}')
    cmb_on = args.char_merge_buffer == 'on'
    if getattr(model, 'char_merge_enabled', False):
        # 运行时切层内开关：r42 config 不加此键（默认 off），故走这里临时开
        model.char_merge.incremental_buffer = cmb_on
        model.char_merge.reset_buffer()   # 换开关后丢掉上一次跑残留的尾巴
    print(f'      CharMerge 增量缓冲 = {"on" if cmb_on else "off"}')
    ignore_index = vocab.pad_idx

    print('[2/4] 加载 val 集（复刻训练期划分）')
    val_dataset, _vocab_from_data = load_val_set(config, args.max_seqs)
    bs = args.batch_size or int(config['training']['batch_size'])
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=False)
    print(f'      val 序列数={len(val_dataset)} batch_size={bs}')

    results = {}
    for m in modes:
        t = time.time()
        print(f'[3/4] 计算 {m} ppl ...')
        model.zero_grad(set_to_none=True)
        acc = MODES[m](model, loader, device, ignore_index)
        results[m] = _finish(acc)
        results[m]['elapsed_s'] = round(time.time() - t, 2)
        print(f'      {m}: loss={results[m]["loss"]:.4f} ppl={results[m]["ppl"]:.3f} '
              f'tokens={results[m]["tokens"]} ({results[m]["elapsed_s"]}s)')

    diff = {}
    if 'tf' in results and 'prefix' in results:
        diff['tf_minus_prefix_loss'] = round(results['tf']['loss'] - results['prefix']['loss'], 6)
        diff['tf_over_prefix_ppl'] = round(results['tf']['ppl'] / max(results['prefix']['ppl'], 1e-9), 6)
    if 'prefix' in results and 'incremental' in results:
        diff['prefix_minus_incremental_loss'] = round(
            results['prefix']['loss'] - results['incremental']['loss'], 6)
    if 'tf' in results and 'incremental' in results:
        diff['tf_minus_incremental_loss'] = round(
            results['tf']['loss'] - results['incremental']['loss'], 6)

    gen = []
    if not args.no_gen:
        print('[4/4] 生成基准')
        gen = run_generation(model, vocab, device, DEFAULT_PROMPTS, args.seed)

    n_params = sum(p.numel() for p in model.parameters())
    payload = {
        'meta': {
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': args.config,
            'model': args.model,
            'model_sha256_16': _sha256_head(args.model),
            'vocab': args.vocab,
            'device': str(device),
            'torch': torch.__version__,
            'seed': args.seed,
            'batch_size': bs,
            'max_seqs': args.max_seqs,
            'val_sequences': len(val_dataset),
            'modes': modes,
            'controller': args.controller,
            'controller_direction': dir_on,
            'char_merge_incremental_buffer': cmb_on,
            'n_params': n_params,
            'pad_idx': int(ignore_index),
            'total_seconds': None,
        },
        'ppl': results,
        'diff': diff,
        'generation': gen,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload['meta']['total_seconds'] = round(time.time() - t0, 2)
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'\n基准已写入 {out_path}（{payload["meta"]["total_seconds"]}s）')


if __name__ == '__main__':
    main()
