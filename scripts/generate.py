import torch
import torch.nn.functional as F
import json
import argparse
import os
from pathlib import Path
import sys
from typing import Dict
from collections import defaultdict, Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.transformer import TransformerModel, apply_repetition_penalty, sample_next_token, _decode_one_step
from models.data_utils import Vocabulary
from models.config_loader import load_vocab
from models.device import get_device
from models.checkpoint import load_model, safe_torch_load
from models.ngram import NGramModel
from models.utils import cli_guard


def generate_text(model, vocab, prompt, max_length=30, temperature=0.8,
                   top_k=50, device='cpu', ngram=None, ngram_weight=0.0,
                   min_length=3, eos_penalty=-5.0, repetition_penalty=1.4):
    """Generate text from prompt（可选融合 n-gram 统计先验做解码期双轨）"""
    tokens = vocab.encode(prompt)
    if tokens[-1] == vocab.eos_idx:
        tokens = tokens[:-1]
    generated = model.generate(tokens, max_length=max_length,
                              temperature=temperature, top_k=top_k,
                              device=device, repetition_penalty=repetition_penalty,
                              ngram_fn=(ngram.logprob_vector if ngram else None),
                              ngram_weight=ngram_weight,
                              min_length=min_length,
                              eos_penalty=eos_penalty)
    text = vocab.decode(generated)
    return text.strip()


def interactive_mode(model, vocab, device='cpu', ngram=None, ngram_weight=0.0,
                      igmcg=False, intuition=None, candidates=5):
    """Interactive text generation（可选 n-gram / IGMCG 联合解码）"""
    print("\n" + "="*50)
    print("Text Generation with AI Model")
    print("="*50)
    print("Type prompts to generate text (type 'quit' to exit)")
    print("-"*50 + "\n")
    
    while True:
        prompt = input("Enter prompt: ").strip()
        
        if prompt.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not prompt:
            print("Prompt cannot be empty!")
            continue
        
        # Generate with different parameters
        print("\nGenerating...")
        temp_values = [0.7, 0.9]
        
        for temp in temp_values:
            if igmcg:
                generated, cands = generate_igmcg(
                    model, vocab, prompt, max_length=20, base_temp=temp,
                    top_k=50, device=device, num_candidates=candidates,
                    intuition=intuition, ngram_fn=(ngram.logprob_vector if ngram else None),
                    ngram_weight=ngram_weight,
                    min_length=3,
                    eos_penalty=-5.0)
                score = cands[0]['score'] if cands else 0.0
                print(f"[IGMCG T={temp}]: {generated}  (score={score:.3f})\n")
            else:
                generated = generate_text(model, vocab, prompt, max_length=20, 
                                          temperature=temp, top_k=50, device=device,
                                          ngram=ngram, ngram_weight=ngram_weight,
                                          min_length=3,
                                          eos_penalty=-5.0)
                print(f"[Temperature {temp}]: {generated}\n")
        
        print("-"*50)


def batch_generate(model, vocab, prompts, max_length=30, temperature=0.8, 
                    device='cpu', ngram=None, ngram_weight=0.0):
    """Generate text for multiple prompts"""
    results = []
    for prompt in prompts:
        generated = generate_text(model, vocab, prompt, max_length=max_length, 
                                 temperature=temperature, device=device,
                                 ngram=ngram, ngram_weight=ngram_weight)
        results.append({
            'prompt': prompt,
            'generated': generated
        })
    return results


# ---------------------------------------------------------------------------
# IGMCG：直觉引导多候选生成（推理期可控生成）
# 7 维直觉向量（每项 0~1，0.5=中性）：
#   0 语气 formal↔casual  1 氛围 relaxed↔tense  2 意图 explain↔persuade
#   3 情感 positive↔negative  4 风格 concise↔detailed  5 受众 pro↔general
#   6 创新 conservative↔aggressive
# 流程：按直觉生成多个风格候选 -> 自评(流畅度/重复度/风格匹配) -> 加权选优
# ---------------------------------------------------------------------------
IGMCG_DIMS = ['语气', '氛围', '意图', '情感', '风格', '受众', '创新']
_POS_WORDS = set('好 棒 喜欢 开心 高兴 优秀 美丽 顺利 成功 支持 赞美 爱 赞 不错 幸福'.split())
_NEG_WORDS = set('坏 差 讨厌 生气 难过 糟糕 失败 反对 批评 恨 悲 痛苦 问题 困难 错'.split())


def _repetition(text):
    """重复度惩罚（IGMCG 2.0 改为 distinct-2 多样性度量，比单纯 bigram 重复比更鲁棒）：
    返回 0~1，越高越多样（越低越重复）。用于评分时取负。"""
    toks = [t for t in text.replace(' ', '')]
    if len(toks) < 2:
        return 1.0
    bg = [tuple(toks[i:i+2]) for i in range(len(toks) - 1)]
    # distinct-2 = 不重复 bigram 占比（越高越不重复）
    return len(set(bg)) / max(1, len(bg))


def _zscore(values):
    """对候选列表做跨候选 z-score 标准化（均值0方差1），使不同量纲特征可比、抗尺度。
    单候选或零方差时返回全 0（退化为不贡献）。用 torch 向量化替代原 statistics Python 循环，提速。"""
    if len(values) <= 1:
        return [0.0] * len(values)
    t = torch.tensor(values, dtype=torch.float32)
    mu = t.mean()
    sd = t.std(unbiased=False)
    if sd < 1e-9:
        return [0.0] * len(values)
    return ((t - mu) / sd).tolist()


def _style_features(text):
    """把生成文本映射成 7 维风格特征信号（每项约 -1~1）。"""
    excl = text.count('！') + text.count('!')
    quest = text.count('？') + text.count('?')
    has_conn = any(w in text for w in ('因为', '所以', '例如', '即', '也就是说'))
    pos = sum(w in text for w in _POS_WORDS)
    neg = sum(w in text for w in _NEG_WORDS)
    toks = [t for t in text.replace(' ', '')]
    uniq_ratio = (len(set(toks)) / max(1, len(toks)))
    length = len(toks)
    concise = 1.0 - min(length, 60) / 60.0
    formal = 1.0 if excl == 0 else -1.0
    relaxed = 1.0 if (quest == 0 and excl == 0) else -0.5
    explain = 1.0 if has_conn else -0.3
    sentiment = (pos - neg) / max(1, (pos + neg))
    detailed = 1.0 if length >= 40 else -0.3
    professional = 1.0 if uniq_ratio > 0.7 else -0.3
    novelty = uniq_ratio * 2 - 1.0
    return [formal, relaxed, explain, sentiment, concise, professional, novelty]


def _generate_candidates_batch(model, ids, temps, max_length, top_k, rep_penalty,
                                device, ngram_fn, ngram_weight, pad_id, sep_id, eos_id,
                                min_length=3, eos_penalty=-5.0,
                                intuition=None):
    """并行生成 N 个候选：单次 batch 前向（batch=N），每个候选在 batch 内各自独立维护
    KV-cache / SSM 状态，避免候选间污染，也避免逐候选串行 forward（num_candidates 倍提速）。
    已完成候选喂 pad 占位以保持 batch 对齐。"""
    N = len(temps)
    generated = [list(ids) for _ in range(N)]
    done = [False] * N
    # IGMCG 2.0：直觉向量透传到 model.forward，广播到 batch 维 (N,7)
    intu_batch = None
    if intuition is not None:
        import torch as _t
        iv = _t.tensor(intuition, dtype=_t.float32, device=device).reshape(1, 7)
        intu_batch = iv.expand(N, -1)  # (N, 7) 广播到所有候选

    with torch.no_grad():
        # 重置 n-gram 滚动缓冲，避免跨调用残留上一序列的上下文污染当前生成
        model.reset_ngram_state()
        # 初始前向：所有候选共享同一输入，得到 batched past（batch 维 = N）
        inp = torch.tensor([ids] * N, dtype=torch.long, device=device)
        logits, past = model.forward(inp, past_key_values=None, use_cache=True,
                                     intuition=intu_batch)

        for step in range(max_length):
            if all(done):
                break
            if len(generated[0]) >= model.max_seq_length:
                break
            nt = []
            for n in range(N):
                if done[n]:
                    nt.append(pad_id)
                    continue
                # INT-2：单步采样统一走 sample_next_token（与 model.generate 共用单一事实来源）
                tok = sample_next_token(
                    logits[n, -1, :], temperature=temps[n], repetition_penalty=rep_penalty,
                    generated_ids=generated[n], ngram_fn=ngram_fn, ngram_weight=ngram_weight,
                    device=device, pad_id=pad_id, sep_id=sep_id, eos_id=eos_id,
                    generated_len=len(generated[n]) - len(ids), min_length=min_length,
                    eos_penalty=eos_penalty, top_k=top_k, vocab_size=logits[n, -1, :].shape[0],
                    raw_logits=logits[n, -1, :],
                )
                if tok is None:
                    # 低置信提前终止：填 pad 占位保持 batch 对齐并标记完成
                    nt.append(pad_id)
                    done[n] = True
                else:
                    nt.append(tok)
            # 单次 batched 前向：feed 形状 (N,1)，past 为 batched（batch 维 = N）
            # INT-整合：复用 _decode_one_step 的续前向语义，但批量版需保持 batch 维
            # 对齐（所有候选共享一次 batched forward，已完成的喂 pad），故这里直接走
            # batched forward 而非逐序列调用 _decode_one_step（逐序列会破坏 batch 对齐
            # 且无法共享单次前向）。_decode_one_step 的语义等价于单次 [[tok]] forward。
            feed = torch.tensor(nt, dtype=torch.long, device=device).unsqueeze(1)
            logits, past = model.forward(feed, past_key_values=past, use_cache=True,
                                         intuition=intu_batch)
            for n in range(N):
                if not done[n]:
                    generated[n].append(nt[n])
                    if nt[n] == eos_id and len(generated[n]) - len(ids) >= min_length:
                        done[n] = True

    return generated


def _fluency_batch(model, seqs, device, pad_id):
    """批量计算每条候选序列的平均 token 对数概率（流畅度）。"""
    N = len(seqs)
    if N == 0:
        return []
    maxlen = max(len(s) for s in seqs)
    batch = torch.full((N, maxlen), pad_id, dtype=torch.long, device=device)
    for n, s in enumerate(seqs):
        if s:
            batch[n, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    out = []
    with torch.no_grad():
        logits = model.forward(batch)
        for n, s in enumerate(seqs):
            if len(s) < 2:
                out.append(0.0)
                continue
            lp = F.log_softmax(logits[n, :len(s) - 1].float(), dim=-1)
            tgt = torch.tensor(s[1:], dtype=torch.long, device=device).unsqueeze(1)
            out.append(lp.gather(1, tgt).mean().item())
    return out


def _ngram_coherence(ngram_fn, ids, device):
    """平均 n-gram 模型对序列的预测 log-prob：越高=相邻 token 越相连(越连贯)，
    越低=越碎片化。无 ngram_fn 时返回 0（不参与评分）。这是抑制"碎片化"的关键信号。"""
    if ngram_fn is None or len(ids) < 2:
        return 0.0
    tot, n = 0.0, 0
    for i in range(1, len(ids)):
        lp = ngram_fn(ids[:i], device)
        tot += lp[ids[i]].item()
        n += 1
    return tot / max(1, n)


def generate_igmcg(model, vocab, prompt, intuition=None, num_candidates=4,
                    max_length=60, device='cpu', base_temp=0.7, top_k=30,
                    ngram_fn=None, ngram_weight=0.0, repetition_penalty=1.4,
                    min_length=3, eos_penalty=-5.0,
                    coh_w=1.5, flu_w=0.15, style_w=0.15, rep_w=2.5):
    """IGMCG 多候选生成：返回 (最优文本, 各候选评分详情)。

     intuition: 长度 7 的 float 列表（默认全 0.5 中性）。
     ngram_fn / ngram_weight: 可选叠加 n-gram 统计先验（双轨解码）。
     所有候选并行生成（单次 batch 前向），显著降低多候选开销。

     评分 = COH_W*连贯度(coh) + FLU_W*流畅度 + STYLE_W*风格匹配 - REP_W*重复度
       - coh 直接奖励相邻 token 的相连性，是抑制"碎片化"的核心信号；
       - 流畅度(单 token 置信度)只作轻微 tiebreaker——孤立高频词也会拉高它，故不主导；
       - 风格匹配为直觉引导的温和偏置（在连贯候选之间做微调，绝不压过连贯度）；
       - 候选温度范围收窄(0.75~1.35x)、重复惩罚下调，避免候选本身过度发散成碎片。"""
    if intuition is None:
        intuition = [0.5] * 7
    assert len(intuition) == 7, "直觉向量必须是 7 维"

    ids = [vocab.bos_idx] + vocab.encode(prompt, add_special_tokens=False)
    pad_id = vocab.pad_idx
    sep_id = getattr(vocab, 'sep_idx', 4)
    eos_id = getattr(vocab, 'eos_idx', 3)

    temps = [base_temp * (0.75 + 0.6 * k / max(1, num_candidates - 1))
             for k in range(num_candidates)]
    seqs = _generate_candidates_batch(model, ids, temps, max_length, top_k,
                                       repetition_penalty, device, ngram_fn,
                                       ngram_weight, pad_id, sep_id, eos_id,
                                       min_length=min_length, eos_penalty=eos_penalty,
                                       intuition=intuition)
    flus = _fluency_batch(model, seqs, device, pad_id)

    COH_W, FLU_W, STYLE_W, REP_W = coh_w, flu_w, style_w, rep_w
    candidates = []
    for k, gen in enumerate(seqs):
        cand_ids = gen[len(ids):]
        text = vocab.decode(cand_ids, skip_special=True).strip()
        if not text:
            continue
        rep = _repetition(text)                       # distinct-2 多样性（越高越不重复）
        feat = _style_features(text)                  # 7 维风格特征信号
        # 连贯度：相邻 token 在 n-gram 模型下的平均预测概率（越高越相连）
        coh = _ngram_coherence(ngram_fn, gen, device)
        candidates.append({'text': text, 'flu': flus[k], 'coh': coh,
                           'rep': rep, 'feat': feat, 'temp': round(temps[k], 2)})
    if not candidates:
        return '', []

    # IGMCG 2.0 更聪明打分：跨候选 z-score 标准化各特征（抗尺度），直觉作为"方向"而非
    # 固定逐维符号权重——用候选特征向量与直觉方向的余弦相似度衡量"贴合直觉程度"，
    # 避免手写权重在不同语料/特征分布下错位；再加小幅度多样性奖励避免候选塌缩。
    coh_z = _zscore([c['coh'] for c in candidates])
    flu_z = _zscore([c['flu'] for c in candidates])
    rep_z = _zscore([c['rep'] for c in candidates])
    # 直觉方向（7 维，0~1→-1~1）；与候选风格特征做余弦相似度（贴合直觉=高）
    import math as _m
    wvec = [(v - 0.5) * 2.0 for v in intuition]
    wn = _m.sqrt(sum(x * x for x in wvec)) + 1e-9
    style_scores = []
    for c in candidates:
        f = c['feat']
        num = sum(a * b for a, b in zip(wvec, f))
        den = _m.sqrt(sum(x * x for x in f)) + 1e-9
        style_scores.append(num / (wn * den))        # cosine 相似度 ∈[-1,1]
    style_z = _zscore(style_scores)

    for k, c in enumerate(candidates):
        # 连贯度主导（z 标准化，绝对值可比）；流畅/风格为温和偏置；重复取负（重复越低分越高）。
        score = COH_W * coh_z[k] + FLU_W * flu_z[k] + STYLE_W * style_z[k] - REP_W * rep_z[k]
        c['style'] = style_scores[k]
        c['score'] = score
    candidates.sort(key=lambda c: c['score'], reverse=True)
    return candidates[0]['text'], candidates


@cli_guard
def main():
    # 避免中文在 GBK 控制台打印时崩溃；同时把结果写入 UTF-8 文件便于查看
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    parser = argparse.ArgumentParser(description='Text generation with trained model')
    parser.add_argument('--model', type=str, default='./checkpoints/final_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, default='./checkpoints/vocab.json',
                        help='Path to vocabulary file')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Text prompt for generation')
    parser.add_argument('--prompt-file', type=str, default=None,
                        help='Path to a UTF-8 file containing the prompt (avoids console GBK encoding issues with Chinese)')
    parser.add_argument('--max-length', type=int, default=30,
                        help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (0.5-1.5)')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k sampling')
    parser.add_argument('--repetition-penalty', type=float, default=1.4,
                        help='重复惩罚值（>1 抑制重复，1.0=关闭）')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: auto (detect) / cuda / cpu / dml')
    parser.add_argument('--cpu-threads', type=int, default=4,
                        help='CPU 生成时使用的线程数（降功耗；单流解码速度影响很小）')
    parser.add_argument('--quantize', action='store_true',
                        help='推理时启用 int8 动态量化（仅 CPU，降内存带宽/功耗，几乎无质量损失）')
    parser.add_argument('--compile', action='store_true',
                        help='推理时对模型做 torch.compile（需本机有 C++ 编译器；无则自动回退 eager）')
    parser.add_argument('--dtype', choices=['fp32', 'bf16', 'auto'], default='auto',
                        help='推理精度：auto=支持的 CPU/CUDA 用 bf16（约 1.5~1.8x 提速且质量基本无损），否则 fp32')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--ngram', action='store_true',
                        help='解码期融合 Bigram/Trigram 统计先验（神经+统计双轨）')
    parser.add_argument('--ngram-corpus', default='data/pretrain_corpus/merged.txt',
                        help='构建 n-gram 统计所用的语料文件')
    parser.add_argument('--ngram-weight', type=float, default=0.3,
                        help='n-gram 先验叠加权重（0=关闭）')
    parser.add_argument('--igmcg', action='store_true',
                        help='启用 IGMCG 直觉引导解码；与 --ngram 同时开启即为 n-gram+IGMCG 联合解码')
    parser.add_argument('--igmcg-candidates', type=int, default=5,
                        help='IGMCG 候选数')
    parser.add_argument('--intuition', type=str, default='0.5,0.5,0.5,0.5,0.5,0.5,0.5',
                        help='IGMCG 7 维直觉向量(逗号分隔, 0~1)')
    parser.add_argument('--igmcg-coh-w', type=float, default=1.5,
                        help='IGMCG 连贯度权重')
    parser.add_argument('--igmcg-flu-w', type=float, default=0.15,
                        help='IGMCG 流畅度权重')
    parser.add_argument('--igmcg-style-w', type=float, default=0.15,
                        help='IGMCG 风格匹配权重')
    parser.add_argument('--igmcg-rep-w', type=float, default=2.5,
                        help='IGMCG 重复惩罚权重')
    
    args = parser.parse_args()

    # CPU 生成时限制线程数以降功耗（单流自回归解码对线程数不敏感，省电明显）
    if args.cpu_threads and args.cpu_threads > 0:
        torch.set_num_threads(max(1, args.cpu_threads))
        torch.set_num_interop_threads(max(1, args.cpu_threads // 2))

    # Check if model exists
    if not Path(args.model).exists():
        print(f"Model not found at {args.model}")
        print("Please train the model first using: python scripts/train.py")
        return

    # Load model (自动适配 CUDA / DirectML(AMD) / CPU；--device 默认 auto 自动探测)
    device = get_device(args.device)
    print(f"Loading model from {args.model}...")
    model, vocab = load_model(args.model, args.vocab, device=device, quantize=args.quantize, compile_model=args.compile)
    print(f"Model loaded successfully!")
    print(f"Vocabulary size: {len(vocab)}")

    # 推理精度：bf16 在支持的 CPU/CUDA 上约 1.5~1.8x 提速，且质量基本无损（此机实测困惑度更优）
    # 注意：torch 2.4.1 无 torch.cpu.get_cpu_capability()（该 API 在更新版本才存在），
    # 旧代码调用它会抛异常被吞掉、实际回退 fp32 但打印"bf16"——属假 bf16 打印 bug。
    # 修正：CPU bf16 自 torch 2.0 起由 oneDNN 稳定支持，CUDA 用官方 is_bf16_supported() 探测，
    # 探测失败才回退 fp32，且打印与实际行为严格一致。
    dtype = args.dtype
    if dtype == 'auto':
        dtype = 'bf16' if device.type in ('cpu', 'cuda') else 'fp32'
        if device.type == 'cuda':
            if not torch.cuda.is_bf16_supported():
                dtype = 'fp32'
        # CPU 下 torch>=2.0 默认支持 bf16 自动混合精度，无需额外探测
    if dtype == 'bf16' and device.type in ('cpu', 'cuda'):
        # 在对应后端启用 bf16 自动混合精度（注意必须按实际 device.type 开启，
        # 原实现只开了 'cpu' autocast，导致 CUDA 下 bf16 实际未生效）
        torch.set_autocast_enabled(device.type, True)
        torch.set_autocast_dtype(device.type, torch.bfloat16)
        print("推理精度: bf16（%s autocast，约 1.5~1.8x 提速）" % ("CPU" if device.type == 'cpu' else "CUDA"))
    else:
        print("推理精度: fp32")
    
    # 构建 n-gram 统计模型（解码期双轨）
    ngram = None
    if args.ngram:
        print(f"Building n-gram model from {args.ngram_corpus} ...")
        ngram = NGramModel(vocab, args.ngram_corpus, max_order=10, smoothing=1.0)
        print(f"n-gram ready (weight={args.ngram_weight})")
    
    # Generation mode
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, 'r', encoding='utf-8-sig') as pf:
            prompt = pf.read().strip()
    
# 联合解码：IGMCG 候选生成（若同时 --ngram，则每个候选都叠加 n-gram 先验）
    use_igmcg = args.igmcg
    intuition = [float(x) for x in args.intuition.split(',')]
    assert len(intuition) == 7, "直觉向量需 7 维"

    if args.interactive:
        interactive_mode(model, vocab, device, ngram=ngram, ngram_weight=args.ngram_weight,
                         igmcg=use_igmcg, intuition=intuition, candidates=args.igmcg_candidates)
    elif prompt:
        if use_igmcg:
            generated, cands = generate_igmcg(
                model, vocab, prompt, max_length=args.max_length,
                base_temp=args.temperature, top_k=args.top_k, device=device,
                num_candidates=args.igmcg_candidates, intuition=intuition,
                ngram_fn=(ngram.logprob_vector if ngram else None),
                ngram_weight=args.ngram_weight,
                min_length=3,
                eos_penalty=-5.0,
                coh_w=args.igmcg_coh_w,
                flu_w=args.igmcg_flu_w,
                style_w=args.igmcg_style_w,
                rep_w=args.igmcg_rep_w)
            best = cands[0] if cands else None
            gen_info = (f"\n  [IGMCG候选={len(cands)}, 最优分={best['score']:.3f}, 重复度={best['rep']:.3f}]"
                        if best else "")
        else:
            generated = generate_text(model, vocab, prompt,
                                       max_length=args.max_length,
                                       temperature=args.temperature,
                                       top_k=args.top_k,
                                       device=device,
                                       ngram=ngram, ngram_weight=args.ngram_weight,
                                       min_length=3,
                                       eos_penalty=-5.0,
                                       repetition_penalty=args.repetition_penalty)
            gen_info = ""
        # 同时写 UTF-8 结果文件，方便中文查看（控制台可能是 GBK）
        out_path = os.path.join('logs', 'generation_output.txt')
        os.makedirs('logs', exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as of:
            of.write(f"Prompt: {prompt}\nGenerated: {generated}\n")
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}{gen_info}\n")
        print(f"(结果已同时写入 {out_path})")
    else:
        # Default example
        examples = [
            "Hello, how are you",
            "The weather is",
            "I love",
            "Machine learning"
        ]
        print("\nGenerating text for example prompts:\n")
        for prompt in examples:
            generated = generate_text(model, vocab, prompt, 
                                      max_length=20,
                                      temperature=0.8,
                                      top_k=50,
                                      device=device,
                                      ngram=ngram, ngram_weight=args.ngram_weight,
                                      min_length=3,
                                      eos_penalty=-5.0)
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated}\n")


if __name__ == '__main__':
    main()
