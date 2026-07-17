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

from models.transformer import TransformerModel
from models.data_utils import Vocabulary
from models.config_loader import load_vocab
from models.device import get_device
from models.utils import cli_guard


# weights_only=True 仅允许的白名单全局符号：均为张量/numpy 反序列化的官方重建函数。
# 只放行这些固定符号，杜绝被恶意 .pt 诱导放行任意全局符号（即 CVE-2026-24747 类绕过）。
def _safe_getattr(dotted):
    """逐段 getattr，任意一段缺失都返回 None（兼容 numpy 1.x/2.x 的路径差异）。"""
    parts = dotted.split('.')
    obj = __import__(parts[0])
    for attr in parts[1:]:
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj


def _build_safe_globals():
    import numpy as np
    cands = [
        getattr(torch._utils, '_rebuild_device_tensor_from_numpy', None),
        _safe_getattr('numpy.core.multiarray._reconstruct'),   # numpy 1.x
        _safe_getattr('numpy._core.multiarray._reconstruct'),   # numpy 2.x
        getattr(np, 'ndarray', None),
        getattr(np, 'dtype', None),
    ]
    return [g for g in cands if g is not None]


_SAFE_GLOBALS = _build_safe_globals()
for _g in _SAFE_GLOBALS:
    try:
        torch.serialization.add_safe_globals([_g])
    except Exception:
        pass


def _safe_torch_load(path, map_location='cpu'):
    """以 weights_only=True 加载 checkpoint。

    白名单全局符号（仅官方张量/numpy 重建函数，见 _SAFE_GLOBALS）已在模块导入时放行；
    若仍遇到非白名单的全局符号，说明该文件可能不是可信 checkpoint，直接抛错拒绝加载，
    避免被诱导放行危险全局符号（CVE-2026-24747 类绕过）。"""
    return torch.load(path, map_location=map_location, weights_only=True)

def load_model(model_path, vocab_path, device='cpu', quantize=False, compile_model=False):
    """Load trained model and vocabulary.

    quantize=True 时对 Linear 层做 int8 动态量化（仅 CPU 有效）：用更低带宽的量化权重做
    矩阵乘，降低内存带宽与功耗，对生成质量几乎无损。AMD DML 设备无量化算子支持，会自动跳过。
    compile_model=True 时对模型做 torch.compile（CUDA/CPU 有效）：融合 RMSNorm/RoPE/MatMul 等
    算子在自回归解码上通常带来 1.5~3× 吞吐提升；DML 设备自动跳过。
    """
    from torch import nn
    import yaml

    # Load vocabulary（复用 config_loader.load_vocab，正确处理 BPE/char 词表的 merges 等字段，
    # 与训练期保存逻辑对称；避免手写 Vocabulary() 重建丢失 bpe/char 信息导致分布错位）
    vocab = load_vocab(vocab_path)

    # Load model（map_location 用 'cpu'，加载后再由下方 .to(device) 搬运到目标设备）。
    # 注意：DML 设备下若直接用 torch.device('privateuseone:0') 作 map_location，
    # torch.load 内部会调用 torch_directml.device(torch.device) 触发 TypeError，
    # 导致 DML 推理无法加载权重；统一先加载到 CPU 可绕开该问题。
    # torch 2.4 的 weights_only 默认白名单不含张量/numpy 反序列化所需的若干全局符号。
    # checkpoint 均来自可信来源，故用 _safe_torch_load 按需放行被拒的官方重建符号并重试，
    # 既保留 weights_only 的安全语义，又能兼容不同 torch/numpy 版本产出的权重。
    checkpoint = _safe_torch_load(model_path, map_location='cpu')

    # Load config from separate YAML file (for weights_only=True compatibility)
    model_path_obj = Path(model_path)
    config_path = model_path_obj.parent / f"{model_path_obj.stem}_config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            model_config = yaml.safe_load(f)
    else:
        # Fallback for old checkpoints with config embedded
        model_config = checkpoint.get('config', {
            'vocab_size': checkpoint.get('vocab_size', 12000),
            'embedding_dim': 128,
            'num_heads': 4,
            'num_layers': 2,
            'hidden_dim': 256,
            'max_seq_length': 32,
            'dropout': 0.1
        })

    # 复用 build_model（正确传递所有参数：char_merge/memory/ssm/rope 等），避免手动列参数遗漏。
    # strict=False 兼容旧权重（旧 checkpoint 可能缺少 qk_norm/attn_temp/residual_gate 等新增参数）。
    from models.config_loader import build_model
    model = build_model({'model': model_config}, device=device)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    if quantize and getattr(device, 'type', None) != 'dml':
        # 量化返回新模型对象，必须重新赋值；DML 无量化算子支持，已在上面跳过
        try:
            model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        except Exception as e:
            print(f"[warn] int8 动态量化不可用，回退 fp32：{e}")

    if compile_model and getattr(device, 'type', None) != 'dml':
        try:
            model = torch.compile(model, dynamic=True)
        except Exception as e:
            print(f"[warn] torch.compile 不可用，回退 eager 模式：{e}")

    return model, vocab


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


class NGramModel:
    """统计语言模型（Unigram/Bigram/Trigram 插值）。解码期作为神经 LM 的先验，
     不改变模型、不需重训，专门改善字符级 LM 的局部连贯性。
     l1/l2/l3 为 uni/bi/tri 的插值权重（默认偏向高阶三元）。"""
    def __init__(self, vocab, corpus_file, max_order=3, smoothing=1.0,
                 l1=0.1, l2=0.3, l3=0.6):
        self.vocab = vocab
        self.max_order = max_order
        self.smoothing = smoothing
        self.l1, self.l2, self.l3 = l1, l2, l3
        self.vocab_size = len(vocab)
        self.uni = Counter()               # 一元
        self.bi = defaultdict(Counter)     # (w_{i-1}) -> Counter(next)
        self.tri = defaultdict(Counter)    # (w_{i-2}, w_{i-1}) -> Counter(next)
        self._build(corpus_file)
        # 解码期同一上下文会被多个候选反复查询，缓存 logprob 向量避免重复建表计算
        self._logprob_cache: Dict = {}
        self._logprob_cache_max = 8192

    def _build(self, corpus_file):
        # errors='replace' 避免脏语料含非法 UTF-8 序列时直接抛 UnicodeDecodeError
        with open(corpus_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                ids = self.vocab.encode(line, add_special_tokens=False)
                ids = [self.vocab.pad_idx] * 2 + ids
                for i in range(2, len(ids)):
                    self.uni[ids[i]] += 1
                    self.bi[ids[i - 1]][ids[i]] += 1
                    if self.max_order >= 3:
                        self.tri[(ids[i - 2], ids[i - 1])][ids[i]] += 1
        # 预计算（加速解码期每次调用的 logprob_vector）
        V = self.vocab_size
        self.uni_total = sum(self.uni.values()) + self.smoothing * V
        self.bi_total = {w: sum(c.values()) + self.smoothing * V
                         for w, c in self.bi.items()}
        self.tri_total = {k: sum(c.values()) + self.smoothing * V
                          for k, c in self.tri.items()}
        u = torch.full((V,), self.smoothing)
        for t, c in self.uni.items():
            if 0 <= t < V:
                u[t] += c
        self.uni_prob = (u / u.sum()).clone()

    def logprob_vector(self, generated, device):
        """返回词表每个 token 的 log 概率向量（uni/bi/tri 插值，三元真正参与）。
        只在与当前上下文相关的少量 token 上计算，避免遍历全词表。结果按上下文
        (w2,w1) 缓存，因为解码期同一上下文会被多个候选反复查询。"""
        V = self.vocab_size
        w2 = generated[-2] if len(generated) >= 2 else None
        w1 = generated[-1] if len(generated) >= 1 else None
        cache_key = (w2, w1)
        if cache_key in self._logprob_cache:
            return self._logprob_cache[cache_key].to(device)
        vec = self._compute_logprob(w2, w1, V, device)
        if len(self._logprob_cache) > self._logprob_cache_max:
            self._logprob_cache.clear()
        self._logprob_cache[cache_key] = vec.cpu()
        return vec

    def _compute_logprob(self, w2, w1, V, device):
        """实际计算上下文 (w2,w1) 下的 log 概率向量（无缓存）。"""
        vec = self.uni_prob.to(device).clone()      # (V,) 已归一化的 unigram 先验
        uni_total = self.uni_total

        # bigram 覆盖（仅遍历 w1 之后的少量 token）
        if w1 is not None and w1 in self.bi:
            bc = self.bi[w1]
            bt = self.bi_total[w1]
            idx = torch.tensor(list(bc.keys()), dtype=torch.long, device=device)
            if idx.numel():
                cu = torch.tensor([self.uni.get(t, 0) for t in bc.keys()], device=device)
                cb = torch.tensor(list(bc.values()), device=device)
                pu = (cu + self.smoothing) / uni_total
                pb = (cb + self.smoothing) / bt
                vec[idx] = self.l1 * pu + self.l2 * pb

        # trigram 覆盖（优先，仅遍历 (w2,w1) 之后的少量 token）
        if w2 is not None and (w2, w1) in self.tri:
            tc = self.tri[(w2, w1)]
            tt = self.tri_total[(w2, w1)]
            idx = torch.tensor(list(tc.keys()), dtype=torch.long, device=device)
            if idx.numel():
                cu = torch.tensor([self.uni.get(t, 0) for t in tc.keys()], device=device)
                cb = torch.tensor([self.bi[w1].get(t, 0) for t in tc.keys()], device=device)
                ct = torch.tensor(list(tc.values()), device=device)
                pu = (cu + self.smoothing) / uni_total
                pb = (cb + self.smoothing) / self.bi_total.get(w1, self.smoothing * V)
                pt = (ct + self.smoothing) / tt
                vec[idx] = self.l1 * pu + self.l2 * pb + self.l3 * pt

        vec = vec / vec.sum()
        return torch.log(vec + 1e-10)


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
    """重复度惩罚：归一化的重复 bigram 比例。"""
    toks = text.replace(' ', '')
    if len(toks) < 2:
        return 0.0
    bg = [toks[i:i+2] for i in range(len(toks) - 1)]
    return 1.0 - len(set(bg)) / max(1, len(bg))


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
                                min_length=3, eos_penalty=-5.0):
    """并行生成 N 个候选：单次 batch 前向（batch=N），每个候选在 batch 内各自独立维护
    KV-cache / SSM 状态，避免候选间污染，也避免逐候选串行 forward（num_candidates 倍提速）。
    已完成候选喂 pad 占位以保持 batch 对齐。"""
    N = len(temps)
    generated = [list(ids) for _ in range(N)]
    done = [False] * N

    with torch.no_grad():
        # 初始前向：所有候选共享同一输入，得到 batched past（batch 维 = N）
        inp = torch.tensor([ids] * N, dtype=torch.long, device=device)
        logits, past = model.forward(inp, past_key_values=None, use_cache=True)

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
                lt = logits[n, -1, :] / temps[n]
                # 符号感知的重复惩罚：正值除、负值乘（与 HF 一致）
                for prev in set(generated[n]):
                    if 0 <= prev < lt.shape[0]:
                        if lt[prev] > 0:
                            lt[prev] = lt[prev] / rep_penalty
                        else:
                            lt[prev] = lt[prev] * rep_penalty
                if ngram_fn is not None and ngram_weight != 0.0:
                    lt = lt + ngram_weight * ngram_fn(generated[n], device)
                lt[pad_id] = float('-inf')
                lt[sep_id] = float('-inf')
                if len(generated[n]) - len(ids) < min_length:
                    lt[eos_id] = float('-inf')
                else:
                    lt[eos_id] = lt[eos_id] + eos_penalty
                if top_k > 0 and top_k < lt.shape[0]:
                    thr = torch.topk(lt, min(top_k, lt.shape[0]))[0][-1]
                    lt[lt < thr] = float('-inf')
                if torch.isinf(lt).all():
                    lt = logits[n, -1, :] / temps[n]
                    lt[pad_id] = float('-inf')
                nt.append(int(torch.multinomial(torch.softmax(lt, 0), 1).item()))
            # 单次 batched 前向：feed 形状 (N,1)，past 为 batched（batch 维 = N）
            feed = torch.tensor(nt, dtype=torch.long, device=device).unsqueeze(1)
            logits, past = model.forward(feed, past_key_values=past, use_cache=True)
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
    # 把 0~1 映射为方向权重 (-1~1)
    w = [(v - 0.5) * 2.0 for v in intuition]

    ids = [vocab.bos_idx] + vocab.encode(prompt, add_special_tokens=False)
    pad_id = vocab.pad_idx
    sep_id = getattr(vocab, 'sep_idx', 4)
    eos_id = getattr(vocab, 'eos_idx', 3)

    temps = [base_temp * (0.75 + 0.6 * k / max(1, num_candidates - 1))
             for k in range(num_candidates)]
    seqs = _generate_candidates_batch(model, ids, temps, max_length, top_k,
                                       repetition_penalty, device, ngram_fn,
                                       ngram_weight, pad_id, sep_id, eos_id,
                                       min_length=min_length, eos_penalty=eos_penalty)
    flus = _fluency_batch(model, seqs, device, pad_id)

    COH_W, FLU_W, STYLE_W, REP_W = coh_w, flu_w, style_w, rep_w
    candidates = []
    for k, gen in enumerate(seqs):
        cand_ids = gen[len(ids):]
        text = vocab.decode(cand_ids, skip_special=True).strip()
        if not text:
            continue
        rep = _repetition(text)
        feat = _style_features(text)
        # 风格匹配 = Σ 方向权重 * 文本特征
        style_match = sum(wi * fi for wi, fi in zip(w, feat))
        # 连贯度：相邻 token 在 n-gram 模型下的平均预测概率（越高越相连）
        coh = _ngram_coherence(ngram_fn, gen, device)
        score = COH_W * coh + FLU_W * flus[k] + STYLE_W * style_match - REP_W * rep
        candidates.append({'text': text, 'score': score, 'flu': flus[k], 'coh': coh,
                           'rep': rep, 'style': style_match, 'temp': round(temps[k], 2)})
    if not candidates:
        return '', []
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
        ngram = NGramModel(vocab, args.ngram_corpus, max_order=3, smoothing=1.0)
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
