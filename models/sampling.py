from __future__ import annotations
from typing import Optional, List, Callable
import torch


def apply_repetition_penalty(logits: torch.Tensor, generated_ids: List[int],
                             penalty: float, device: torch.device) -> torch.Tensor:
    """加性频率重复惩罚（INT-1 去重点：sample_step 与 _generate_candidates_batch 共用）。

    对已出现 token 按出现次数减去 penalty*count（对称稳定、无乘性正负不对称问题），
    返回修正后的 logits（原地修改传入张量）。生成路径（model.generate）与 IGMCG 批量
    解码路径（_generate_candidates_batch）统一调用，避免两处惩罚公式漂移。
    """
    if penalty <= 0 or not generated_ids:
        return logits
    from collections import Counter
    freq = Counter(generated_ids)
    vocab_size = logits.shape[-1]
    prev_toks = torch.tensor(list(freq.keys()), dtype=torch.long, device=device)
    prev_counts = torch.tensor(list(freq.values()), dtype=torch.float, device=device)
    valid = (prev_toks >= 0) & (prev_toks < vocab_size)
    logits[prev_toks[valid]] -= penalty * prev_counts[valid]
    return logits


def sample_next_token(logits_t: torch.Tensor, *, temperature: float,
                      repetition_penalty: float, generated_ids: List[int],
                      ngram_fn: Optional[Callable[[List[int], str], torch.Tensor]],
                      ngram_weight: float, device: str,
                      pad_id: int, sep_id: int, eos_id: int,
                      generated_len: int, min_length: int, eos_penalty: float,
                      top_k: int, vocab_size: int,
                      raw_logits: Optional[torch.Tensor] = None) -> Optional[int]:
    """INT-2：单步采样单一事实来源——model.generate（单序列）与 generate.py 批量候选
    解码（_generate_candidates_batch）共用，消除两套采样循环公式漂移。

    流程：温度缩放 → 加性重复惩罚 → n-gram 先验叠加 → pad/sep 屏蔽 → min_length/eos 处理
    → top_k 截断 → 全 -inf 回退 → softmax → multinomial。低置信（probs.max()<0.01）返回
    None 表示提前终止。返回的是 token id（或 None）。"""
    lt = logits_t / temperature
    apply_repetition_penalty(lt, generated_ids, repetition_penalty, device)
    if ngram_fn is not None and ngram_weight != 0.0:
        lt = lt + ngram_weight * ngram_fn(generated_ids, device)
    lt[pad_id] = float('-inf')
    lt[sep_id] = float('-inf')
    if generated_len < min_length:
        lt[eos_id] = float('-inf')
    else:
        lt[eos_id] = lt[eos_id] + eos_penalty
    if top_k > 0 and top_k < vocab_size:
        top_k_vals = torch.topk(lt, min(top_k, vocab_size))[0]
        threshold = top_k_vals[..., -1]
        lt[lt < threshold] = float('-inf')
    if torch.isinf(lt).all():
        # 全 -inf 回退（设计行为，非 bug）：所有合法 token 都被屏蔽（pad/sep/eos/
        # top_k/惩罚后无候选）的极端边界，放弃已处理分布、用原始温度分布仅屏蔽 pad
        # 以产出合法 token 避免崩溃。raw_logits 为未被惩罚污染的前向原始 logits。
        rb = (raw_logits if raw_logits is not None else logits_t) / temperature
        rb[pad_id] = float('-inf')
        lt = rb
    probs = torch.softmax(lt, dim=-1)
    if probs.max() < 0.01:
        return None
    return torch.multinomial(probs, num_samples=1).item()
def _decode_one_step(model: "TransformerModel", next_token: int,
                     past: Optional[Any], cur_pos: int, *, device: str,
                     use_cache: bool = True) -> Tuple[Any, torch.Tensor, int]:
    """单步续前向（KV-cache 驱动）原语，供 model.generate 与 scripts/generate.py
    的批量解码共用，消除两套解码主循环重复的「input_ids=[[tok]] -> forward(past)
    -> cur_pos+=1」驱动逻辑。语义与原 generate 循环体完全一致（数值不变）。"""
    input_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
    logits, past = model.forward(input_ids, past_key_values=past, use_cache=use_cache)
    cur_pos += 1
    return past, logits, cur_pos
