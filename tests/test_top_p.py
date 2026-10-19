"""top-p（nucleus）截断：默认 1.0 与加入前逐位一致；阈值很小时只剩最高概率 token。"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sampling import sample_next_token


def _call(lt, top_p, seed=0, temperature=1.0):
    torch.manual_seed(seed)
    return sample_next_token(lt.clone(), temperature=temperature, repetition_penalty=0.0,
                             generated_ids=[], ngram_fn=None, ngram_weight=0.0, device='cpu',
                             pad_id=0, sep_id=1, eos_id=3, generated_len=5, min_length=3,
                             eos_penalty=0.0, top_k=0, vocab_size=lt.shape[0], raw_logits=lt.clone(),
                             top_p=top_p)


def test_default_is_noop():
    lt = torch.randn(50)
    for sd in range(20):
        assert _call(lt, 1.0, sd) == _call(lt, 0.0, sd)  # 1.0 与 0（关）都不截断


def test_tiny_top_p_keeps_only_argmax():
    lt = torch.randn(50)
    lt[0] = lt[1] = -1e4  # pad/sep 会被屏蔽，别让它们成为最大值
    best = int(torch.argmax(lt[2:])) + 2
    assert all(_call(lt, 1e-6, sd) == best for sd in range(20))


def test_nucleus_set():
    # 概率 0.5 / 0.3 / 0.15 / 0.05：top_p=0.8 应保留前两个（累计 0.5 < 0.8，0.8 处首次达到）
    p = torch.tensor([1e-9, 1e-9, 0.5, 0.3, 0.15, 0.05])
    lt = p.log()
    got = {_call(lt, 0.8, sd) for sd in range(200)}
    assert got == {2, 3}


def test_greedy_unaffected():
    lt = torch.randn(50)
    assert _call(lt, 0.3, temperature=0.0) == _call(lt, 1.0, temperature=0.0)
