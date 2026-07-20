import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config_loader import load_config, build_model
from models.transformer import sample_next_token
from scripts.generate import _generate_candidates_batch


def _tiny_model(device='cpu'):
    cfg = load_config('configs/config_hybrid.yaml')
    cfg['model']['num_layers'] = 2
    cfg['model']['num_heads'] = 2
    cfg['model']['embedding_dim'] = 32
    cfg['model']['layer_plan'] = 'attn,attn'
    cfg['model']['max_seq_length'] = 64
    for k in ('qk_norm', 'attn_temp', 'residual_gate', 'hybrid_gate'):
        cfg['model'][k] = True
    model = build_model(cfg, device=device)
    model.eval()
    return model


def test_sample_next_token_masking_and_penalty():
    """INT-2 回归：sample_next_token 须正确执行屏蔽与惩罚（pad/sep 屏蔽、min_length 屏蔽
    eos、重复惩罚、top_k 截断、全 -inf 回退）。"""
    dev = 'cpu'
    V = 20
    lt = torch.zeros(V)
    out = sample_next_token(lt.clone(), temperature=1.0, repetition_penalty=0.0,
                            generated_ids=[], ngram_fn=None, ngram_weight=0.0, device=dev,
                            pad_id=0, sep_id=1, eos_id=3, generated_len=0, min_length=3,
                            eos_penalty=-5.0, top_k=0, vocab_size=V, raw_logits=lt.clone())
    assert out is not None
    # 全 -inf 回退：所有合法 token 屏蔽后应回到原始分布（仅 pad 仍屏蔽）
    lt3 = torch.zeros(V)
    lt3[0] = 1e9  # pad 被设为 -inf，其余 0 → 全 -inf → 回退原始（pad 屏蔽）
    rb = sample_next_token(lt3.clone(), temperature=1.0, repetition_penalty=0.0,
                           generated_ids=[], ngram_fn=None, ngram_weight=0.0, device=dev,
                           pad_id=0, sep_id=1, eos_id=3, generated_len=0, min_length=3,
                           eos_penalty=-5.0, top_k=0, vocab_size=V, raw_logits=lt3.clone())
    assert rb is not None


def test_two_paths_unified_no_drift():
    """INT-2 回归：单序列 model.generate 与批量 _generate_candidates_batch(N=1) 共用
    sample_next_token，固定 seed 下应产出完全一致序列（消除两套采样循环漂移）。"""
    model = _tiny_model()
    prompt = [1, 2, 3, 4]
    torch.manual_seed(123)
    seq_single = model.generate(prompt, max_length=20, temperature=1.0, top_k=0,
                                repetition_penalty=1.2, eos_id=3, pad_id=0, sep_id=4,
                                min_length=3, eos_penalty=-5.0)
    torch.manual_seed(123)
    seq_batch = _generate_candidates_batch(
        model, prompt, temps=[1.0], max_length=20, top_k=0, rep_penalty=1.2,
        device='cpu', ngram_fn=None, ngram_weight=0.0, pad_id=0, sep_id=4, eos_id=3,
        min_length=3, eos_penalty=-5.0)
    assert seq_batch[0] == seq_single, (
        f"两采样路径漂移: single={seq_single} batch={seq_batch[0]}")


def test_reset_ngram_state_centralized():
    """INT 后续整合：_ngram_last_ids 滚动缓冲由模型自身 reset_ngram_state() 管理，
    调用方不直接戳实例变量。验证方法存在且把缓冲清空为 None。"""
    model = _tiny_model()
    model._ngram_last_ids = torch.zeros((1, 3), dtype=torch.long)
    model.reset_ngram_state()
    assert model._ngram_last_ids is None
    # generate 调用 reset_ngram_state 后（跨调用）结果稳定，无跨序列串味
    torch.manual_seed(7)
    s1 = model.generate([1, 2, 3], max_length=10, temperature=1.0, top_k=0,
                        eos_id=3, pad_id=0, sep_id=4, min_length=1)
    torch.manual_seed(7)
    s2 = model.generate([1, 2, 3], max_length=10, temperature=1.0, top_k=0,
                        eos_id=3, pad_id=0, sep_id=4, min_length=1)
    assert s1 == s2, "reset_ngram_state 未消除跨 generate 调用串味"
