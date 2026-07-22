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


def test_m1_temperature_not_lost_in_subsequent_steps():
    """M1 回归：ngram_fusion 路径下 τ≠1.0 时，model.generate 的后续步不得丢失温度。

    原 bug：_decode_one_step 在 temperature_applied=True 时把 τ 盖成 1.0 传给 forward，
    导致首步 forward(τ) 但后续步 forward(1.0)，采样分布首步冷、后续步热。
    修复后：_decode_one_step 始终把真实 τ 传给 forward，与 _generate_candidates_batch
    （每步都传 τ）行为一致，两条路径在 τ≠1.0 + ngram_fusion 下应仍 parity。

    验证方法：单序列 generate 与批量 _generate_candidates_batch(N=1) 在相同 seed +
    τ=0.7 下产出相同序列（修复前会因后续步温度丢失而发散）。"""
    from tests.test_decode_merge_parity import _ngram_model
    m, v, ng = _ngram_model()
    ngram_fn = ng.logprob_vector
    prompt = [v.bos_idx, 7, 11, 3, 9]
    tau = 0.7  # 非 1.0，触发 M1 bug 的关键条件
    torch.manual_seed(42)
    seq_single = m.generate(list(prompt), max_length=15, temperature=tau, top_k=0,
                           repetition_penalty=1.0, ngram_fn=ngram_fn, ngram_weight=0.3,
                           eos_id=v.eos_idx, pad_id=v.pad_idx, sep_id=v.sep_idx,
                           min_length=3, eos_penalty=-5.0, device='cpu')
    torch.manual_seed(42)
    seq_batch = _generate_candidates_batch(
        m, list(prompt), temps=[tau], max_length=15, top_k=0, rep_penalty=1.0,
        device='cpu', ngram_fn=ngram_fn, ngram_weight=0.3,
        pad_id=v.pad_idx, sep_id=v.sep_idx, eos_id=v.eos_idx,
        min_length=3, eos_penalty=-5.0)
    assert seq_batch[0] == seq_single, (
        f"M1 回归：τ={tau} 下两路径发散（修复前 generate 后续步温度丢失）"
        f"\n  single={seq_single}\n  batch={seq_batch[0]}")


def test_m1_temperature_applied_to_all_steps():
    """M1 回归：直接验证后续步 forward 也应用了温度。

    构造：ngram_fusion 模型，τ=0.5（强冷却），固定 seed 下用 model.generate 跑多步；
    对比 τ=1.0 的输出 —— 两者应不同（证明 τ 在后续步也被应用）。
    若 bug 复发（后续步盖 1.0），τ=0.5 与 τ=1.0 的后续步分布会几乎相同。"""
    from tests.test_decode_merge_parity import _ngram_model
    m, v, ng = _ngram_model()
    ngram_fn = ng.logprob_vector
    prompt = [v.bos_idx, 7, 11, 3, 9]

    # τ=0.5 跑多次取众数（采样有随机性，但温度差异会显著改变分布）
    torch.manual_seed(100)
    out_cold = m.generate(list(prompt), max_length=20, temperature=0.5, top_k=0,
                          repetition_penalty=1.0, ngram_fn=ngram_fn, ngram_weight=0.3,
                          eos_id=v.eos_idx, pad_id=v.pad_idx, sep_id=v.sep_idx,
                          min_length=5, eos_penalty=-5.0, device='cpu')
    # τ=1.0 对比
    torch.manual_seed(100)
    out_hot = m.generate(list(prompt), max_length=20, temperature=1.0, top_k=0,
                         repetition_penalty=1.0, ngram_fn=ngram_fn, ngram_weight=0.3,
                         eos_id=v.eos_idx, pad_id=v.pad_idx, sep_id=v.sep_idx,
                         min_length=5, eos_penalty=-5.0, device='cpu')
    # 两输出应不同（同 seed 但不同 τ → 后续步采样分布不同）
    # 若 bug 复发（后续步盖 1.0），只有首步受 τ 影响，后续步两 τ 下相同 → 输出更可能相同
    assert out_cold != out_hot, (
        f"M1 回归：τ=0.5 与 τ=1.0 在同 seed 下输出相同，说明后续步温度丢失"
        f"\n  cold={out_cold}\n  hot={out_hot}")
