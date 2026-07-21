import torch
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer import TransformerModel, _decode_one_step
from scripts.generate import _generate_candidates_batch

from tests.test_new_mechanisms import _small_ngram


def _ngram_model():
    v, ng = _small_ngram()
    # mixer 用普通 attn；启用 ngram_fusion 以覆盖 reset_ngram_state / ngram 融合路径
    m = TransformerModel(vocab_size=len(v), embedding_dim=64, num_heads=4,
                         num_layers=2, hidden_dim=128, max_seq_length=32,
                         ngram_fusion=True, ngram_model=ng)
    m.eval()
    return m, v, ng


def test_decode_one_step_matches_inline_forward():
    """_decode_one_step 必须与原 generate 循环体内的续前向等价（KV-cache 驱动、
    cur_pos 自增）。在等价 n-gram 滚动状态下（均从干净状态起第一步）比较 logits。"""
    m, v, ng = _ngram_model()
    device = 'cpu'
    tok = 5
    m.reset_ngram_state()  # 干净起点，对齐 ngram 滚动缓冲
    inp = torch.tensor([[tok]], dtype=torch.long, device=device)
    logits_ref, past_ref = m.forward(inp, past_key_values=None, use_cache=True)
    m.reset_ngram_state()  # _decode_one_step 内部不管理 ngram 状态，调用方自行对齐
    past2, logits_new, cur_pos2 = _decode_one_step(m, tok, None, 0, device=device)
    diff = (logits_ref - logits_new).abs().max().item()
    assert diff < 1e-6, f"续前向不一致：max_diff={diff}"
    assert cur_pos2 == 1


def test_decode_one_step_temperature_applied():
    """回归：_decode_one_step 在 temperature_applied=True 时必须跳过 forward 内部温度缩放
    （传 temperature=1.0），否则与 sample_next_token(temperature_applied=True) 组合会导致
    双重除温（非 IGMCG 路径 ngram_fusion 关闭时的回归点）。"""
    m, v, ng = _ngram_model()
    device = 'cpu'
    tok = 5
    temp = 0.8
    # temperature_applied=True → _decode_one_step 应传 temperature=1.0 给 forward
    m.reset_ngram_state()
    inp = torch.tensor([[tok]], dtype=torch.long, device=device)
    logits_ref, _ = m.forward(inp, past_key_values=None, use_cache=True, temperature=1.0)
    m.reset_ngram_state()
    _, logits_applied, _ = _decode_one_step(m, tok, None, 0, device=device,
                                            temperature=temp, temperature_applied=True)
    diff_applied = (logits_ref - logits_applied).abs().max().item()
    assert diff_applied < 1e-6, \
        f"temperature_applied=True 未跳过温度缩放：diff={diff_applied}"
    # temperature_applied=False → 应传 temperature=0.8 给 forward
    m.reset_ngram_state()
    logits_ref_temp, _ = m.forward(inp, past_key_values=None, use_cache=True, temperature=temp)
    m.reset_ngram_state()
    _, logits_not_applied, _ = _decode_one_step(m, tok, None, 0, device=device,
                                                 temperature=temp, temperature_applied=False)
    diff_not = (logits_ref_temp - logits_not_applied).abs().max().item()
    assert diff_not < 1e-6, \
        f"temperature_applied=False 温度缩放不一致：diff={diff_not}"


def test_generate_deterministic_fixed_seed():
    """model.generate 在固定 seed 下多次调用必须确定性输出（无随机性漂移）。"""
    m, v, ng = _ngram_model()
    ngram_fn = ng.logprob_vector
    prompt = [v.bos_idx, 7, 11, 3]
    kwargs = dict(max_length=12, temperature=1.0, top_k=0,
                  repetition_penalty=1.0, ngram_fn=ngram_fn, ngram_weight=0.3,
                  min_length=3, eos_penalty=-5.0, device='cpu')
    torch.manual_seed(1234)
    out1 = m.generate(list(prompt), **kwargs)
    torch.manual_seed(1234)
    out2 = m.generate(list(prompt), **kwargs)
    assert out1 == out2, "model.generate 固定 seed 下不确定"


def test_generate_vs_batch_parity():
    """相同 prompt、相同参数、相同 seed 下，model.generate（单序列）与
    _generate_candidates_batch（批量 N=1）必须产出完全一致 token 序列——两接口
    共享 sample_next_token 与 _decode_one_step 同一解码驱动原语。"""
    m, v, ng = _ngram_model()
    ngram_fn = ng.logprob_vector
    prompt = [v.bos_idx, 7, 11, 3, 9]
    # 单序列接口
    torch.manual_seed(42)
    seq = m.generate(list(prompt), max_length=12, temperature=1.0, top_k=0,
                     repetition_penalty=1.0, ngram_fn=ngram_fn, ngram_weight=0.3,
                     min_length=3, eos_penalty=-5.0, device='cpu')
    # 批量接口（N=1，单候选，温度相同），重置 seed 以对齐 RNG 消耗序列
    torch.manual_seed(42)
    batched = _generate_candidates_batch(
        m, prompt, temps=[1.0], max_length=12, top_k=0, rep_penalty=1.0,
        device='cpu', ngram_fn=ngram_fn, ngram_weight=0.3,
        pad_id=v.pad_idx, sep_id=getattr(v, 'sep_idx', 4),
        eos_id=getattr(v, 'eos_idx', 3), min_length=3, eos_penalty=-5.0)
    assert len(batched) == 1
    # 批量版返回的是含 prompt 的完整序列；与单序列版去掉相同 prompt 前缀后比较
    assert batched[0][:len(prompt)] == list(prompt)
    gen_batched = batched[0][len(prompt):]
    gen_seq = seq[len(prompt):]
    assert gen_seq == gen_batched, (
        f"generate 与批量接口输出不一致：\n generate={gen_seq}\n batch={gen_batched}")
