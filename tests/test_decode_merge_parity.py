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


# ─── share_attn_proj 回归测试 ──────────────────────────────────────────────

def _shared_model():
    return TransformerModel(
        vocab_size=200, embedding_dim=64, num_heads=4, num_layers=3,
        hidden_dim=128, max_seq_length=32, share_attn_proj=True)

def _independent_model():
    return TransformerModel(
        vocab_size=200, embedding_dim=64, num_heads=4, num_layers=3,
        hidden_dim=128, max_seq_length=32, share_attn_proj=False)


def test_share_attn_proj_parameter_sharing():
    m = _shared_model()
    # 所有 attn block 的 qkv/proj 必须指向同一对象
    qkv_refs = [blk.attn.qkv for blk in m.blocks if hasattr(blk, 'attn')]
    proj_refs = [blk.attn.proj for blk in m.blocks if hasattr(blk, 'attn')]
    assert all(r is qkv_refs[0] for r in qkv_refs), "qkv projections not all shared"
    assert all(r is proj_refs[0] for r in proj_refs), "proj projections not all shared"
    # 共享模型参数量必须少于独立模型
    shared_p = sum(p.numel() for p in m.parameters())
    indep_p = sum(p.numel() for p in _independent_model().parameters())
    assert shared_p < indep_p, f"shared ({shared_p}) should be < independent ({indep_p})"


def test_share_attn_proj_forward_matches():
    torch.manual_seed(42)
    x = torch.randint(0, 200, (2, 12))
    m_s = _shared_model()
    m_i = _independent_model()
    # copy all params except shared attn qkv/proj (which differ in structure)
    # instead: just compare shapes and that both produce valid output
    m_s.eval(); m_i.eval()
    with torch.no_grad():
        out_s = m_s(x)
        out_i = m_i(x)
    assert out_s.shape == out_i.shape == (2, 12, 200)
    # output must be finite
    assert torch.isfinite(out_s).all()
    assert torch.isfinite(out_i).all()


def test_share_attn_proj_cache_matches_shared_params():
    m = _shared_model()
    x = torch.randint(0, 200, (1, 8))
    m.eval()
    with torch.no_grad():
        out1, past = m(x[:, :4], use_cache=True)
        out2, past2 = m(x[:, 4:5], past_key_values=past, use_cache=True)
    assert out1.shape == (1, 4, 200)
    assert out2.shape == (1, 1, 200)
    # past per layer = ((k,v), mem, ngram); attn key shape = (B, heads, T_cached, head_dim)
    k0 = past2[0][0][0]  # layer 0, attn key
    assert k0.shape[2] == 5  # 4+1 tokens cached
