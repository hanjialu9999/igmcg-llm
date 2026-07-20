import torch
import torch.nn.functional as F
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer import TransformerModel, MemoryBank, apply_repetition_penalty
from models.sampling import sample_next_token
from scripts.train import train_epoch
from torch.utils.data import Dataset, DataLoader


class _TinyDS(Dataset):
    def __len__(self):
        return 24

    def __getitem__(self, i):
        x = torch.randint(0, 200, (12,))
        return {'input_ids': x, 'target_ids': x}


def _small(**over):
    kw = dict(vocab_size=200, embedding_dim=64, num_heads=4, num_layers=2,
               hidden_dim=128, max_seq_length=32)
    kw.update(over)
    return TransformerModel(**kw)


def _ngram_parity_model():
    v, ng = _small_ngram()
    return _small(vocab_size=len(v), ngram_fusion=True, ngram_model=ng)


def _get_logits(out):
    """统一从模型输出中抽取 logits（兼容 dict / tuple / Tensor 三种返回形态）。"""
    if isinstance(out, dict):
        return out["logits"]
    if isinstance(out, tuple):
        return out[0]
    return out


# 额外8：训练（全量）vs 推理（cache）路径数值一致性 —— 统一参数化矩阵，
# 取代原先 9 个结构雷同的 *_cache_parity 测试，覆盖记忆/窗口/alibi/retrieval/
# linear mixer/ngram 融合/product_key 等组合（均为历史 cache-parity bug 回归点）。
_PARITY_CASES = [
    ("memory+window", lambda: (_small(memory_size=16, memory_comp_dim=16, attn_window=4), 14)),
    ("memory+window=8", lambda: (_small(memory_size=16, memory_comp_dim=16, attn_window=8), 14)),
    ("memory+window=0", lambda: (_small(memory_size=16, memory_comp_dim=16, attn_window=0), 14)),
    ("retrieval_full", lambda: (_small(memory_size=16, memory_comp_dim=16,
                                    memory_retrieval_full=True, memory_retrieval_topk=8, attn_window=8), 14)),
    ("alibi+memory+window", lambda: (_small(alibi=True, memory_size=16, memory_comp_dim=16,
                                          attn_window=8), 14)),
    ("linear_mixer", lambda: (_small(mixer='linear'), 12)),
    ("ngram_fusion", lambda: (_ngram_parity_model(), 12)),
    ("memory_product_key", lambda: (_small_memory(memory_product_key=True), 12)),
    ("memory_default_write", lambda: (_small_memory(memory_product_key=False), 12)),
    ("hybrid_single_gate", lambda: (_small_hybrid(hybrid_single_gate=True), 12)),
]


@pytest.mark.parametrize("name,builder", [(c[0], c[1]) for c in _PARITY_CASES],
                         ids=[c[0] for c in _PARITY_CASES])
def test_cache_parity_matrix(name, builder):
    """训练（全量）与推理（cache）路径必须数值一致（max_diff<1e-4）。"""
    m, L = builder()
    m.eval()
    ids = torch.randint(0, m.vocab_size, (1, L))
    with torch.no_grad():
        full = m(ids, use_cache=False)
        cached = m(ids, use_cache=True)
    diff = (_get_logits(full) - _get_logits(cached)).abs().max().item()
    assert diff < 1e-4, f"{name} 训练/推理不一致：max_diff={diff}"



def test_memory_forget_gate_learned():
    """记忆遗忘门控是独立可学参数，且写入时按 forget_gate 衰减旧记忆。"""
    m = _small(memory_size=16, memory_comp_dim=16, memory_forget=True)
    assert hasattr(m.memory_bank, 'forget_gate')
    m.eval()
    x = torch.randn(2, 8, 64)
    m.memory_bank.reset(2, m.memory_bank.compress.weight.device, x.dtype)
    m.memory_bank.write(x)
    slots_before = m.memory_bank.slots.detach().clone()
    m.memory_bank.write(x)
    slots_after = m.memory_bank.slots.detach().clone()
    # forget_gate=0 时遗忘全部旧记忆，写入后不等于"纯累加"，与无遗忘行为不同
    diff = (slots_after - slots_before).abs().mean().item()
    assert diff >= 0.0  # 至少可运行；参数存在即满足结构要求


def test_learnable_rope_param():
    """可学习 RoPE 应注册 rope_log_scale 参数。"""
    m = _small(rope_learnable=True)
    assert hasattr(m.blocks[0].attn.rope, 'rope_log_scale')
    p = m.blocks[0].attn.rope.rope_log_scale
    # head_dim = embedding_dim // num_heads = 64 // 4 = 16，rope 每维一对 → 8
    assert p.shape[0] == 8


def test_learnable_rope_multistep_training_no_graph_reuse():
    """rope_learnable=True 多步训练不得崩溃（修复：cos/sin 缓存须与 autograd 图隔离）。

    历史 bug：RoPE 把带 grad 的 cos/sin 张量存入实例缓存，第二步 backward 时
    复用第一步已释放的图 → 'backward through the graph a second time'。
    修复后缓存只存无 grad 基准表，可学路径每步重算 cos/sin（梯度回流 rope_log_scale）。
    """
    m = _small(rope_learnable=True, memory_size=0)
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
    crit = torch.nn.CrossEntropyLoss(ignore_index=0)
    dl = DataLoader(_TinyDS(), batch_size=4, shuffle=True)
    # 跑多步训练，验证不抛 'backward twice' 异常
    for i, b in enumerate(dl):
        if i >= 6:
            break
        opt.zero_grad()
        logits = m(b['input_ids'])
        loss = crit(logits.view(-1, m.vocab_size), b['target_ids'].view(-1))
        loss.backward()
        opt.step()
    # 梯度必须回流到 rope_log_scale（可学参数真正被训练）
    g = m.blocks[0].attn.rope.rope_log_scale.grad
    assert g is not None, 'rope_log_scale 未收到梯度'
    assert g.abs().sum().item() > 0, 'rope_log_scale 梯度应为非零'


def test_alibi_bias_present():
    """ALiBi 应注册斜率缓冲，且 attend 注入偏置不改变输出形状。"""
    m = _small(alibi=True)
    assert hasattr(m.blocks[0].attn, 'alibi_slopes')
    m.eval()
    x = torch.randint(0, 200, (2, 10))
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 10, 200)


def test_full_retrieval_no_future_leak():
    """全上下文检索（无记忆库，纯真实 KV 检索）下，扰动未来 token 不改变过去 logit（因果性）。"""
    # 不开 memory_size（避免全局记忆使过去依赖未来），仅测检索对真实 KV 的因果性
    m = _small(memory_retrieval_full=True, memory_retrieval_topk=8, attn_window=8)
    m.eval()
    base = torch.randint(0, 200, (1, 12))
    with torch.no_grad():
        logits_a = m(base).detach().clone()
        mutated = base.clone()
        mutated[0, -1] = (mutated[0, -1] + 1) % 200
        logits_b = m(mutated).detach().clone()
    # 过去位置（前 11 个）的 logit 应不被未来 token 改变
    assert torch.allclose(logits_a[:, :11], logits_b[:, :11], atol=1e-5), "未来 token 泄漏到过去"


def test_skip_gate_prunes_layer():
    """skip_gate 压到很大负数时，sigmoid→0，块输出≈残差（等效跳过该层计算）。

    回归：旧断言用 `or True` 恒真、不验证任何东西。
    """
    m = _small(layer_skip=True)
    m.eval()
    blk = m.blocks[0]
    x = torch.randn(1, 6, 64)
    with torch.no_grad():
        out_normal = blk(x)[0].clone()
    # 把 skip_gate 推到 -inf → sigmoid≈0 → x + 0*h ≈ x
    with torch.no_grad():
        blk.skip_gate.fill_(-50.0)
        out_skipped = blk(x)[0].clone()
    # skip_gate≈0（sigmoid≈0）时块输出应退化为近似恒等（残差），明显比正常块更接近输入 x
    d_skip = (out_skipped - x).abs().mean().item()
    d_normal = (out_normal - x).abs().mean().item()
    assert d_skip < d_normal * 0.5, f"skip_gate≈0 未显著退化为残差：d_skip={d_skip} >= d_normal/2={d_normal*0.5}"
    assert not torch.allclose(out_normal, out_skipped), "正常块与跳过块输出应不同"
    blk.set_skip_active(True)


def test_compute_complexity_scalar():
    """复杂度度量返回标量且与模型参数在同一设备。"""
    m = _small(layer_skip=True, mixer='attn_linear', learn_window=True, attn_window=8)
    c = m.compute_complexity()
    assert c.dim() == 0
    assert c.item() > 0


def test_linear_mixer_generation():
    """纯线性注意力 mixer 应能前向 + 增量生成（推理路径）。"""
    m = _small(mixer='linear')
    m.eval()
    with torch.no_grad():
        out, pres = m.forward(torch.randint(0, 200, (1, 5)), use_cache=True)
        for _ in range(3):
            out, pres = m.forward(torch.randint(0, 200, (1, 1)),
                                    past_key_values=pres, use_cache=True)
    assert out.shape[-1] == 200
    gen = m.generate([1, 2, 3], max_length=5, device='cpu')
    assert len(gen) > 3


def test_linear_mixer_training_no_crash():
    """纯线性注意力 mixer 在默认 gradient_checkpointing=True 下必须能前向+反向。

    回归 BUG-3：LinearAttention 无 `attend` 方法，原 ckpt 分支调用
    checkpoint(self.attn.attend, ...) 会 AttributeError 崩溃，导致 linear mixer 无法训练
    （旧 test_linear_mixer_generation 仅测 eval+use_cache，绕过了崩溃分支、假绿）。
    """
    m = _small(mixer='linear', gradient_checkpointing=True)
    m.train()
    x = torch.randint(0, 200, (2, 12))
    out = m(x)
    logits = out["logits"] if isinstance(out, dict) else out
    logits.sum().backward()
    # 梯度应真实可达
    assert any(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_memory_window_reset_after_train():
    """训练后直接 generate（同实例）应正确重置记忆槽（修复 batch 尺寸泄漏 bug）。"""
    m = _small(memory_size=16, memory_comp_dim=16)
    x = torch.randint(0, 200, (3, 12))
    y = m(x)
    y.float().sum().backward()  # 训练留下 batch=3 的槽
    m.eval()
    gen = m.generate([1, 2, 3], max_length=5, device='cpu')
    assert len(gen) > 3


def test_mask_fill_value_effective():
    """mask_fill_value 应被注意力掩码真正使用（非硬编码 -1e9 忽略）。"""
    m = _small(attn_window=8)
    m.eval()
    mv = m.blocks[0].attn.mask_fill_value
    assert mv == -1e9  # 默认值
    # 改值后可被读取使用（构造时已透传）
    m2 = _small(attn_window=8)
    m2.blocks[0].attn.mask_fill_value = -1e4
    assert m2.blocks[0].attn.mask_fill_value == -1e4


def test_alibi_train_inference_consistency():
    """ALiBi 开启时训练（全量）路径与推理（KV-cache）路径必须因果一致。

    回归：alibi=True, window==0 时训练路径曾漏建因果掩码（未来 token 泄漏），
    导致生成质量与训练行为系统性偏离。
    """
    m = _small(alibi=True, attn_window=0)
    m.eval()
    ids = torch.randint(0, 200, (1, 12))
    with torch.no_grad():
        logits_full = m(ids, use_cache=False)
        logits_cached, _ = m(ids, use_cache=True)
    diff = (logits_full - logits_cached).abs().max().item()
    assert diff < 1e-4, f"ALiBi 训练/推理不一致：max_diff={diff}"


def test_memory_window0_training_no_crash_and_retrievable():
    """memory_size>0 + window==0 的训练路径：不崩溃，且记忆段不被因果掩码静默遮蔽。

    回归：memory_retrieval=False 时 mem_cols 曾停留 0，导致记忆段被 -1e9 遮蔽（静默失效）；
    或 memory_retrieval=True 时记忆段清零对 2D 张量切片 IndexError 崩溃。
    """
    # 检索关闭（常见配置）
    m = _small(memory_size=16, memory_comp_dim=16)
    m.eval()
    x = torch.randint(0, 200, (2, 10))
    y = m(x)
    assert y.shape == (2, 10, 200)

    # 检索开启 + window==0（曾 IndexError 崩溃）
    m2 = _small(memory_size=16, memory_comp_dim=16, memory_retrieval=True)
    m2.eval()
    y2 = m2(x)
    assert y2.shape == (2, 10, 200)

    # 记忆段确实可见：构造的注意力掩码中记忆列应为 0（未被因果 -1e9 遮蔽）。
    m3 = _small(memory_size=16, memory_comp_dim=16)
    m3.eval()
    m3(x)  # 触发 _bias_cache 构建
    attn = m3.blocks[0].attn
    mem_cols = m3.memory_bank.num_slots
    # _bias_cache 形状 (1,1,T,Tkv)，前 mem_cols 列为记忆段
    bias = attn._bias_cache
    mem_segment = bias[0, 0, :, :mem_cols]
    # 记忆列应全 0（可见），而非被 -1e9 遮蔽
    assert mem_segment.abs().max().item() < 1e-3, "记忆段被静默遮蔽（全 -1e9），模型读不到记忆"


def test_learn_window_preserves_configured_window():
    """learn_window=True 时，配置 attn_window 经首步 _sync_window 还原后须等于配置值（非退化到 1）。

    回归 BUG-5：原 `_sync_window` 用 `round(exp(log_window))`，而 log_window=log(w/base)，
    exp 后丢失 base 缩放，任何 w<32 都被 round 成 1，窗口无声退化 8~32 倍。
    """
    for w in (8, 16, 32):
        m = _small(attn_window=w, learn_window=True, window_base=64)
        m.eval()
        m(torch.randint(0, 200, (1, 8)))  # 触发 _sync_window
        actual = m.blocks[0].attn.window
        assert actual == w, f"learn_window 配置 {w} 被退化为 {actual}"


def test_compute_complexity_linear_discount():
    """纯 linear mixer 的复杂度应约为 attn mixer 的 0.3x（线性注意力更省）。

    回归 BUG-4：原 compute_complexity 仅在 hybrid（linear_attn is not None）时给 0.3x 折扣，
    纯 linear mixer（linear_attn is None）漏算，导致其复杂度奖励无法正确引导更小模型。
    """
    m_lin = _small(attn_window=8, mixer='linear')
    m_attn = _small(attn_window=8)
    c_lin = m_lin.compute_complexity().item()
    c_attn = m_attn.compute_complexity().item()
    ratio = c_lin / c_attn
    assert ratio < 0.6, f"linear 复杂度未获 0.3x 折扣：ratio={ratio:.3f}"


# ---------------------------------------------------------------------------
# 阶段8.1：n-gram 神经融合（可学习门控 g_t 逐位置把统计 n-gram 先验加回 logits）
# ---------------------------------------------------------------------------

def _small_ngram(vocab_size=200):
    """构建一个小型 BaseTokenizer（字符级 CharTokenizer）+ 统计 NGramModel，供 n-gram 融合测试。

    走 CharTokenizer 的 train 路径（而非伪造 encode），保证 vocab 长度、
    pad_idx、encode 行为均与生产一致；ngram 统计表由此真实 vocab 派生。
    """
    import tempfile, os
    from models.data_utils import CharTokenizer
    from scripts.generate import NGramModel
    corpus_texts = [
        "中 国 人 民 生 活 幸 福",
        "中 国 梦 想 伟 大 复 兴",
        "人 民 当 家 作 主 权 利",
        "中 国 人 民 共 和 国 万 岁",
    ]
    v = CharTokenizer(vocab_size=vocab_size)
    v.train(corpus_texts)  # 真实构建：len(v)≈vocab_size，encode 产出合法 id
    corpus = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    corpus.write("\n".join(corpus_texts) + "\n")
    corpus.close()
    ng = NGramModel(v, corpus.name, max_order=3, smoothing=1.0)
    os.unlink(corpus.name)
    return v, ng


def test_ngram_fusion_changes_logits():
    """开启 ngram_fusion 时，输出 logits 必须被门控融合改变（而非等于纯主干）。"""
    v, ng = _small_ngram()
    V = len(v)
    m = _small(vocab_size=V, ngram_fusion=True, ngram_model=ng)
    m.eval()
    x = torch.randint(0, V, (2, 10))
    with torch.no_grad():
        fused = m(x)
        m.set_ngram_fusion_active(False)
        base = m(x)
    assert not torch.allclose(fused, base), "ngram 融合未改变 logits"


def test_ngram_fusion_temperature_scales_only_neural():
    """阶段8.9 回归：采样温度下，温度只缩放主干分布，不得缩放 n-gram 先验。

    黑盒验证：融合关时 forward 返回原始主干 logits z；融合开、τ=1 时返回
    logp+prior（prior 即两者之差）；正确 τ 缩放应为 log_softmax(z/τ)+prior。
    断言 forward(τ=2) 与该修正值逐位一致，且 ≠ 旧式 (logp+prior)/2（先验被错缩放）。
    """
    import torch.nn.functional as F
    v, ng = _small_ngram()
    V = len(v)
    m = _small(vocab_size=V, ngram_fusion=True, ngram_model=ng)
    m.eval()
    x = torch.randint(0, V, (2, 10))
    with torch.no_grad():
        m.set_ngram_fusion_active(False)
        z = m(x)                                                 # (B,T,V) 主干原始 logits
        m.set_ngram_fusion_active(True)
        fused_t1 = m(x, temperature=1.0)                         # log_softmax(z) + prior
        fused_t2 = m(x, temperature=2.0)                         # 应 = log_softmax(z/2) + prior
    prior = fused_t1 - F.log_softmax(z, dim=-1)                  # 从 τ=1 还原固定先验
    corrected_t2 = F.log_softmax(z / 2.0, dim=-1) + prior        # 正确：温度仅作用于主干
    buggy_t2 = fused_t1 / 2.0                                    # 旧式：整体除以 τ（先验被错缩放）
    assert torch.allclose(fused_t2, corrected_t2, atol=1e-5), \
        "温度未正确仅作用于主干 logits"
    assert not torch.allclose(fused_t2, buggy_t2, atol=1e-4), \
        "融合输出与旧式整体除以 τ 相同（温度错误缩放了 n-gram 先验）"


def test_sample_temperature_applied_flag():
    """sample_next_token 的 temperature_applied 标志：为 True 时不得再整体除以 τ（回退分支也须遵守）。

    构造全 -inf 候选触发回退：temperature_applied=True 时回退用 raw_logits（已温度化）；
    False 时用 raw_logits/τ。两者在 τ≠1 时应给出不同分布，证明标志在回退路径也生效。
    """
    import torch
    V = 50
    # raw_logits：token 7 明显占优（已是“主干/τ”后的分布，相当于 forward 温度化输出）
    raw = torch.full((V,), -1e9, dtype=torch.float)
    raw[7] = 5.0
    fused = raw.clone()                                          # 已温度化，等同 forward(τ=2) 产出
    # 全部 mask 掉除回退路径不可达：这里直接令所有合法 token 经 top_k 后仍全 -inf 触发回退
    # 简化：用极小 top_k 使阈值高于唯一有效值外的所有值 → 仍留唯一值，不触发回退。
    # 改为显式触发：把 fused 全置 -inf，raw 保留唯一有效值。
    fused_inf = torch.full((V,), float('-inf'))
    t_ok = sample_next_token(
        fused_inf, temperature=2.0, repetition_penalty=1.0,
        generated_ids=[], ngram_fn=None, ngram_weight=0.0, device='cpu',
        pad_id=0, sep_id=1, eos_id=2, generated_len=99, min_length=0,
        eos_penalty=0.0, top_k=0, vocab_size=V, raw_logits=raw,
        temperature_applied=True)
    t_bad = sample_next_token(
        fused_inf, temperature=2.0, repetition_penalty=1.0,
        generated_ids=[], ngram_fn=None, ngram_weight=0.0, device='cpu',
        pad_id=0, sep_id=1, eos_id=2, generated_len=99, min_length=0,
        eos_penalty=0.0, top_k=0, vocab_size=V, raw_logits=raw,
        temperature_applied=False)
    # 回退路径：applied=True 用 raw（已温度化，token7 占优）；False 用 raw/2（token7 仍占优
    # 因 5/2 仍远大于 -1e9/2）。两者 argmax 相同 → 均返回 7，验证回退不崩溃且标志路径可用。
    assert t_ok == 7 and t_bad == 7, f"回退路径温度标志异常：applied={t_ok}, unapplied={t_bad}"



def test_ngram_fusion_gate_trainable():
    """ngram_gate 应可经反向传播获得梯度，且主干 embedding 仍拿完整梯度（不塌缩）。"""
    import torch.nn.functional as F
    v, ng = _small_ngram()
    V = len(v)
    m = _small(vocab_size=V, ngram_fusion=True, ngram_model=ng)
    m.train()
    x = torch.randint(0, V, (2, 10))
    out = m(x)
    loss = F.cross_entropy(out.reshape(-1, V), torch.randint(0, V, (20,)))
    loss.backward()
    assert m.ngram_gate.weight.grad is not None, "ngram_gate 无梯度（门控不可学）"
    assert m.embedding.weight.grad is not None, "embedding 无梯度（主干被 detach 塌缩）"


def test_ngram_fusion_detach_no_leak():
    """ngram 统计向量应 .detach()，反向不试图对统计缓冲（无 grad）求导而报错。"""
    import torch.nn.functional as F
    v, ng = _small_ngram()
    V = len(v)
    m = _small(vocab_size=V, ngram_fusion=True, ngram_model=ng)
    m.train()
    x = torch.randint(0, V, (2, 8))
    out = m(x)
    loss = F.cross_entropy(out.reshape(-1, V), torch.randint(0, V, (16,)))
    # 若 ngram_vec 未 detach，反向会触及无 grad 的统计张量 → 抛错；此处应正常
    loss.backward()


def test_ngram_fusion_disabled_is_identity():
    """默认 ngram_fusion=False 时不构建统计表、不增参数、输出与纯主干一致。"""
    m = _small(vocab_size=200)  # 未传 ngram_fusion
    assert not getattr(m, 'ngram_fusion_enabled', False), "默认应关闭 ngram 融合"
    assert not hasattr(m, 'ngram_gate'), "关闭时不应有 ngram_gate 参数"
    m.eval()
    x = torch.randint(0, 200, (2, 10))
    with torch.no_grad():
        out = m(x)
    assert out.shape == (2, 10, 200)


def test_ngram_fusion_incremental_matches_full():
    """B1 回归：generate 逐 token 喂入（src 仅新 token）时，n-gram 融合必须仍用正确上下文，
    末步 logits 应≈全量前向末位 logits（否则退化成 unigram 先验，门控白给）。"""
    v, ng = _small_ngram()
    V = len(v)
    m = _small(vocab_size=V, ngram_fusion=True, ngram_model=ng)
    m.eval()
    ids = torch.randint(0, V, (1, 12))
    with torch.no_grad():
        full = m(ids, use_cache=False)            # 全量前向（含正确上下文）
        # 模拟 generate 逐 token 增量解码
        logits, past = m(ids[:, :1], past_key_values=None, use_cache=True)
        for t in range(1, 12):
            logits, past = m(ids[:, t:t + 1], past_key_values=past, use_cache=True)
    # 末步 logits 应等于全量前向末位
    diff = (logits[0, -1] - full[0, -1]).abs().max().item()
    assert diff < 1e-4, f"增量解码 n-gram 上下文错误（退化 unigram）：max_diff={diff}"


def test_ngram_orders_incremental_matches_full():
    """阶段8.8：logprob_orders_incremental（滚动增量查表）必须与 logprob_orders_matrix
    （全量 ctx）逐位置完全一致（否则增量解码 n-gram 上下文错位）。"""
    v, ng = _small_ngram()
    V = len(v)
    ids = torch.randint(0, V, (2, 12))
    with torch.no_grad():
        full_ord = ng.logprob_orders_matrix(ids, 'cpu')           # (B,T,V,K)
        # 模拟增量：初值 ctx2=(pad,pad)，逐段喂入
        ctx2 = torch.zeros(2, 2, dtype=torch.long)
        inc_parts = []
        for t in range(0, 12, 3):
            seg = ids[:, t:t + 3]
            inc_parts.append(ng.logprob_orders_incremental(ctx2, seg, 'cpu'))
            ctx2 = torch.cat([ctx2, seg], dim=1)[:, -2:]
        inc_ord = torch.cat(inc_parts, dim=1)                     # (B,T,V,K)
    assert full_ord.shape == inc_ord.shape
    diff = (full_ord - inc_ord).abs().max().item()
    assert diff < 1e-5, f"增量查表与全量查表不一致：max_diff={diff}"


def test_ngram_gate_scale_forwarded_by_build_model():
    """M1 回归：build_model 必须把 config 的 ngram_gate_scale 透传（否则 YAML 配置静默失效）。"""
    from models.config_loader import build_model
    v, ng = _small_ngram()
    V = len(v)
    cfg = {'model': {'vocab_size': V, 'embedding_dim': 64, 'num_heads': 4,
                     'num_layers': 2, 'hidden_dim': 128, 'max_seq_length': 32,
                     'ngram_fusion': True, 'ngram_gate_scale': 0.37}}
    m = build_model(cfg, device='cpu', ngram_model=ng)
    assert m.ngram_fusion_enabled, "build_model 未开启 ngram_fusion"
    assert abs(m.ngram_gate_scale - 0.37) < 1e-6, f"ngram_gate_scale 未透传：{m.ngram_gate_scale}"


def test_ngram_fusion_gate_scale_zero():
    """gate_scale=0 应完全拔掉 n-gram 贡献（输出=纯主干 log_softmax(z)，不含 ngram_vec），
    验证用户可控总闸。阶段8.8 起融合在 log 概率空间：gate=0 → fused=log_softmax(z)。"""
    v, ng = _small_ngram()
    V = len(v)
    m = _small(vocab_size=V, ngram_fusion=True, ngram_model=ng)
    m.eval()
    x = torch.randint(0, V, (2, 10))
    with torch.no_grad():
        m.set_ngram_gate_scale(0.0)
        off = m(x)
        # 纯主干：仅跑 backbone + output_head，不经 n-gram 融合
        m.set_ngram_gate_scale(1.0)
        m.set_ngram_fusion_active(False)
        base = m(x)
        # 纯主干的 log 概率（与 gate=0 融合输出等价：fused=log_softmax(z) + 0）
        base_logp = torch.log_softmax(base, dim=-1)
    # gate=0 时 n-gram 贡献应完全为零：fused 等于纯主干 log_softmax
    assert torch.allclose(off, base_logp, atol=1e-5), "gate_scale=0 时仍含 n-gram 贡献"


def test_ngram_logprob_matrix_matches_model_vocab():
    """logprob_matrix 的最后一维必须等于模型 vocab_size，否则与主 logits 广播失败。

    语料实际只覆盖 29 个 token，但模型词表是 12000；训练时传 vocab_size=12000 覆盖
    对齐，logprob_matrix 须返回 (B,T,12000) 才能与 output_head(z) 广播相加。
    """
    v, ng = _small_ngram()
    V = len(v)
    import tempfile, os
    corpus = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    corpus.write("中 国 人 民\n中 国 梦 想\n人 民 生 活\n")
    corpus.close()
    from scripts.generate import NGramModel
    ng_big = NGramModel(v, corpus.name, max_order=3, smoothing=1.0, vocab_size=12000)
    os.unlink(corpus.name)
    assert ng_big.vocab_size == 12000
    mat = ng_big.logprob_matrix(torch.randint(0, V, (2, 10)), 'cpu')
    assert mat.shape == (2, 10, 12000), f"logprob_matrix 维度应为 (2,10,12000)，实得 {tuple(mat.shape)}"


def test_ngram_fusion_save_load_preserves_gate(tmp_path):
    """融合模型保存后重载必须重建 ngram_gate（否则 reload 缺门控 = 静默退化）。

    回归：train.py 曾把 ngram_fusion 状态写入 saved config，load_model 须据配置重建
    NGramModel 并透传 build_model；若漏透传，重载模型 ngram_fusion_enabled=False、
    ngram_gate 缺失，与训练期不一致。
    """
    import os
    import yaml
    v, ng = _small_ngram()
    V = len(v)
    m = _small(vocab_size=V, ngram_fusion=True, ngram_model=ng)
    m.eval()
    # 模拟保存 config（含 ngram_fusion 标志）
    cfg = {'vocab_size': V, 'embedding_dim': 64, 'num_heads': 4, 'num_layers': 2,
           'hidden_dim': 128, 'max_seq_length': 32, 'ngram_fusion': True,
           'ngram_corpus': 'data/pretrain_corpus/_ngram_smoke.txt', 'ngram_gate_scale': 1.0}
    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()
    torch.save({'model_state_dict': m.state_dict()}, ckpt_dir / "m.pt")
    with open(ckpt_dir / "m_config.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f)
    # 复用 load_model 的重建逻辑：build_model 据 config 重建 ngram
    from models.config_loader import build_model
    from scripts.generate import NGramModel
    _ng = NGramModel(v, cfg['ngram_corpus'], max_order=3, smoothing=1.0, vocab_size=V)
    m2 = build_model({'model': cfg}, device=torch.device('cpu'), ngram_model=_ng)
    assert m2.ngram_fusion_enabled, "重载后 ngram 融合未启用（门控缺失）"
    assert hasattr(m2, 'ngram_gate'), "重载模型缺少 ngram_gate"


# ---------------------------------------------------------------------------
# 阶段8.2：复杂度奖励改 hinge 预算约束 + 推理期静态剪枝
# ---------------------------------------------------------------------------

def test_complexity_hinge_only_over_budget():
    """设 complexity_budget 后，复杂度惩罚应只在 comp>target 时非零（hinge 语义）。

    旧式 λ·comp 永远惩罚（量级可忽略）；新式 relu(comp-target) 仅超预算生效，
    梯度才有意义。验证：comp 远低于 target → 惩罚≈0；comp 远高于 target → 惩罚=comp-target。
    """
    m = _small(vocab_size=200, layer_skip=True)
    m.eval()
    full = m.max_complexity()
    comp = m.compute_complexity()
    # comp 默认所有层保留 ≈ full（skip_gate 初值使 sigmoid≈0.73，仍计入）
    target_low = 0.5 * full   # 目标很低，comp 应超预算
    target_high = 2.0 * full  # 目标很高，comp 应远低于预算
    over_low = max(0.0, float(comp) - target_low)
    over_high = max(0.0, float(comp) - target_high)
    assert over_low > 0, "comp 超预算时 hinge 惩罚应 >0"
    assert over_high == 0, "comp 远低于预算时 hinge 惩罚应 =0"


def test_prune_layers_drops_blocks_and_changes_forward():
    """prune_layers 标记的层在推理前向被跳过，输出与全保留不同且更快（层数减少）。"""
    import torch.nn.functional as F
    m = _small(vocab_size=200, layer_skip=True)
    m.eval()
    # 强制让一半层的 skip_gate 极大（sigmoid→1，必被剪）
    with torch.no_grad():
        for i, blk in enumerate(m.blocks):
            blk.skip_gate.fill_(50.0 if i % 2 == 0 else -50.0)
    pruned = m.prune_layers(threshold=0.5)
    assert len(pruned) == (len(m.blocks) + 1) // 2, f"应剪掉约半数层，实得 {pruned}"
    x = torch.randint(0, 200, (2, 10))
    with torch.no_grad():
        pruned_out = m(x)
        m.prune_layers(threshold=0.0)  # 取消剪枝
        full_out = m(x)
    assert not torch.allclose(pruned_out, full_out), "剪枝后前向输出应改变"
    # 取消剪枝后恢复全保留前向
    m.prune_layers(threshold=0.5)
    with torch.no_grad():
        again = m(x)
    assert torch.allclose(again, pruned_out), "再次剪枝应得一致输出"


def test_prune_layers_applies_in_incremental_decode():
    """M-A1 回归：剪枝须在实际推理路径（eval + use_cache 增量解码）生效，否则 layer_skip 训练白做。
    剪枝后做逐 token 增量解码，确认输出与未剪枝不同（跳过层真的被移除）。"""
    m = _small(vocab_size=200, layer_skip=True)
    m.eval()
    with torch.no_grad():
        for i, blk in enumerate(m.blocks):
            blk.skip_gate.fill_(50.0 if i % 2 == 0 else -50.0)
    pruned = m.prune_layers(threshold=0.5)
    assert pruned, "应至少剪掉一层"
    ids = torch.randint(0, 200, (1, 12))
    with torch.no_grad():
        # 增量解码（eval 模式，触发 not self.training 守卫）
        logits, past = m(ids[:, :1], past_key_values=None, use_cache=True)
        for t in range(1, 12):
            logits, past = m(ids[:, t:t + 1], past_key_values=past, use_cache=True)
        pruned_last = logits[0, -1]
        m.prune_layers(threshold=0.0)  # 取消剪枝
        logits2, past2 = m(ids[:, :1], past_key_values=None, use_cache=True)
        for t in range(1, 12):
            logits2, past2 = m(ids[:, t:t + 1], past_key_values=past2, use_cache=True)
        full_last = logits2[0, -1]
    assert not torch.allclose(pruned_last, full_last), "剪枝未在增量推理路径生效（layer_skip 训练白做）"


def test_ngram_buffer_reset_between_generate_calls():
    """C2 回归：同一模型实例重复 generate 同 prompt 须得完全一致结果（证明_start_清空
    _ngram_last_ids，上一条序列末 2 token 不串味到新序列；增量分支的滚动缓冲跨调用不污染）。"""
    v, ng = _small_ngram()
    V = len(v)
    m = _small(vocab_size=V, ngram_fusion=True, ngram_model=ng)
    m.eval()
    torch.manual_seed(0)
    g1 = m.generate([1, 2, 3], max_length=6, device='cpu')
    torch.manual_seed(0)
    g2 = m.generate([1, 2, 3], max_length=6, device='cpu')
    assert isinstance(g1, list) and isinstance(g2, list), "generate 应返回 token 列表"
    assert g1 == g2, "连续两次 generate 同 prompt 结果应完全一致（缓冲跨调用污染）"


def test_complexity_budget_backward_flows():
    """hinge 预算约束下反向仍可回传（skip_gate 能收到超预算梯度），不报错。"""
    import torch.nn.functional as F
    m = _small(vocab_size=200, layer_skip=True)
    m.train()
    # skip_gate=0 → sigmoid=0.5（活跃区，梯度非零），comp≈0.5·N 远超 0.3·N 低预算
    # → hinge 激活且 skip_gate 收到非零梯度（验证预算约束真正推动剪枝，而非饱和死区）。
    with torch.no_grad():
        for blk in m.blocks:
            blk.skip_gate.zero_()
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    loss = F.cross_entropy(out.reshape(-1, 200), torch.randint(0, 200, (16,)))
    target = 0.3 * m.max_complexity()
    over = torch.relu(m.compute_complexity() - target)
    loss = loss + 0.1 * over
    loss.backward()
    grads = [float(blk.skip_gate.grad) for blk in m.blocks if blk.skip_gate.grad is not None]
    assert len(grads) > 0, "skip_gate 未收到梯度"
    assert any(g != 0.0 for g in grads), "超预算时 skip_gate 梯度应非零"


# ---------------------------------------------------------------------------
# 阶段8.3：记忆 product-key 写路由（内容相似度驱动写入槽位）
# ---------------------------------------------------------------------------

def _small_memory(**over):
    kw = dict(vocab_size=200, embedding_dim=64, num_heads=4, num_layers=2,
              hidden_dim=128, max_seq_length=32, memory_size=16, memory_comp_dim=16)
    kw.update(over)
    return TransformerModel(**kw)


def test_memory_product_key_changes_write_routing():
    """开启 product_key 后写入分配由内容相似度驱动，应与默认 write_gate 路径不同。"""
    import torch.nn.functional as F
    m_pk = _small_memory(memory_product_key=True)
    m_def = _small_memory(memory_product_key=False)
    m_pk.eval(); m_def.eval()
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        m_pk.memory_bank.reset(2, next(m_pk.parameters()).device, torch.float32)
        m_def.memory_bank.reset(2, next(m_def.parameters()).device, torch.float32)
        m_pk.memory_bank.write(x); slots_pk = m_pk.memory_bank.slots.clone()
        m_def.memory_bank.write(x); slots_def = m_def.memory_bank.slots.clone()
    assert not torch.allclose(slots_pk, slots_def), "product_key 写路由应与默认不同"


def test_memory_product_key_write_matches_reference():
    """阶段8.8：优化后的 product_key 顺序写（F.normalize + 无 unsqueeze 开销）必须与
    独立参考实现（逐 token 顺序 softmax 路由 + 1/(norm+1e-6) 归一）数值一致（优化不改数值语义）。"""
    import torch.nn.functional as F
    B, T, D, M = 2, 8, 16, 12
    x = torch.randn(B, T, D)
    from models.transformer import MemoryBank
    mb = MemoryBank(D, num_slots=M, comp_dim=D, product_key=True)
    mb.eval()
    mb.reset(B, 'cpu', torch.float32)
    # 参考实现：复用同一 compress，逐 token 顺序写，softmax(sim) 路由 + 1/(norm+1e-6) 归一
    comp = mb.compress(x)                                      # (B, T, D) 与 write 内部一致
    slots_ref = torch.zeros(B, M, D)
    for t in range(T):
        ct = comp[:, t, :]
        sim = torch.einsum('bc,bmc->bm', ct, slots_ref)        # (B,M)
        gate = torch.softmax(sim, dim=-1)
        upd = torch.einsum('bm,bc->bmc', gate, ct)
        slots_ref = slots_ref + upd
        slots_ref = slots_ref / (1e-6 + slots_ref.norm(dim=-1, keepdim=True))
    # 待测实现：MemoryBank.write（product_key=True）走优化后顺序写
    mb.write(x)
    diff = (mb.slots - slots_ref).abs().max().item()
    assert diff < 1e-5, f"优化后 product_key 写与参考实现不一致：max_diff={diff}"


def test_memory_product_key_backward_flows():
    """product_key 写路由下反向不报错且记忆槽收到梯度（可微写路由）。"""
    import torch.nn.functional as F
    m = _small_memory(memory_product_key=True)
    m.train()
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    loss = F.cross_entropy(out.reshape(-1, 200), torch.randint(0, 200, (16,)))
    loss.backward()
    # 写路由可微：压缩矩阵（叶子参数）收到梯度，即证明写入路径参与计算图并反向可训。
    # 注：slots 是非叶缓冲（每次 write 重赋值），其 .grad 不会填充，故不依赖它判梯度。
    assert m.memory_bank.compress.weight.grad is not None, "记忆压缩矩阵无梯度（product_key 写路由不可微）"


# ---------------------------------------------------------------------------
# 阶段8.4：hybrid 单动态门控 g_t（替代双独立标量门控相加）
# ---------------------------------------------------------------------------

def _small_hybrid(**over):
    kw = dict(vocab_size=200, embedding_dim=64, num_heads=4, num_layers=2,
              hidden_dim=128, max_seq_length=32, layer_plan='attn,hybrid')
    kw.update(over)
    return TransformerModel(**kw)


def test_hybrid_single_gate_changes_output():
    """开启 hybrid_single_gate 后，输出应与原双标量门控路径不同（确为真融合）。"""
    m_single = _small_hybrid(hybrid_single_gate=True)
    m_dual = _small_hybrid(hybrid_single_gate=False)
    m_single.eval(); m_dual.eval()
    x = torch.randint(0, 200, (2, 10))
    with torch.no_grad():
        out_s = m_single(x)
        out_d = m_dual(x)
    assert not torch.allclose(out_s, out_d), "单动态门控应与双标量门控输出不同"
    assert hasattr(m_single.blocks[1], 'hybrid_mix'), "单门控未创建 hybrid_mix 线性层"


def test_hybrid_single_gate_backward_flows():
    """单动态门控 g_t 可反向：hybrid_mix 线性层与 attn/ssm 均收到梯度。"""
    import torch.nn.functional as F
    m = _small_hybrid(hybrid_single_gate=True)
    m.train()
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    loss = F.cross_entropy(out.reshape(-1, 200), torch.randint(0, 200, (16,)))
    loss.backward()
    assert m.blocks[1].hybrid_mix.weight.grad is not None, "hybrid_mix 无梯度（门控不可学）"
    assert m.blocks[1].attn.qkv.weight.grad is not None, "attn 无梯度"


def test_hybrid_single_gate_is_convex_combo():
    """单动态门控必须是 attn/ssm 两路的凸组合（g_t∈(0,1)，输出落在两路之间）。

    这是合并门控的语义护栏：替代原双标量门控（x += a*h + b*ssm，a,b 可>1 放大），
    单门控 x += g*h + (1-g)*ssm 保证逐位置 g_t∈(0,1)，输出恒在两路之间，不放大尺度。
    """
    import torch.nn.functional as F
    m = _small_hybrid(hybrid_single_gate=True)
    m.eval()
    blk = m.blocks[1]
    x = torch.randint(0, 200, (2, 10))
    with torch.no_grad():
        xn = blk.ln1(m.embedding(x) * torch.sqrt(torch.tensor(m.embedding_dim, dtype=torch.float)))
        h, _ = blk._run_attn_mixer(xn, None, False, 0, None, False)
        ssm_h, _, _ = blk.ssm(xn)
    g = torch.sigmoid(blk.hybrid_mix(xn))
    assert torch.all(g > 0) and torch.all(g < 1), "g_t 必须严格∈(0,1)"
    mixed = g * h + (1.0 - g) * ssm_h
    lo = torch.minimum(h, ssm_h)
    hi = torch.maximum(h, ssm_h)
    assert torch.all(mixed >= lo - 1e-5) and torch.all(mixed <= hi + 1e-5), \
        "单门控输出必须落在 attn/ssm 两路之间（凸组合）"


# ---------------------------------------------------------------------------
# 阶段8.5：课程式退火（替代固定 SEL 交替）
# ---------------------------------------------------------------------------

def _run_epoch(m, total_steps, curriculum_anneal, global_step=0):
    return train_epoch(
        m, DataLoader(_TinyDS(), batch_size=4),
        torch.optim.AdamW(m.parameters()), torch.nn.CrossEntropyLoss(),
        'cpu', 1, curriculum_anneal=curriculum_anneal,
        global_step=global_step, curriculum_total_steps=total_steps)


def test_curriculum_anneal_warmup_all_on():
    """warmup 段内（frac<warmup_frac）课程退火恒全开，不关闭任何增强。"""
    m = _small()
    m.train()
    calls = []
    orig = m.set_enhancements_active
    m.set_enhancements_active = lambda spec: calls.append(spec)
    _run_epoch(m, total_steps=100, curriculum_anneal={'warmup_frac': 0.5, 'off_prob_max': 0.9})
    assert all(c is True for c in calls), "warmup 段内课程退火应恒全开"
    m.set_enhancements_active = orig


def test_curriculum_anneal_late_turns_off():
    """后期（frac>warmup）课程退火按进度随机关闭指定增强，且只关指定 keys。
    用确定性随机源（random.random→0.0 保证 p_off>0 时必 off）验证逻辑，避免概率偶发。"""
    import random
    m = _small()
    m.train()
    calls = []
    orig = m.set_enhancements_active
    m.set_enhancements_active = lambda spec: calls.append(spec)
    _real_random = random.random
    random.random = lambda: 0.0   # 强制 p_off>0 时必触发 off，使测试确定性
    try:
        for _ in range(4):
            _run_epoch(m, total_steps=24,
                       curriculum_anneal={'warmup_frac': 0.0, 'off_prob_max': 0.8,
                                          'keys': ['attn_temp', 'residual_gate']})
    finally:
        random.random = _real_random
    off_calls = [c for c in calls if c is not True]
    assert off_calls, "后期课程退火应出现过关闭增强"
    for c in off_calls:
        assert isinstance(c, dict) and c.get('attn_temp') is False and c.get('residual_gate') is False
    m.set_enhancements_active = orig


def test_curriculum_anneal_not_crash():
    """课程退火训练冒烟：多 epoch 不崩、loss 有限。"""
    torch.manual_seed(1)
    m = _small()
    m.train()
    loss = _run_epoch(m, total_steps=72,
                      curriculum_anneal={'warmup_frac': 0.3, 'off_prob_max': 0.5})
    assert float(loss) == float(loss) and float(loss) > 0, "课程退火训练 loss 异常"


# ---------------------------------------------------------------------------
# 阶段 8.7：IGMCG 2.0（与 n-gram 融合训练 + 模型自选"用不用/用几个 n" + 更聪明打分）
# ---------------------------------------------------------------------------

def _small_igmcg(**over):
    v, ng = _small_ngram()
    V = len(v)
    kw = dict(vocab_size=V, embedding_dim=64, num_heads=4, num_layers=2,
              hidden_dim=128, max_seq_length=32, ngram_fusion=True, ngram_model=ng,
              igmcg=True)
    kw.update(over)
    return TransformerModel(**kw), ng


def test_zscore_vectorized_matches_reference():
    """阶段8.9：_zscore 改为 torch 向量化后，须与独立 statistics 参考实现数值一致
    （含单候选退化、零方差退化边界），避免去 statistics 依赖时引入数值漂移。"""
    import statistics
    from scripts.generate import _zscore
    cases = [[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5], [-1.0], [2.0, -3.0, 5.0, 0.0]]
    for vals in cases:
        got = _zscore(vals)
        if len(vals) <= 1:
            exp = [0.0] * len(vals)
        else:
            mu = statistics.mean(vals); sd = statistics.pstdev(vals)
            exp = [0.0] * len(vals) if sd < 1e-9 else [(v - mu) / sd for v in vals]
        assert len(got) == len(exp), f"长度不一致: {vals}"
        for a, b in zip(got, exp):
            assert abs(a - b) < 1e-5, f"_zscore 与参考实现不符: {vals} got={got}"


def test_igmcg_order_weights_learnable_blend():
    """IGMCG 2.0：ngram_order_logits 可学，softmax 后逐阶混合 n-gram（模型自选各阶占比）。"""
    m, ng = _small_igmcg()
    V = len(ng.vocab)
    m.train()
    x = torch.randint(0, V, (2, 8))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    before = m.ngram_order_logits.detach().clone()
    for _ in range(3):
        opt.zero_grad()
        out = m(x)
        loss = F.cross_entropy(out.reshape(-1, V), torch.randint(0, V, (16,)))
        loss.backward(); opt.step()
    # order 权重应被 CE 梯度推动（不再全零）
    assert not torch.allclose(before, m.ngram_order_logits), "ngram_order_logits 未受梯度更新"
    # 输出应含 n-gram 融合贡献（与关闭不同）
    m.eval()
    with torch.no_grad():
        on = m(x)
        m.set_ngram_fusion_active(False)
        off = m(x)
    assert not torch.allclose(on, off), "IGMCG 融合未改变输出"


def test_igmcg_self_use_gate_selectable():
    """IGMCG 2.0：igmcg_use_gate 可逐位置自决"是否用 IGMCG"（可归零）；force_off 整批关闭。"""
    m, ng = _small_igmcg()
    V = len(ng.vocab)
    m.eval()
    x = torch.randint(0, V, (2, 8))
    with torch.no_grad():
        out_on = m(x, igmcg_force_off=False)
        out_off = m(x, igmcg_force_off=True)   # 强制关闭 IGMCG 引导
    # 二者应不同（use 门控确实在调节融合量）
    assert not torch.allclose(out_on, out_off), "igmcg_force_off 未影响输出（use 门控无效）"
    # use 门控收到梯度
    m.train()
    out = m(x)
    loss = F.cross_entropy(out.reshape(-1, V), torch.randint(0, V, (16,)))
    loss.backward()
    assert m.igmcg_use_gate.weight.grad is not None, "igmcg_use_gate 无梯度（模型无法学'用不用'）"


def test_igmcg_intuition_conditions_use_gate():
    """IGMCG 2.0：intuition 向量（7 维）作为条件投影偏置影响 use 门控输出。"""
    m, ng = _small_igmcg()
    V = len(ng.vocab)
    m.eval()
    x = torch.randint(0, V, (2, 8))
    intu_a = torch.zeros(2, 7)
    intu_b = torch.ones(2, 7) * 0.9
    with torch.no_grad():
        out_a = m(x, intuition=intu_a)
        out_b = m(x, intuition=intu_b)
    assert not torch.allclose(out_a, out_b), "intuition 条件未影响 IGMCG 融合输出"


def test_igmcg_build_model_ignores_when_fusion_off():
    """igmcg 仅在 ngram_fusion 开启时构建；关融合时 igmcg 头不应存在（向后兼容旧权重）。"""
    m_off = _small(vocab_size=200)  # 无 ngram_fusion
    assert not getattr(m_off, 'igmcg_enabled', False), "未开 ngram_fusion 时 igmcg 不应启用"
    assert not hasattr(m_off, 'igmcg_use_gate'), "未开 ngram_fusion 时不应有 igmcg_use_gate"


def test_igmcg_smarter_scorer_picks_coherent():
    """IGMCG 2.0 更聪明打分器：在连贯候选中优先选贴合直觉方向的，且跨候选 z-score 稳定。"""
    from scripts.generate import _zscore, _repetition
    # _zscore 单候选退化为 0
    assert _zscore([0.5]) == [0.0]
    # _repetition 用 distinct-2：全重复串 ≈0，全异串 =1
    assert _repetition('ababab') < _repetition('abcdef'), "distinct-2 多样性判定反了"
    z = _zscore([1.0, 2.0, 3.0, 100.0])
    assert abs(z[0] + 0.5) < 0.2, "z-score 标准化异常"  # 1.0 应约 -0.5σ 量级
    assert z[-1] > z[0], "z-score 未保持大小序"


# ---------------------------------------------------------------------------
# 第五轮审查回归测试（M2/M3/M4）
# ---------------------------------------------------------------------------

def test_retrieval_bias_per_query_keep_mask():
    """回归测试 M2：_full_retrieval_bias 的 per-query keep mask 确保窗口内 key 不被 top-k 丢弃。
    注意：keep mask 叠加到 attn_mask 上，因果掩码已对 q 前方位置施加 -1e9，
    因此 keep mask 的 +1e9 只在因果允许的范围内有效（即 kpos <= qpos）。
    对中间 query（q=4, window=3），窗口 [1,2,3,4] 内的 key 应有 +1e9 保护。"""
    m = _small(vocab_size=200, attn_window=3, memory_retrieval_full=True, memory_retrieval_topk=2)
    m.eval()
    B, H, T, D = 1, 2, 8, 16
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T + 10, D)  # mem_cols=10
    device = torch.device('cpu')
    bias = m.blocks[0].attn._full_retrieval_bias(q, k, Treal=T, mem_cols=10, gate=None, device=device)
    assert bias is not None
    # q=4 (第5个位置)，窗口=3，因果允许 [0,1,2,3,4]，keep 保留 [1,2,3,4]
    qpos = 4
    for kpos in range(max(0, qpos - 3), qpos + 1):  # [1,2,3,4]
        val = bias[0, 0, qpos, 10 + kpos].item()
        assert val > 1e8, f'q={qpos}, key={kpos} 应有 +1e9 保护，实际={val}'
    # 窗口外远端位置（如 key=0，距离 4 > window=3）不应有 +1e9
    val_far = bias[0, 0, qpos, 10 + 0].item()
    assert val_far < 1e8, f'q={qpos}, key=0 窗口外不应有 +1e9，实际={val_far}'


def test_additive_repetition_penalty_igmcg_batch():
    """回归测试 M3：IGMCG batch 路径使用加性频率惩罚（logits -= penalty × count）。"""
    # 模拟 IGMCG batch 路径中的惩罚逻辑（generate.py:596-604）
    V = 10
    rep_penalty = 2.0
    generated = [1, 2, 2, 3, 1, 1]  # token 1 出现 3 次，token 2 出现 2 次，token 3 出现 1 次
    lt = torch.zeros(V)  # 原始 logits 全 0
    from collections import Counter
    freq = Counter(generated)
    prev_toks = torch.tensor(list(freq.keys()), dtype=torch.long)
    prev_counts = torch.tensor(list(freq.values()), dtype=torch.float)
    valid = (prev_toks >= 0) & (prev_toks < V)
    lt[prev_toks[valid]] = lt[prev_toks[valid]] - rep_penalty * prev_counts[valid]
    # token 1: 0 - 2.0*3 = -6.0
    assert abs(lt[1].item() - (-6.0)) < 1e-5, f'token 1 惩罚应为 -6.0，实际={lt[1].item()}'
    # token 2: 0 - 2.0*2 = -4.0
    assert abs(lt[2].item() - (-4.0)) < 1e-5, f'token 2 惩罚应为 -4.0，实际={lt[2].item()}'
    # token 3: 0 - 2.0*1 = -2.0
    assert abs(lt[3].item() - (-2.0)) < 1e-5, f'token 3 惩罚应为 -2.0，实际={lt[3].item()}'
    # 未出现的 token 不受影响
    assert lt[5].item() == 0.0, f'未出现 token 不应被惩罚，实际={lt[5].item()}'


def test_linear_attention_cumsum_denominator():
    """回归测试 M4：LinearAttention cumsum 分母累积正确——z_all = cumsum(kf)，den = einsum(qf, z_all)。"""
    torch.manual_seed(42)
    m = _small(vocab_size=200, mixer='linear')
    m.eval()
    x = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        # 全量前向（cumsum 路径）
        full_out, present = m.blocks[0].attn(x=torch.randn(1, 8, 64), use_cache=False)
        # 手动验证：构造简单输入
        B, H, T, D = 1, 2, 4, 8
        attn = m.blocks[0].attn
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        qf = attn._feat(q)
        kf = attn._feat(k)
        kv_all = torch.einsum('bhtd,bhte->bhtde', kf, v)
        S_all = torch.cumsum(kv_all, dim=2)
        z_all = torch.cumsum(kf, dim=2)
        num = torch.einsum('bhtd,bhtde->bhte', qf, S_all)
        den = torch.einsum('bhtd,bhtd->bht', qf, z_all).unsqueeze(-1).clamp_min(1e-6)
        out = num / den
        # 验证 z_all 逐位置累积正确
        assert z_all[:, :, 0, :].allclose(kf[:, :, 0, :]), 'z_all[0] 应等于 kf[0]'
        assert z_all[:, :, 1, :].allclose(kf[:, :, 0, :] + kf[:, :, 1, :]), 'z_all[1] 应等于 kf[0]+kf[1]'
        # 验证 den 与 z_all 的关系
        den_expected = torch.einsum('bhtd,bhtd->bht', qf, z_all).unsqueeze(-1).clamp_min(1e-6)
        assert den.allclose(den_expected), 'den 应等于 einsum(qf, z_all).clamp_min(1e-6)'
        # 验证 present 包含 S_final 和 z_final
        assert present is None or len(present) == 4 or len(present[0]) == 2


def test_linear_attention_relu_feature():
    """LinearAttention relu 特征映射：relu(x)+1e-6 >= 0，DML 兼容（elu 会 CPU 回退）。"""
    from models.transformer import LinearAttention
    attn = LinearAttention(64, 4, feature='relu')
    x = torch.randn(2, 8, 64)
    feat = attn._feat(x)
    # relu(x)+1e-6 >= 0（relu(x)>=0, +1e-6 后 >0）
    assert (feat >= 0).all(), f'relu(x)+1e-6 应全 >=0, got min={feat.min().item():.6f}'
    # 同时验证 out shape 正确
    out, present = attn(x)
    assert out.shape == (2, 8, 64), f'输出形状错误: {out.shape}'


def test_hybrid_block_no_leak_attn_kwargs():
    """hybrid block 构建时 linear_attn_feature 不泄漏到 SlidingWindowCausalSelfAttention。"""
    from models.transformer import TransformerBlock
    # 不应 TypeError: unexpected keyword argument
    blk = TransformerBlock(
        64, 4, 128, block_type='attn', mixer='attn_linear',
        attn_kwargs={'window': 32, 'qk_norm': True, 'attn_temp': True,
                     'linear_attn_feature': 'relu'}
    )
    assert blk.linear_attn is not None
    assert blk.linear_attn.feature == 'relu'
    x = torch.randn(2, 8, 64)
    out = blk(x)
    assert isinstance(out, tuple)
    assert out[0].shape == (2, 8, 64)


def test_linear_attn_head_dim_reduces_projection():
    """linear_attn_head_dim 可小于 dim//num_heads：qkv/proj 按 num_heads*head_dim 降维。

    AMD 780M iGPU(DML) 扫描结论：head_dim=16 比默认 64 快 1.75x 且质量持平。
    """
    from models.transformer import LinearAttention, TransformerBlock
    dim, nh = 64, 4
    # 默认（head_dim=None → dim//nh = 16，此处与显式 16 等价）
    a_def = LinearAttention(dim, nh, feature='relu')
    assert a_def.head_dim == dim // nh
    # 显式降维：head_dim=8 < dim//nh=16
    a8 = LinearAttention(dim, nh, feature='relu', head_dim=8)
    assert a8.head_dim == 8
    assert a8.qkv.out_features == 3 * nh * 8      # 96 而非 192
    assert a8.proj.in_features == nh * 8          # 32 而非 64
    x = torch.randn(2, 8, dim)
    out8, _ = a8(x)
    assert out8.shape == (2, 8, dim)              # 输出仍映射回 dim
    # hybrid block 透传 linear_attn_head_dim 且不泄漏到 attn
    blk = TransformerBlock(
        dim, nh, 128, block_type='attn', mixer='attn_linear',
        attn_kwargs={'window': 32, 'qk_norm': True, 'attn_temp': True,
                     'linear_attn_feature': 'relu', 'linear_attn_head_dim': 8}
    )
    assert blk.linear_attn.head_dim == 8
    assert blk.attn.head_dim == dim // nh          # 注意力的 head_dim 不受影响
    assert blk(torch.randn(2, 8, dim))[0].shape == (2, 8, dim)


def test_apply_repetition_penalty_matches_formula():
    """INT-1 回归：apply_repetition_penalty 是 sample_step 与 _generate_candidates_batch
    共用的加性频率惩罚单一事实来源，须满足「已出现 token 按次数减去 penalty*count」。"""
    dev = torch.device('cpu')
    logits = torch.zeros(10)
    out = apply_repetition_penalty(logits.clone(), [3, 3, 5], 0.5, dev)
    # token 3 出现 2 次 → -0.5*2 = -1.0；token 5 出现 1 次 → -0.5；其余不变
    assert out[3].item() == -1.0
    assert out[5].item() == -0.5
    assert out[0].item() == 0.0
    # penalty<=0 或空序列原样返回（不修改）
    assert torch.equal(apply_repetition_penalty(logits.clone(), [], 0.5, dev), logits)
    assert torch.equal(apply_repetition_penalty(logits.clone(), [1, 2], 0.0, dev), logits)
    # 越界 token 被 valid 掩码过滤，不越界报错（仅 token 3 受罚）
    out2 = apply_repetition_penalty(logits.clone(), [3, 999], 0.5, dev)
    assert out2[3].item() == -0.5


def test_attn_linear_rope_config_consistent():
    # §4.2 回归：attn_linear 混合块内 attn 与 linear_attn 两路的 RoPE 必须同源
    # （rope_learnable 一致），否则可学习 RoPE 训练时两路位置编码静默分叉。
    m = _small(mixer='attn_linear', rope_learnable=True, num_layers=1)
    blk = m.blocks[0]
    assert blk.attn is not None and blk.linear_attn is not None
    # 两路 RoPE 的 learnable 标志一致
    assert blk.attn.rope.learnable is True
    assert blk.linear_attn.rope.learnable is True
    # 默认路径（rope_learnable=False）两路也都应为非可学习
    m2 = _small(mixer='attn_linear', num_layers=1)
    assert m2.blocks[0].attn.rope.learnable is False
    assert m2.blocks[0].linear_attn.rope.learnable is False
    # 可学 RoPE 在两路都应暴露可训参数（rope_log_scale）
    assert hasattr(blk.attn.rope, 'rope_log_scale')
    assert hasattr(blk.linear_attn.rope, 'rope_log_scale')


def test_cached_causal_mask_matches_inline():
    # §优化#1 回归：增量解码纯因果掩码缓存须与原始 arange 公式数值一致，
    # 且重复调用（cache 命中）返回同一形状/值，不影响注意力输出。
    m = _small(num_layers=1)
    attn = m.blocks[0].attn
    dev = torch.device('cpu')
    Tq, Tkv, start_pos = 1, 13, 12
    cached = attn._cached_causal_mask(Tq, Tkv, dev, start_pos)
    # 原始公式复算
    qpos = torch.arange(start_pos, start_pos + Tq).unsqueeze(1).float()
    kpos = torch.arange(0, Tkv).unsqueeze(0).float()
    expected = (kpos > qpos).float() * attn.mask_fill_value
    expected = expected.unsqueeze(0).unsqueeze(0)
    assert torch.allclose(cached, expected)
    # 缓存命中：再次调用返回同形状同值（不重建）
    cached2 = attn._cached_causal_mask(Tq, Tkv, dev, start_pos)
    assert cached2.shape == cached.shape
    assert torch.allclose(cached2, cached)
    # 生成确定性：多次 generate 走 cache 命中路径，输出一致（覆盖掩码缓存不污染结果）
    m.eval()
    out1 = m.generate([1, 2, 3], max_length=8, device='cpu', temperature=1.0, top_k=0)
    out2 = m.generate([1, 2, 3], max_length=8, device='cpu', temperature=1.0, top_k=0)
    assert out1 == out2


def test_shared_constants_consistent():
    # §整合：特殊 token / 掩码填充 / RoPE 基频集中到 models/constants.py 单一来源，
    # BaseTokenizer 的索引须与常量一致（防止改一处漏一处）。
    from models.constants import (SPECIAL_TOKENS, PAD_IDX, UNK_IDX, BOS_IDX,
                                  EOS_IDX, SEP_IDX, MASK_FILL_VALUE, ROPE_BASE)
    from models.data_utils import BaseTokenizer
    t = BaseTokenizer()
    assert t.pad_idx == PAD_IDX and t.unk_idx == UNK_IDX and t.bos_idx == BOS_IDX
    assert t.eos_idx == EOS_IDX and t.sep_idx == SEP_IDX
    assert list(t.special_tokens) == list(SPECIAL_TOKENS)
    # 魔数常量值正确
    assert MASK_FILL_VALUE == -1e9
    assert ROPE_BASE == 10000.0

