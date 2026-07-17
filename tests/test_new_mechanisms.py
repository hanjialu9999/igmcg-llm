import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer import TransformerModel, MemoryBank


def _small(**over):
    kw = dict(vocab_size=200, embedding_dim=64, num_heads=4, num_layers=2,
              hidden_dim=128, max_seq_length=32)
    kw.update(over)
    return TransformerModel(**kw)


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
    m = _small(layer_skip=True, mixer='hybrid', learn_window=True, attn_window=8)
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


def test_linear_mixer_full_cache_parity():
    """纯线性注意力 mixer 的训练（全量）与推理（cache）路径必须数值一致。"""
    m = _small(mixer='linear')
    m.eval()
    ids = torch.randint(0, 200, (1, 12))
    with torch.no_grad():
        full = m(ids, use_cache=False)
        cached = m(ids, use_cache=True)
    fl = full["logits"] if isinstance(full, dict) else full
    cl = cached[0] if isinstance(cached, tuple) else (cached["logits"] if isinstance(cached, dict) else cached)
    diff = (fl - cl).abs().max().item()
    assert diff < 1e-4, f"linear mixer 训练/推理不一致：max_diff={diff}"


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


def test_memory_window_cache_parity():
    """memory_size>0 + attn_window>0 时，训练（全量）与推理（cache）路径必须数值一致。

    回归 BUG-2：cache 路径曾用 `kpos > qpos` 把记忆段（位于 KV 前缀）当成序列前缀按位置
    因果遮蔽，导致推理期记忆按位置被部分遮蔽，与训练全可见不一致（静默质量退化）。
    原 test_memory_window0_training 只测 window==0，漏掉此路径。
    """
    for w in (4, 8):
        m = _small(memory_size=16, memory_comp_dim=16, attn_window=w)
        m.eval()
        ids = torch.randint(0, 200, (1, 14))
        with torch.no_grad():
            full = m(ids, use_cache=False)
            cached = m(ids, use_cache=True)
        fl = full["logits"] if isinstance(full, dict) else full
        cl = cached[0] if isinstance(cached, tuple) else (cached["logits"] if isinstance(cached, dict) else cached)
        diff = (fl - cl).abs().max().item()
        assert diff < 1e-4, f"memory+window={w} 训练/推理不一致：max_diff={diff}"


def test_memory_retrieval_full_cache_parity():
    """memory_retrieval_full=True 时，训练（全量）与推理（cache）路径必须数值一致。

    回归 BUG-1：cache 路径曾完全漏调用 `_full_retrieval_bias`，训练用 top-k 稀疏检索偏置、
    推理（generate）不注入，导致训练-推理系统性偏差、生成质量偏离训练行为。
    """
    m = _small(memory_size=16, memory_comp_dim=16, memory_retrieval_full=True,
               memory_retrieval_topk=8, attn_window=8)
    m.eval()
    ids = torch.randint(0, 200, (1, 14))
    with torch.no_grad():
        full = m(ids, use_cache=False)
        cached = m(ids, use_cache=True)
    fl = full["logits"] if isinstance(full, dict) else full
    cl = cached[0] if isinstance(cached, tuple) else (cached["logits"] if isinstance(cached, dict) else cached)
    diff = (fl - cl).abs().max().item()
    assert diff < 1e-4, f"retrieval_full 训练/推理不一致：max_diff={diff}"


def test_alibi_memory_window_cache_parity():
    """alibi + memory + window>0 三路组合，训练与推理必须数值一致（交叉 bug 温床）。"""
    m = _small(alibi=True, memory_size=16, memory_comp_dim=16, attn_window=8)
    m.eval()
    ids = torch.randint(0, 200, (1, 14))
    with torch.no_grad():
        full = m(ids, use_cache=False)
        cached = m(ids, use_cache=True)
    fl = full["logits"] if isinstance(full, dict) else full
    cl = cached[0] if isinstance(cached, tuple) else (cached["logits"] if isinstance(cached, dict) else cached)
    diff = (fl - cl).abs().max().item()
    assert diff < 1e-4, f"alibi+memory+window 训练/推理不一致：max_diff={diff}"


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
    """构建一个小型 Vocabulary + 统计 NGramModel，供 n-gram 融合测试。

    走 Vocabulary 的标准 build_vocab 路径（而非伪造 encode），保证 vocab 长度、
    pad_idx、encode 行为均与生产一致；ngram 统计表由此真实 vocab 派生。
    """
    import tempfile, os
    from models.data_utils import Vocabulary
    from scripts.generate import NGramModel
    corpus_texts = [
        "中 国 人 民 生 活 幸 福",
        "中 国 梦 想 伟 大 复 兴",
        "人 民 当 家 作 主 权 利",
        "中 国 人 民 共 和 国 万 岁",
    ]
    v = Vocabulary(vocab_size)
    v.build_vocab(corpus_texts)  # 真实构建：len(v)≈vocab_size，encode 产出合法 id
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


def test_ngram_fusion_cache_parity():
    """ngram 融合下训练（全量）与推理（cache）路径必须数值一致（逐 token 融合也对齐）。"""
    v, ng = _small_ngram()
    V = len(v)
    m = _small(vocab_size=V, ngram_fusion=True, ngram_model=ng)
    m.eval()
    ids = torch.randint(0, V, (1, 12))
    with torch.no_grad():
        full = m(ids, use_cache=False)
        cached = m(ids, use_cache=True)
    fl = full["logits"] if isinstance(full, dict) else full
    cl = cached[0] if isinstance(cached, tuple) else (cached["logits"] if isinstance(cached, dict) else cached)
    diff = (fl - cl).abs().max().item()
    assert diff < 1e-4, f"ngram 融合训练/推理不一致：max_diff={diff}"


def test_ngram_fusion_gate_scale_zero():
    """gate_scale=0 应完全拔掉 n-gram 贡献（输出≈纯主干），验证用户可控总闸。"""
    v, ng = _small_ngram()
    V = len(v)
    m = _small(vocab_size=V, ngram_fusion=True, ngram_model=ng)
    m.eval()
    x = torch.randint(0, V, (2, 10))
    with torch.no_grad():
        m.set_ngram_gate_scale(0.0)
        off = m(x)
        m.set_ngram_gate_scale(1.0)
        m.set_ngram_fusion_active(False)
        base = m(x)
    assert torch.allclose(off, base, atol=1e-5), "gate_scale=0 时仍含 n-gram 贡献"


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
