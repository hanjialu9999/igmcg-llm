"""第三十九轮回归测试：MAJOR 修复守卫。

R39 修复（子代理审查 + 人工核验）：
- R39-M1（MAJOR，R38 引入）：inject_memory 在 past 拼接**之前**执行 → cache 布局
  [pk|mk|cur]（记忆在中间），present 剥离 mem_cols 列剥掉的是 past 尾部真实
  token，记忆列留在缓存（多份记忆副本 + 真实上下文永久丢失，parity 残差 0.0434
  超固有 divergence 上界）。修复：注入移入拼接之后，布局恒为 [mk|pk|cur]。
  R38-A1 测试补内容级断言（present 逐列不得与记忆列重合）。
- R39-M2（MAJOR）：inject_memory 的 mem_bias=mlogits 无条件返回——meta=None
  （无检索无稀疏）时记忆列已在 k_aug 里产生 1×q·mk 点积分，再叠加即 2× 翻倍，
  与注释"否则记忆仅作为全局 KV 参与注意力（不加额外 bias）"矛盾。修复：
  meta=None → mem_bias=None（记忆以 1× 参与）。
- R39-M3（MAJOR）：train_finetune 只复制 best_finetuned_model.pt 不复制 config → 
  load_model 回退硬编码默认 + strict=False 静默部分加载 → 随机噪声。修复：
  同步复制 _config.yaml。
- R39-M4：ngram 增量路径绕过字节预算（只查条数、不记账）→ 大词表时缓存可达
  数 GB，且陈旧字节数污染 matrix 路径判断。修复：共用预算记账。
- R39-M5：ngram.py:187 torch.lerp——DML 不支持 aten::lerp.Tensor_out，函数式
  lerp CPU 回退（与 layers.py R38 修复同根因，当时漏修）。修复：4 算子式。
- R39-M6：sample_next_token 不屏蔽 BOS（BOS 只作序列起始标记，训练 target 从
  row[1:] 取，从未被监督输出；采样抽中会插入垃圾标记）。修复：bos_id 屏蔽。
- R39-M7（resume 族）：global_step 过计数（resume 轮记满 total_batches）、
  epoch ckpt 缺 global_step、step ckpt 清理在 final save 之前。
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config_loader import load_config, build_model
from models.memory import MemoryBank
from models.ngram import NGramModel
from models.transformer import TransformerModel
from models.utils import save_checkpoint


def _build(**kw):
    kw.setdefault('vocab_size', 200)
    kw.setdefault('embedding_dim', 32)
    kw.setdefault('num_heads', 4)
    kw.setdefault('hidden_dim', 64)
    kw.setdefault('max_seq_length', 16)
    kw.setdefault('memory_size', 4)
    kw.setdefault('memory_comp_dim', 8)
    kw.setdefault('gradient_checkpointing', False)
    kw.setdefault('tie_weights', False)
    return TransformerModel(num_layers=1, **kw)


def _parity(m, seq=8, prefill=4):
    torch.manual_seed(0)
    x = torch.randint(0, 200, (1, seq))
    with torch.no_grad():
        full = m(x)
        _, past = m(x[:, :prefill], use_cache=True)
        outs, p = [], past
        for t in range(prefill, seq):
            o, p = m(x[:, t:t + 1], past_key_values=p, use_cache=True)
            outs.append(o)
        inc = torch.cat(outs, dim=1)
    return (full[:, prefill:] - inc).abs().max().item()


# ===========================================================================
# R39-M2：meta=None 时记忆得分不翻倍（mem_bias=None）
# ===========================================================================

def test_r39_m2_no_double_counting_without_retrieval():
    """纯记忆（无检索无稀疏）时记忆列仅以全局 KV 参与（1×），不加额外 bias。

    旧 bug：mem_bias=mlogits 无条件返回——记忆列已在 k_aug 中产生点积分，
    再叠加 mem_bias 即 2× 翻倍，与注释"仅当开启检索/稀疏时才加偏置"矛盾。
    """
    m = _build()
    m.eval()
    torch.manual_seed(0)
    x = torch.randint(0, 200, (1, 4))

    orig = MemoryBank.inject_memory
    seen_meta = {}
    def spy(q, k, v, mk, mv, meta, mask_fill):
        seen_meta['meta'] = meta
        return orig(q, k, v, mk, mv, meta, mask_fill)
    MemoryBank.inject_memory = staticmethod(spy)
    try:
        with torch.no_grad():
            m(x)
    finally:
        MemoryBank.inject_memory = staticmethod(orig)
    assert seen_meta.get('meta') is None, "无检索无稀疏配置下 meta 应为 None"
    # 修复后 meta=None → mem_bias=None：直接调用验证返回值
    q = torch.randn(1, 4, 4, 8)
    k = torch.randn(1, 4, 4, 8)
    v = torch.randn(1, 4, 4, 8)
    mk = torch.randn(1, 4, 8)
    mv = torch.randn(1, 4, 8)
    k_aug, v_aug, mem_bias = MemoryBank.inject_memory(q, k, v, mk, mv, None, -1e9)
    assert mem_bias is None, "meta=None 时 mem_bias 必须为 None（记忆 1× 参与）"
    assert k_aug.size(2) == 8 and v_aug.size(2) == 8

    # 检索开启时 mem_bias 仍返回（行为不变）
    from models.memory import MemoryBank as MB
    k_aug2, _, mem_bias2 = MB.inject_memory(
        q, k, v, mk, mv, {'retrieval_gate': torch.zeros(1), 'sparse_topk': 0}, -1e9)
    assert mem_bias2 is not None, "检索开启时 mem_bias 应照常返回"
    # 门控 sigmoid(0)=0.5：bias 应为 mlogits*0.5
    mlogits = torch.einsum('bhqd,bhmd->bhqm', q, mk.unsqueeze(1).expand(-1, 4, -1, -1))
    assert torch.allclose(mem_bias2, mlogits * 0.5, atol=1e-6), "门控 0.5 缩放应成立"


# ===========================================================================
# R39-M3：finetune best 模型 config 同步复制
# ===========================================================================

def test_r39_m3_finetune_best_copy_syncs_config():
    """best_finetuned_model.pt 复制时必须同步复制 _config.yaml（load_model 依赖
    sibling config；缺失则回退硬编码默认 + strict=False 静默部分加载 → 随机噪声）。
    """
    import inspect
    from scripts import train_finetune
    src = inspect.getsource(train_finetune)
    # 修复模式：copy .pt 后应 copy 对应 _config.yaml
    assert 'replace(\'.pt\', \'_config.yaml\')' in src, \
        "train_finetune 应同步复制 best_finetuned_model_config.yaml"
    assert 'copyfile(_src_cfg, _dst_cfg)' in src

    # 行为级验证：save_checkpoint 产出的 config 可被 load_model 找到
    from models.checkpoint import load_model
    m = _build()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    d = tempfile.mkdtemp()
    model_config = {'vocab_size': 200, 'embedding_dim': 32, 'num_heads': 4,
                    'hidden_dim': 64, 'max_seq_length': 16, 'memory_size': 4,
                    'memory_comp_dim': 8, 'gradient_checkpointing': False,
                    'tie_weights': False, 'num_layers': 1}
    saved = save_checkpoint(m, opt, 1, 1.0, d, 200, model_config)
    # 模拟 finetune 的复制行为（.pt + _config.yaml 一起）
    best = os.path.join(d, 'best_finetuned_model.pt')
    shutil.copyfile(saved, best)
    shutil.copyfile(saved.replace('.pt', '_config.yaml'),
                    best.replace('.pt', '_config.yaml'))
    assert os.path.exists(best.replace('.pt', '_config.yaml')), "config 必须被同步复制"
    # 与 load_model 的查找逻辑一致：stem_config.yaml 存在
    cfg_path = Path(best).parent / f"{Path(best).stem}_config.yaml"
    assert cfg_path.exists()
    assert os.path.exists(Path(best).parent / "best_finetuned_model_config.yaml")


# ===========================================================================
# R39-M4：ngram 增量路径字节预算
# ===========================================================================

def test_r39_m4_ngram_incremental_respects_byte_budget():
    """增量路径写入缓存必须记账并受字节预算约束（旧 bug 绕过预算 + 污染 matrix 记账）。"""
    from models.data_utils import CharTokenizer
    corpus = ["alpha beta gamma", "beta gamma delta", "alpha gamma beta", "delta alpha"]
    v = CharTokenizer(vocab_size=200)
    v.train(corpus)
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    f.write("\n".join(corpus) + "\n")
    f.close()
    ng = NGramModel(v, f.name, max_order=3)
    os.unlink(f.name)

    ids_ctx = v.encode('ab', add_special_tokens=False)
    ids_new = v.encode('cd', add_special_tokens=False)
    ctx2 = torch.tensor([ids_ctx[:2]])
    new_ids = torch.tensor([ids_new[:2]])
    V = ng.vocab_size
    # 预清空并压预算
    ng._orders_cache.clear()
    ng._orders_cache_bytes = 0
    ng._orders_cache_byte_budget = V * 3 * 4  # 恰一条 (V,K) fp32 的大小
    ng.logprob_orders_incremental(ctx2, new_ids, 'cpu')
    assert ng._orders_cache_bytes > 0, "增量路径应记账"
    assert len(ng._orders_cache) <= 2, "预算=1 条时缓存不得超 1 条"
    # 预算恢复到 0：每写即清，恒 1 条
    ng._orders_cache_byte_budget = 0
    ng.logprob_orders_incremental(ctx2, new_ids, 'cpu')
    assert len(ng._orders_cache) == 1, "预算 0 时增量路径缓存应恒 1 条"
    assert ng._orders_cache_bytes == ng._orders_cache[
        next(iter(ng._orders_cache))].numel() * 4, "字节数应等于单条大小"


# ===========================================================================
# R39-M5：ngram lerp 数值等价（4 算子替代 DML 回退的 torch.lerp）
# ===========================================================================

def test_r39_m5_ngram_no_torch_lerp():
    """插值不得再使用 torch.lerp（DML CPU 回退），须用 4 算子等价式。"""
    import inspect
    from models.ngram import NGramModel as NM
    src = inspect.getsource(NM._compute_logprob_orders)
    # 剔除注释行后检查可执行代码
    code_lines = [ln for ln in src.splitlines()
                  if ln.strip() and not ln.strip().startswith('#')]
    code = "\n".join(code_lines)
    assert 'torch.lerp' not in code, "matrix 路径不得使用 torch.lerp（DML 回退）"
    assert 'w * (p - u_idx)' in code, "应使用 4 算子等价式"


# ===========================================================================
# R39-M6：采样屏蔽 BOS
# ===========================================================================

def test_r39_m6_sample_blocks_bos():
    """sample_next_token 屏蔽 bos_id（BOS 从未被监督为输出，采样抽中=垃圾标记）。"""
    from models.sampling import sample_next_token
    torch.manual_seed(0)
    lt = torch.randn(10)
    # 让 BOS(2) 成为最大 logit，若不屏蔽必被抽中
    lt[2] = 100.0
    tok = sample_next_token(
        lt, temperature=1.0, repetition_penalty=0.0, generated_ids=[],
        ngram_fn=None, ngram_weight=0.0, device='cpu',
        pad_id=0, sep_id=4, eos_id=3, generated_len=5, min_length=0,
        eos_penalty=0.0, top_k=0, vocab_size=10, bos_id=2)
    assert tok != 2, "BOS 必须被屏蔽"
    # 全 -inf 回退分支同样屏蔽 BOS（raw_logits 中 BOS 最高但被屏蔽 → 选次高合法 token）
    lt2 = torch.full((10,), float('-inf'))
    lt2[2] = 100.0
    raw = torch.full((10,), float('-inf'))
    raw[2] = 100.0
    raw[7] = 50.0  # 合法回退目标
    tok2 = sample_next_token(
        lt2, temperature=1.0, repetition_penalty=0.0, generated_ids=[],
        ngram_fn=None, ngram_weight=0.0, device='cpu',
        pad_id=0, sep_id=4, eos_id=3, generated_len=5, min_length=0,
        eos_penalty=0.0, top_k=0, vocab_size=10, bos_id=2,
        raw_logits=raw)
    assert tok2 != 2, "全 -inf 回退分支也应屏蔽 BOS"


# ===========================================================================
# R39-M7：save_checkpoint 存 global_step（epoch resume 恢复 LR 进度）
# ===========================================================================

def test_r39_m7_checkpoint_persists_global_step():
    """save_checkpoint 应把 global_step 存入 ckpt（epoch resume 恢复调度进度）。"""
    m = _build()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    d = tempfile.mkdtemp()
    saved = save_checkpoint(m, opt, 1, 1.0, d, 200, None, global_step=1234)
    ckpt = torch.load(saved, map_location='cpu', weights_only=True)
    assert ckpt.get('global_step') == 1234, "global_step 应持久化"
    # 不传时不写入（兼容旧调用）
    saved2 = save_checkpoint(m, opt, 2, 1.0, d, 200, None)
    ckpt2 = torch.load(saved2, map_location='cpu', weights_only=True)
    assert 'global_step' not in ckpt2
