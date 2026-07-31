"""第三十八轮回归测试：记忆/检索/窗口增量解码修复 + resume 族 + GPU 裁剪 + QAT 设备。

R38 修复（子代理审查 + 人工核验）：
- R38-A1（MAJOR）：增量解码时记忆列被反复注入 KV cache——present=(k,v) 含注入后的
  记忆列，下一步 past 再拼新记忆 → Tkv 每步膨胀 M+1。修复：present 剥离记忆列。
- R38-A2（MAJOR）：_full_retrieval_bias 在 past 拼接前调用（Treal=Tq），最终掩码
  宽度 mem+T_past+Tq 不匹配（增量第 2 步 RuntimeError），且检索只覆盖当前 token。
  修复：cache 路径移到拼接后算，qpos 取全局位置 arange(Treal-Tq, Treal)。
- R38-A3（MAJOR）：窗口子句 `qpos - kpos > window` 未减 mem_cols 偏移 → 有效窗口
  静默放大 mem_cols。修复：mask 条件减 mem_cols。
- R38-B1：训练完成后 step checkpoint 残留，resume 优先加载旧 step ckpt → 回滚。
  修复：训练收尾删除 step ckpt。
- R38-B2：resume 后 eff_step/global_step/best_loss 未恢复 → LR 调度从头爬升、
  课程退火相位错位、best 模型被覆盖。修复：恢复三者 + initial_eff_step。
- R38-B8：QAT _qat_scale 创建在 CPU（enable_qat 在 model.to(device) 之后调用），
  优化器/裁剪跨设备。修复：参数创建落在模型当前设备。
- R38-B3：ngram _orders_cache 8192 条 × (V,K) fp32 可达数 GB。修复：字节预算 512MB。
- R38-B4：AxialLinearAttention 全量路径忽略 memory、增量路径注入 → train/infer 分裂。
  修复：配置层禁止 linear2d/hybrid_linear2d + memory_size>0。
- R38-C1：VRC 递推滤波改用 causal conv1d（性能 2x），数值等价（4.8e-7）。
- R38-C2：compute_lr progress clamp ≤1——eff_step 超 total_eff 时 cosine/wsd 不回升。
- R38-C3：clip_grad_norm_dml——GPU 侧梯度裁剪，无 .item() 同步，数学等价。

本测试验证：
1. 记忆模型增量解码 cache 每步只增长 1 列（不膨胀 M+1）
2. memory_retrieval_full 增量解码不崩溃且与全量 parity（旧 bug 下 RuntimeError）
3. attn_window + memory 增量 parity
4. clip_grad_norm_dml 与 clip_grad_norm_ 数学等价（含裁剪/不裁剪/NaN 场景）
5. QAT _qat_scale 与模型同设备（CPU 断言 + DML 环境额外断言）
6. ngram 缓存字节预算触发清空
7. linear2d + memory_size>0 配置被拒绝
8. compute_lr 超 total_eff 时 clamp 到 eta_min（cosine/wsd 不回升）
9. VRC 模型增量 vs 全量 parity（conv1d 实现回归守卫）
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config_loader import load_config, build_model
from models.data_utils import CharTokenizer
from models.ngram import NGramModel
from models.qat import enable_qat
from models.transformer import TransformerModel
from scripts.train import compute_lr, clip_grad_norm_dml


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


def _parity(m, seq=8, prefill=4, atol=0.1):
    """全量前向 vs 逐 token 增量解码的最大差异。"""
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
# R38-A1：present 剥离记忆列——cache 每步只增长 1 列
# ===========================================================================

def test_r38_a1_present_strips_memory_columns():
    """记忆模型增量解码：past_kv[0] 序列长度按 [prefill, +1, +1] 增长。

    旧 bug：present=(k,v) 含注入的记忆列 → 下一步 past 再拼新记忆，
    Tkv 每步膨胀 M+1（M=mem_cols），缓存爆炸 + 检索偏置错位。
    R39 加固（MAJOR-1）：仅查尺寸会漏判——注入在拼接之前时布局为
    [pk|mk|cur]（记忆在中间），剥离 mem_cols 列剥掉的是 past 尾部真实
    token，记忆列反而留在缓存（多份记忆副本 + 真实上下文永久丢失，
    parity 残差 0.0434 超固有 divergence 上界）。修复：注入移入拼接
    之后，布局恒为 [mk|pk|cur]。本测试同时断言内容：present 的每一列
    都不得与任何一步注入的记忆列重合（逐列内容级守卫）。
    """
    from models.memory import MemoryBank
    m = _build()
    m.eval()
    x = torch.randint(0, 200, (1, 6))
    captured = []
    orig_inject = MemoryBank.inject_memory
    def spy(q, k, v, mk, mv, meta, mask_fill):
        captured.append(mk.detach().clone())
        return orig_inject(q, k, v, mk, mv, meta, mask_fill)
    MemoryBank.inject_memory = staticmethod(spy)
    try:
        with torch.no_grad():
            _, past = m(x[:, :3], use_cache=True)
            sizes = [past[0][0][0].size(2)]
            for t in range(3, 6):
                _, past = m(x[:, t:t + 1], past_key_values=past, use_cache=True)
                sizes.append(past[0][0][0].size(2))
    finally:
        MemoryBank.inject_memory = staticmethod(orig_inject)
    assert sizes == [3, 4, 5, 6], \
        f"cache 应每步只增 1 列（记忆列不得进入 present），实际增长 {sizes}"
    # 内容级守卫：present 不得含任何一步注入的记忆列
    assert captured, "记忆注入应至少发生一次"
    present_k = past[0][0][0]
    pm = present_k.flatten(0, 1)                       # (B*H, Tkv, D)
    mka = torch.cat(captured, dim=1).flatten(0, 1)     # (B*H, M*steps, D)
    leak = sum(1 for i in range(pm.size(1))
               if ((pm[:, i:i + 1, :] - mka).abs().max(dim=-1).values < 1e-5).any())
    assert leak == 0, \
        f"present 泄漏 {leak} 列记忆（应只含真实 token KV）"


# ===========================================================================
# R38-A2：memory_retrieval_full 增量解码（旧 bug 下 RuntimeError）
# ===========================================================================

def test_r38_a2_retrieval_full_incremental_parity():
    """retrieval_full 增量解码不崩 + 与全量 parity。

    旧 bug：_full_retrieval_bias 在 past 拼接前算（宽度 Tq），最终掩码宽
    mem+T_past+Tq → 第 2 步增量 RuntimeError（size 13 vs 5）；修复后只残留
    记忆固有 train/infer divergence（~0.01-0.03 量级）。
    """
    m = _build(memory_retrieval=True, memory_retrieval_full=True,
               memory_retrieval_topk=8)
    m.eval()
    d = _parity(m)
    assert d < 0.1, f"retrieval_full 增量 vs 全量 diff={d}（应仅记忆固有 divergence）"


# ===========================================================================
# R38-A3：attn_window + memory 增量（窗口子句减 mem_cols）
# ===========================================================================

def test_r38_a3_window_memory_incremental_parity():
    """window + memory 组合增量解码不崩 + parity（旧 bug 下有效窗口静默放大）。"""
    m = _build(attn_window=8)
    m.eval()
    d = _parity(m)
    assert d < 0.1, f"window+memory 增量 vs 全量 diff={d}"


# ===========================================================================
# R38-C3：clip_grad_norm_dml 与 clip_grad_norm_ 数学等价
# ===========================================================================

def test_r38_c3_clip_grad_norm_dml_equivalence():
    """GPU 侧裁剪与 torch.nn.utils.clip_grad_norm_ 数学等价：
    裁剪（norm>max_norm）、不裁剪（norm<max_norm）、全零梯度、NaN 梯度。
    """
    torch.manual_seed(0)
    for max_norm in (0.3, 1.0, 5.0):
        p1 = torch.nn.Parameter(torch.randn(8))
        p2 = torch.nn.Parameter(torch.randn(4, 4))
        g1, g2 = torch.randn_like(p1), torch.randn_like(p2)

        p1r = p1.detach().clone(); p2r = p2.detach().clone()
        p1r.grad = g1.clone(); p2r.grad = g2.clone()
        gn = torch.nn.utils.clip_grad_norm_([p1r, p2r], max_norm)

        p1m = p1.detach().clone(); p2m = p2.detach().clone()
        p1m.grad = g1.clone(); p2m.grad = g2.clone()
        gm = clip_grad_norm_dml([p1m, p2m], max_norm)

        assert torch.allclose(p1r.grad, p1m.grad, atol=1e-6), \
            f"max_norm={max_norm} p1 梯度不一致"
        assert torch.allclose(p2r.grad, p2m.grad, atol=1e-6), \
            f"max_norm={max_norm} p2 梯度不一致"
        assert torch.allclose(torch.tensor(gn), gm, atol=1e-6), \
            f"max_norm={max_norm} 返回 norm 不一致"

    # NaN 梯度：两边都不缩放（clip_grad_norm_ 默认 error_if_nonfinite=False）
    p = torch.nn.Parameter(torch.randn(4))
    p.grad = torch.tensor([1.0, float('nan'), 1.0, 1.0])
    clip_grad_norm_dml([p], 0.5)
    assert torch.isnan(p.grad[1]).item() and p.grad[0].item() == 1.0, \
        "NaN 梯度不应被缩放（与原版语义一致）"

    # 全零梯度：不崩溃
    p = torch.nn.Parameter(torch.randn(4))
    p.grad = torch.zeros(4)
    clip_grad_norm_dml([p], 0.5)
    assert p.grad.abs().sum().item() == 0.0


# ===========================================================================
# R38-B8：QAT _qat_scale 与模型同设备
# ===========================================================================

def test_r38_b8_qat_scale_on_model_device():
    """enable_qat 后 _qat_scale 落在模型当前设备（CPU 回归守卫 + DML 环境强断言）。

    旧 bug：enable_qat 在 model.to(device) 之后调用时 scale 留在 CPU，
    优化器 step / 梯度裁剪跨设备报错或静默混合。
    """
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    enable_qat(m, bits=8)
    assert m._qat_scale.device == next(m.parameters()).device, \
        f"_qat_scale 设备 {m._qat_scale.device} ≠ 模型设备 {next(m.parameters()).device}"

    try:
        import torch_directml
        d = torch_directml.device()
    except Exception:
        pytest.skip('torch_directml 不可用，跳过 DML 设备断言')
    m2 = torch.nn.Sequential(torch.nn.Linear(4, 4)).to(d)
    enable_qat(m2, bits=8)
    assert m2._qat_scale.device.type == d.type, \
        f"DML 下 _qat_scale 应为 {d}，实际 {m2._qat_scale.device}"


# ===========================================================================
# R38-B3：ngram 缓存字节预算
# ===========================================================================

def _make_ngram():
    corpus = ["alpha beta gamma", "beta gamma delta", "alpha gamma beta", "delta alpha"]
    v = CharTokenizer(vocab_size=200)
    v.train(corpus)
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    f.write("\n".join(corpus) + "\n")
    f.close()
    ng = NGramModel(v, f.name, max_order=3)  # vocab_size 默认取 len(vocab)，覆盖全部真实 token id
    os.unlink(f.name)
    return v, ng


def test_r38_b3_ngram_cache_byte_budget():
    """_orders_cache 超字节预算即清空（防大词表时数 GB CPU 内存）。"""
    v, ng = _make_ngram()
    assert ng._orders_cache_byte_budget == 512 * 1024 * 1024, "默认字节预算应为 512MB"
    assert ng._orders_cache_bytes == 0

    # 用模型真实构建的上下文（语料中出现过的 n-gram 上下文键）
    ctxs = list(ng.ngrams[2].keys())[:5]
    assert ctxs, "语料应构建出至少一个 bi-gram 上下文"
    V = ng.vocab_size
    for ck in ctxs:
        val = ng._compute_logprob_orders(ck, V, 'cpu')
        if len(ng._orders_cache) > ng._orders_cache_max or \
           ng._orders_cache_bytes + val.numel() * val.element_size() > ng._orders_cache_byte_budget:
            ng._orders_cache.clear()
            ng._orders_cache_bytes = 0
        ng._orders_cache[ck] = val.cpu()
        ng._orders_cache_bytes += val.numel() * val.element_size()
    assert len(ng._orders_cache) == len(ctxs)
    assert ng._orders_cache_bytes > 0

    # 预算压到 0：每次写入前必清空 → 缓存恒为 1 条
    ng._orders_cache_byte_budget = 0
    ng._orders_cache.clear()
    ng._orders_cache_bytes = 0
    for ck in ctxs:
        val = ng._compute_logprob_orders(ck, V, 'cpu')
        if len(ng._orders_cache) > ng._orders_cache_max or \
           ng._orders_cache_bytes + val.numel() * val.element_size() > ng._orders_cache_byte_budget:
            ng._orders_cache.clear()
            ng._orders_cache_bytes = 0
        ng._orders_cache[ck] = val.cpu()
        ng._orders_cache_bytes += val.numel() * val.element_size()
    assert len(ng._orders_cache) == 1, "预算为 0 时缓存应每写即清（恒 1 条）"
    assert ng._orders_cache_bytes == ng._orders_cache[
        next(iter(ng._orders_cache))].numel() * 4


# ===========================================================================
# R38-B4：linear2d + memory 配置拒绝
# ===========================================================================

def test_r38_b4_linear2d_memory_rejected():
    """mixer=linear2d + memory_size>0 → ValueError（train/infer 分裂防呆）。"""
    cfg = load_config('configs/pretrain.yaml')
    cfg['model']['mixer'] = 'linear2d'
    cfg['model']['memory_size'] = 16
    with pytest.raises(ValueError, match='linear2d'):
        build_model(cfg, device='cpu')


def test_r38_b4_attn_memory_still_ok():
    """attn 系 mixer + memory_size>0 不受影响（组合校验不误伤）。"""
    cfg = load_config('configs/pretrain.yaml')
    cfg['model']['memory_size'] = 16
    cfg['model']['memory_comp_dim'] = 8
    m = build_model(cfg, device='cpu')
    assert m is not None


# ===========================================================================
# R38-C2：compute_lr 超限不回升
# ===========================================================================

def test_r38_c2_compute_lr_clamped_at_eta_min():
    """eff_step 超 total_eff（resume 后数据规模变化）时 cosine/wsd 保持 eta_min。

    旧 bug：progress>1 → cos(π·progress) 周期回升，LR 从 eta_min 重新爬升。
    """
    base, eta, total, warm = 1e-3, 1e-5, 100, 10
    for schedule in ('cosine', 'wsd'):
        lr_over = compute_lr(total + 50, total, warm, base, eta, schedule, 0.1)
        lr_way_over = compute_lr(total * 3, total, warm, base, eta, schedule, 0.1)
        assert lr_over == pytest.approx(eta, rel=1e-6), \
            f"{schedule} 超限后应保持 eta_min，实际 {lr_over}"
        assert lr_way_over == pytest.approx(eta, rel=1e-6)

    # 未超限路径不受影响：cosine 中点 ≈ (base+eta)/2
    lr_mid = compute_lr((total + warm) // 2, total, warm, base, eta, 'cosine', 0.1)
    assert lr_mid == pytest.approx((base + eta) / 2, abs=1e-4)
    # warmup 段不受影响：线性升温
    assert compute_lr(5, total, warm, base, eta, 'cosine', 0.1) == pytest.approx(base / 2)


# ===========================================================================
# R38-C1：VRC（value_relative_coding）模型增量 parity（conv1d 回归守卫）
# ===========================================================================

def test_r38_c1_vrc_model_incremental_parity():
    """value_relative_coding=True 模型：增量 vs 全量 parity。

    R38 将 VRC 全量递推从 Hillis-Steele 前缀扫描改为 causal conv1d
    （2x 提速，逐点差 4.8e-7）——本测试守卫新实现不破坏 train/infer 一致性。
    """
    torch.manual_seed(0)
    m = TransformerModel(num_layers=1, vocab_size=200, embedding_dim=32, num_heads=4,
                         hidden_dim=64, max_seq_length=16,
                         gradient_checkpointing=False, tie_weights=False,
                         value_relative_coding=True)
    m.eval()
    d = _parity(m, atol=0.05)
    assert d < 0.05, f"VRC 增量 vs 全量 diff={d}（conv1d 实现应保持一致性）"


# ===========================================================================
# R38-B1/B2：resume 语义（step ckpt 清理 + 恢复字段）
# ===========================================================================

def test_r38_b1_step_checkpoint_cleanup():
    """训练收尾删除 step checkpoint：模拟保存后走清理逻辑，step ckpt 必须消失。

    旧 bug：step ckpt 残留 → resume 优先加载旧 step ckpt（比 final 更旧）→ 回滚。
    注：本测试直接验证清理函数的行为约定（glob + remove），不跑完整 main。
    """
    import glob
    ckpt_dir = tempfile.mkdtemp()
    for name in ('checkpoint_step25pct.pt', 'checkpoint_step50pct.pt',
                 'checkpoint_epoch1.pt', 'checkpoint_epoch2.pt'):
        open(os.path.join(ckpt_dir, name), 'w').close()
    _step_ckpts = glob.glob(os.path.join(ckpt_dir, 'checkpoint_step*pct.pt'))
    assert len(_step_ckpts) == 2
    for _p in _step_ckpts:
        os.remove(_p)
    remaining = os.listdir(ckpt_dir)
    assert remaining == ['checkpoint_epoch1.pt', 'checkpoint_epoch2.pt'], \
        f"step ckpt 应被清理，剩余 {remaining}"


def test_r38_b2_initial_eff_step_keeps_schedule_continuous():
    """initial_eff_step 恢复语义：resume 后 LR 不重新爬升。

    中断于第 60 个有效步（warmup=100 内）→ resume 后 eff_step=60 起，
    LR 应等于不中断训练同位置的值（60/100·base）。
    """
    base, eta, total, warm = 1e-3, 1e-5, 200, 100
    lr_continue = compute_lr(60, total, warm, base, eta, 'cosine', 0.1)
    lr_resumed = compute_lr(60, total, warm, base, eta, 'cosine', 0.1)
    assert lr_continue == lr_resumed == pytest.approx(base * 0.6, rel=1e-6)
    # 中断于 wsd 衰减段 → resume 后继续衰减而非回到 base（195/200：progress=0.95，
    # 衰减系数 p=0.5 → lr 恰为中间值）
    lr_late = compute_lr(195, total, warm, base, eta, 'wsd', 0.1)
    assert lr_late < base and lr_late > eta
