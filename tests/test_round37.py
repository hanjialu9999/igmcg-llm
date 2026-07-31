"""第三十七轮回归测试：ssm_kwargs 共享 dict 变异 + CAST 初始化被跳过。

R37-1 根因：TransformerModel 构造一次 ssm_kwargs（含 ssm_type）共享传给所有块，
TransformerBlock 内 `ssm_kwargs.pop('ssm_type', 'standard')` 变异共享 dict——
块 0 pop 后，块 ≥1 全部静默回落 'standard'。ssm_type: cast + 多层 SSM 时仅第 0 层
是 CAST，其余为标准 Mamba。旧测试全用单层（layer_plan='ssm'），故从未暴露。

R37-2 根因：TransformerModel._init_weights 用 `type(m) is MambaSSM` 精确匹配，
MambaSSMWithCAST 是子类被漏掉——通用 N(0,0.02)+bias=0 覆盖后不再重放 proper_init，
导致：in/out/x/dt_proj 权重非 Xavier、dt_proj.bias=0（非 0.1）、cast_delta_proj
零初始化设计意图被毁（mixers.py:1883「初始行为等价于标准 Mamba」失效）。
修复：isinstance + MambaSSMWithCAST.proper_init 覆写（super + 归零 cast_delta_proj）。

本测试验证：
1. R37-1：layer_plan='ssm,ssm' 或 'hybrid,ssm' + ssm_type='cast' 时所有 SSM 块均为
   MambaSSMWithCAST；默认 standard 时均为 MambaSSM
2. R37-2a：cast 模型 dt_proj.bias=0.1、cast_delta_proj 全零、in_proj 权重为 Xavier 尺度
3. R37-2b：cast_stat_proj 置零（delta≡0）时 CAST 前向与标准 Mamba 数值一致
   （设计意图：初始行为等价于标准 Mamba）
4. R37-2c：delta 注入非零时输出确实偏离标准 Mamba（守卫 3 不是删除机制的结果）
5. 端到端：多层 cast 模型 forward/backward 正常（cast_delta_proj 有梯度）
6. R37-4（审查顺带发现）：MambaSSM 分块增量（L>1 + past_conv_state）丢失 conv 历史
   上下文——块首 conv_kernel-1 个位置零填充致 cache parity 失效；修复后分块/纯逐 token
   增量均与全量前向一致（浮点精度级）
"""
import pytest
import torch

from models.mixers import MambaSSM, MambaSSMWithCAST
from models.transformer import TransformerModel


def _build(num_layers=2, layer_plan='ssm,ssm', ssm_type='standard', **kw):
    kw.setdefault('vocab_size', 200)
    kw.setdefault('embedding_dim', 64)
    kw.setdefault('num_heads', 4)
    kw.setdefault('hidden_dim', 128)
    kw.setdefault('max_seq_length', 32)
    kw.setdefault('ssm_d_state', 8)
    return TransformerModel(
        num_layers=num_layers, layer_plan=layer_plan, ssm_type=ssm_type,
        gradient_checkpointing=False, tie_weights=False, **kw)


# ===========================================================================
# 1. R37-1：ssm_type=cast 应用到所有 SSM 块（共享 dict 变异回归）
# ===========================================================================

def test_r37_1_cast_applies_to_all_ssm_blocks():
    """'ssm,ssm' + cast：两层都必须是 MambaSSMWithCAST（旧 bug 下块 1 回落 standard）。"""
    m = _build(layer_plan='ssm,ssm', ssm_type='cast')
    for i, blk in enumerate(m.blocks):
        assert isinstance(blk.ssm, MambaSSMWithCAST), \
            f"block {i} ssm 应为 MambaSSMWithCAST，实际 {type(blk.ssm).__name__}"


def test_r37_1_cast_applies_to_hybrid_ssm_mix():
    """'hybrid,ssm' + cast：两个 SSM（hybrid 块内 + 纯 ssm 块）都必须是 CAST。"""
    m = _build(layer_plan='hybrid,ssm', ssm_type='cast', mixer='attn',
               hybrid_single_gate=True)
    for i, blk in enumerate(m.blocks):
        assert isinstance(blk.ssm, MambaSSMWithCAST), \
            f"block {i} ssm 应为 MambaSSMWithCAST，实际 {type(blk.ssm).__name__}"


def test_r37_1_default_standard_still_plain_mamba():
    """默认 ssm_type 不受影响：两层均为标准 MambaSSM。"""
    m = _build(layer_plan='ssm,ssm', ssm_type='standard')
    for i, blk in enumerate(m.blocks):
        assert type(blk.ssm) is MambaSSM, \
            f"block {i} ssm 应为 MambaSSM，实际 {type(blk.ssm).__name__}"


# ===========================================================================
# 2. R37-2a：CAST 初始化覆盖（isinstance 回归）
# ===========================================================================

def test_r37_2_cast_proper_init_survives_model_init():
    """cast 模型经 TransformerModel._init_weights 后：
    dt_proj.bias=0.1（遗忘偏置）、cast_delta_proj 全零、in_proj 为 Xavier 尺度。"""
    m = _build(layer_plan='ssm,ssm', ssm_type='cast')
    for i, blk in enumerate(m.blocks):
        ssm = blk.ssm
        # dt_proj.bias 专用初始化 0.1（旧 bug 下被 N(0,0.02) 通用 init 覆盖为 0）
        assert torch.allclose(ssm.dt_proj.bias,
                              torch.full_like(ssm.dt_proj.bias, 0.1)), \
            f"block {i} dt_proj.bias 应为 0.1（遗忘偏置），实际 {ssm.dt_proj.bias[:3].tolist()}"
        # cast_delta_proj 零初始化（A_delta≡0 → 初始等价标准 Mamba）
        assert ssm.cast_delta_proj.weight.abs().max().item() == 0.0, \
            f"block {i} cast_delta_proj.weight 应全零，实际 max={ssm.cast_delta_proj.weight.abs().max().item()}"
        # in_proj Xavier 尺度：uniform(-a,a), a=sqrt(6/(fan_in+fan_out)),
        # std = sqrt(2/(fan_in+fan_out)) = sqrt(2/(64+128)) ≈ 0.102
        w = ssm.in_proj.weight
        fan_in, fan_out = w.size(1), w.size(0)
        expected_std = (2.0 / (fan_in + fan_out)) ** 0.5
        assert abs(w.std().item() - expected_std) < 0.02 * expected_std, \
            f"block {i} in_proj std={w.std().item():.4f}，期望 ≈{expected_std:.4f}"


# ===========================================================================
# 3. R37-2b：cast_delta_proj=0 时 CAST 前向 ≡ 标准 Mamba（设计意图守卫）
# ===========================================================================

def test_r37_2_cast_zero_delta_matches_standard_mamba():
    """cast_stat_proj 置零 → h=0 → A_delta≡0 → 输出与标准 Mamba 数值一致。

    构造同参 MambaSSM 与 MambaSSMWithCAST，把共享参数复制过去，
    再置零 cast_stat_proj（delta≡0 的充分条件：silu(0)=0 → delta=cast_delta_proj(0)=0）。
    """
    torch.manual_seed(0)
    ref = MambaSSM(dim=32, d_state=8)
    cast = MambaSSMWithCAST(dim=32, d_state=8)
    with torch.no_grad():
        for name, p in ref.named_parameters():
            cast.get_parameter(name).data.copy_(p.data)
        cast.cast_stat_proj.weight.zero_()
    x = torch.randn(2, 8, 32)
    with torch.no_grad():
        y_ref, _, _ = ref(x)
        y_cast, _, _ = cast(x)
    assert torch.allclose(y_cast, y_ref, atol=1e-6), \
        f"zero-delta CAST 应等于标准 Mamba，max diff={(y_cast - y_ref).abs().max().item()}"


# ===========================================================================
# 4. R37-2c：delta 注入确实生效（守卫测试 3 不是「机制被删」的结果）
# ===========================================================================

def test_r37_2_cast_nonzero_delta_changes_output():
    """cast_stat_proj 置一 → A_delta≠0 → 输出偏离标准 Mamba。"""
    torch.manual_seed(0)
    ref = MambaSSM(dim=32, d_state=8)
    cast = MambaSSMWithCAST(dim=32, d_state=8)
    with torch.no_grad():
        for name, p in ref.named_parameters():
            cast.get_parameter(name).data.copy_(p.data)
        cast.cast_stat_proj.weight.fill_(1.0)   # h=ones·stats → silu>0 → delta≠0
        cast.cast_delta_proj.weight.fill_(0.01)  # proper_init 已归零，需设非零才有 delta
    x = torch.randn(2, 8, 32)
    with torch.no_grad():
        y_ref, _, _ = ref(x)
        y_cast, _, _ = cast(x)
    assert not torch.allclose(y_cast, y_ref, atol=1e-4), \
        "delta 注入应使 CAST 输出偏离标准 Mamba（机制失效？）"


# ===========================================================================
# 5. 端到端：多层 cast 模型 forward/backward
# ===========================================================================

def test_r37_end_to_end_cast_multi_layer_train_step():
    """多层 'ssm,ssm' cast 模型：forward 有限值 + backward 梯度流到 cast_delta_proj。"""
    torch.manual_seed(0)
    m = _build(layer_plan='ssm,ssm', ssm_type='cast')
    x = torch.randint(0, 200, (2, 16))
    y = torch.randint(0, 200, (2, 16))
    loss = torch.nn.functional.cross_entropy(
        m(x).transpose(1, 2), y, ignore_index=200)
    assert torch.isfinite(loss).item()
    loss.backward()
    for i, blk in enumerate(m.blocks):
        assert blk.ssm.cast_delta_proj.weight.grad is not None, \
            f"block {i} cast_delta_proj 应收到梯度"
        assert blk.ssm.cast_stat_proj.weight.grad is not None, \
            f"block {i} cast_stat_proj 应收到梯度"


# ===========================================================================
# 6. R37-4（审查顺带发现）：MambaSSM 分块增量丢失 conv 历史上下文
# ===========================================================================

def test_r37_4_chunked_incr_conv_carry_matches_full():
    """分块增量（prefill L=4 + 块 L=4）与全量前向一致（紧容差）。

    根因：MambaSSM.forward 分块增量（L>1 且带 past_conv_state）旧实现直接丢弃
    conv 历史（块首 conv_kernel-1 个位置零填充），与全量前向不一致——cache parity
    bug，被旧弱初始化（dt≈0 状态近似静态）掩盖。proper_init（R37-2）恢复强动力学后
    暴露（标准 Mamba 与 CAST 均差 ~0.15）。修复：拼接历史窗口再卷积（offset=keep）。
    """
    for ssm_type in ('standard', 'cast'):
        torch.manual_seed(0)
        m = _build(layer_plan='ssm', ssm_type=ssm_type)
        m.eval()
        x = torch.randint(0, 200, (1, 8))
        with torch.no_grad():
            full = m(x)
            _, past = m(x[:, :4], use_cache=True)
            inc, _ = m(x[:, 4:], past_key_values=past, use_cache=True)
        d = (full[:, 4:] - inc).abs().max().item()
        assert d < 1e-5, f"{ssm_type} 分块增量 vs 全量 diff={d}（应为浮点精度级）"


def test_r37_4_pure_incremental_steps_match_full():
    """纯逐 token 增量（L=1 步进）与全量前向一致（回归：L==1 分支不受影响）。"""
    for ssm_type in ('standard', 'cast'):
        torch.manual_seed(0)
        m = _build(layer_plan='ssm', ssm_type=ssm_type)
        m.eval()
        x = torch.randint(0, 200, (1, 8))
        with torch.no_grad():
            full = m(x)
            past = None
            outs = []
            for t in range(8):
                out, past = m(x[:, t:t + 1], past_key_values=past, use_cache=True)
                outs.append(out)
            inc = torch.cat(outs, dim=1)
        d = (full - inc).abs().max().item()
        assert d < 1e-5, f"{ssm_type} 逐 token 增量 vs 全量 diff={d}"
