"""第三十二轮回归测试：DifferentialAttention qkv12 + TransformerBlock highway_gates GEMM 合并。

覆盖：
- R32-1: DifferentialAttention 的 qkv + qkv2 合并为单个 qkv12 GEMM
  - 数学等价：cat([qkv(x), qkv2(x)], dim=-1) == qkv12(x).reshape(B,T,5,H,D)
  - state_dict 兼容：convert_legacy_state_dict 正确转换旧格式
  - 参数创建：shared_qkv=None 时创建 qkv12 而非 qkv/qkv2
  - shared_qkv 路径：保持原双 GEMM 结构
  - 梯度回流：qkv12.weight 收到梯度
  - cache parity：训练全量 vs 增量解码数值一致
  - 端到端：DifferentialAttention 模型可训练

- R32-2: TransformerBlock 的 sub1_highway + ffn_highway 合并为 highway_gates
  - 数学等价：cat([sub1_highway(x), ffn_highway(x)]) == highway_gates(x).chunk(2)
  - 参数创建：非 hybrid 块创建 highway_gates，hybrid 块保留 ffn_highway
  - 初始化正确：weight=0，bias=3.0（progressive_residual 时按 1/sqrt(depth) 衰减）
  - 梯度回流：highway_gates.weight 收到梯度
  - cache parity：训练全量 vs 增量解码数值一致
  - 端到端：highway_gate 模型可训练
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mixers import DifferentialAttention


# ─── 测试夹具 ──────────────────────────────────────────────────────────────

def _make_diff_attn(shared_qkv=None, **over):
    """构造 DifferentialAttention 模块用于单元测试。"""
    kw = dict(dim=64, num_heads=4, max_seq_length=16)
    kw.update(over)
    return DifferentialAttention(shared_qkv=shared_qkv, **kw)


def _make_model(**over):
    """构造小 TransformerModel 用于端到端测试。"""
    from models.transformer import TransformerModel
    kw = dict(vocab_size=200, embedding_dim=64, num_heads=4, num_layers=2,
              hidden_dim=128, max_seq_length=32, mixer='diff')
    kw.update(over)
    return TransformerModel(**kw)


def _make_highway_model(**over):
    """构造启用 highway_gate 的小 TransformerModel。"""
    from models.transformer import TransformerModel
    kw = dict(vocab_size=200, embedding_dim=64, num_heads=4, num_layers=2,
              hidden_dim=128, max_seq_length=32, highway_gate=True)
    kw.update(over)
    return TransformerModel(**kw)


# ═══════════════════════════════════════════════════════════════════════
# R32-1: DifferentialAttention qkv12 合并
# ═══════════════════════════════════════════════════════════════════════

def test_qkv12_param_created():
    """R32-1: shared_qkv=None 时创建 qkv12 而非 qkv/qkv2。"""
    m = _make_diff_attn()
    assert hasattr(m, 'qkv12'), "qkv12 未创建"
    assert m.qkv12.weight.shape == (5 * 64, 64), f"qkv12.weight 形状错误: {m.qkv12.weight.shape}"
    assert not hasattr(m, 'qkv'), "shared_qkv=None 时不应创建 qkv"
    assert not hasattr(m, 'qkv2'), "shared_qkv=None 时不应创建 qkv2"
    assert m.qkv12.bias is None, "qkv12 应无 bias（与原 qkv/qkv2 一致）"


def test_qkv12_shared_qkv_path():
    """R32-1: shared_qkv 模式保持原双 GEMM 结构。"""
    shared = nn.Linear(64, 3 * 64, bias=False)
    m = _make_diff_attn(shared_qkv=shared)
    assert m.shared_qkv_enabled, "shared_qkv 模式未启用"
    assert m.qkv is shared, "qkv 应为传入的 shared_qkv"
    assert hasattr(m, 'qkv2'), "shared_qkv 模式应创建 qkv2"
    assert not hasattr(m, 'qkv12'), "shared_qkv 模式不应创建 qkv12"


def test_qkv12_numerical_equivalence():
    """R32-1: qkv12(x).reshape(B,T,5,H,D) 与 cat([qkv(x), qkv2(x)]) 数值等价。"""
    torch.manual_seed(42)
    dim, num_heads = 64, 4
    head_dim = dim // num_heads

    # 模拟旧结构：两个独立 Linear
    legacy_qkv = nn.Linear(dim, 3 * dim, bias=False)
    legacy_qkv2 = nn.Linear(dim, 2 * dim, bias=False)
    nn.init.normal_(legacy_qkv.weight, 0, 0.1)
    nn.init.normal_(legacy_qkv2.weight, 0, 0.1)

    # 构造合并后的 Linear：weight = cat([qkv.weight, qkv2.weight], dim=0)
    merged = nn.Linear(dim, 5 * dim, bias=False)
    with torch.no_grad():
        merged.weight.copy_(torch.cat([legacy_qkv.weight, legacy_qkv2.weight], dim=0))

    B, T, H, D = 2, 8, num_heads, head_dim
    x = torch.randn(B, T, dim)

    # 旧路径：cat([qkv(x), qkv2(x)], dim=-1) → reshape
    legacy_out = torch.cat([legacy_qkv(x), legacy_qkv2(x)], dim=-1)  # (B, T, 5*dim)
    legacy_5d = legacy_out.reshape(B, T, 5, H, D)
    legacy_q1, legacy_k1, legacy_v, legacy_q2, legacy_k2 = legacy_5d.unbind(dim=2)

    # 新路径：qkv12(x) → reshape → unbind
    merged_out = merged(x)  # (B, T, 5*dim)
    merged_5d = merged_out.reshape(B, T, 5, H, D)
    merged_q1, merged_k1, merged_v, merged_q2, merged_k2 = merged_5d.unbind(dim=2)

    max_diff = (legacy_5d - merged_5d).abs().max().item()
    assert max_diff < 1e-6, f"qkv12 数值不等价: max_diff={max_diff:.2e}"


def test_qkv12_state_dict_conversion():
    """R32-1: convert_legacy_state_dict 正确转换旧 qkv/qkv2 → qkv12。"""
    dim = 64
    legacy_qkv = nn.Linear(dim, 3 * dim, bias=False)
    legacy_qkv2 = nn.Linear(dim, 2 * dim, bias=False)
    nn.init.normal_(legacy_qkv.weight, 0, 0.1)
    nn.init.normal_(legacy_qkv2.weight, 0, 0.1)

    # 模拟旧 state_dict（带常见前缀）
    legacy_sd = {
        'attn.qkv.weight': legacy_qkv.weight.clone(),
        'attn.qkv2.weight': legacy_qkv2.weight.clone(),
        'attn.proj.weight': torch.randn(dim, dim),
    }

    new_sd = DifferentialAttention.convert_legacy_state_dict(legacy_sd)

    assert 'attn.qkv12.weight' in new_sd, "转换后应含 qkv12.weight"
    assert 'attn.qkv.weight' not in new_sd, "转换后不应再含 qkv.weight"
    assert 'attn.qkv2.weight' not in new_sd, "转换后不应再含 qkv2.weight"
    assert 'attn.proj.weight' in new_sd, "无关字段应保留"

    expected = torch.cat([legacy_qkv.weight, legacy_qkv2.weight], dim=0)
    assert torch.equal(new_sd['attn.qkv12.weight'], expected), "qkv12.weight 应为 cat([qkv.weight, qkv2.weight])"


def test_qkv12_state_dict_conversion_top_level():
    """R32-1: 顶层无前缀的 state_dict 也能正确转换。"""
    dim = 32
    legacy_sd = {
        'qkv.weight': torch.randn(3 * dim, dim),
        'qkv2.weight': torch.randn(2 * dim, dim),
        'proj.weight': torch.randn(dim, dim),
    }
    new_sd = DifferentialAttention.convert_legacy_state_dict(legacy_sd)
    assert 'qkv12.weight' in new_sd
    assert 'qkv.weight' not in new_sd
    assert 'qkv2.weight' not in new_sd
    expected = torch.cat([legacy_sd['qkv.weight'], legacy_sd['qkv2.weight']], dim=0)
    assert torch.equal(new_sd['qkv12.weight'], expected)


def test_qkv12_gradient():
    """R32-1: qkv12.weight 收到梯度。"""
    m = _make_diff_attn()
    m.train()
    x = torch.randn(2, 8, 64)
    out, _ = m(x)
    out.sum().backward()
    assert m.qkv12.weight.grad is not None, "qkv12.weight 无梯度"
    assert m.qkv12.weight.grad.abs().sum() > 0, "qkv12.weight 梯度为零"


def test_qkv12_forward_output_shape():
    """R32-1: qkv12 前向输出形状正确。"""
    m = _make_diff_attn()
    x = torch.randn(2, 8, 64)
    out, _ = m(x)
    assert out.shape == (2, 8, 64), f"输出形状错误: {out.shape}"


def test_qkv12_cache_parity():
    """R32-1: 训练全量前向 vs 增量解码数值一致。"""
    torch.manual_seed(42)
    m = _make_diff_attn()
    m.eval()
    x = torch.randn(1, 6, 64)

    # 全量前向
    with torch.no_grad():
        full_out, _ = m(x, use_cache=False)

    # 增量解码：逐 token 前向，累积 past_kv
    with torch.no_grad():
        past = None
        outs = []
        for t in range(6):
            out_t, present = m(x[:, t:t+1, :], past_kv=past, use_cache=True, start_pos=t)
            outs.append(out_t)
            past = present  # DifferentialAttention 返回 (k1, k2, v) 三元组
        inc_out = torch.cat(outs, dim=1)

    max_diff = (full_out - inc_out).abs().max().item()
    assert max_diff < 1e-4, f"cache parity 失败: max_diff={max_diff:.2e}"


def test_qkv12_end_to_end():
    """R32-1: DifferentialAttention 模型端到端可训练。"""
    from models.transformer import TransformerModel
    m = TransformerModel(vocab_size=100, embedding_dim=64, num_heads=4, num_layers=2,
                          hidden_dim=128, max_seq_length=16, mixer='diff')
    m.train()
    x = torch.randint(0, 100, (2, 8))
    out = m(x)
    logits = out["logits"] if isinstance(out, dict) else out
    loss = logits.sum()
    loss.backward()
    # 验证至少一个 block 的 qkv12 有梯度
    has_grad = any(getattr(blk.attn, 'qkv12', None) is not None
                   and blk.attn.qkv12.weight.grad is not None
                   for blk in m.blocks)
    assert has_grad, "无 qkv12.weight 梯度"


# ═══════════════════════════════════════════════════════════════════════
# R32-2: TransformerBlock highway_gates 合并
# ═══════════════════════════════════════════════════════════════════════

def test_highway_gates_param_created():
    """R32-2: 非 hybrid 块创建 highway_gates（合并自 sub1_highway + ffn_highway）。"""
    m = _make_highway_model(num_layers=2)
    for blk in m.blocks:
        if blk.block_type != 'hybrid':
            assert hasattr(blk, 'highway_gates'), "非 hybrid 块应创建 highway_gates"
            assert not hasattr(blk, 'sub1_highway'), "旧 sub1_highway 应已合并为 highway_gates"
            assert not hasattr(blk, 'ffn_highway'), "旧 ffn_highway 应已合并为 highway_gates"


def test_highway_gates_init():
    """R32-2: highway_gates 初始化 weight=0, bias=3.0。"""
    m = _make_highway_model(num_layers=2, progressive_residual=False)
    for blk in m.blocks:
        if hasattr(blk, 'highway_gates'):
            assert blk.highway_gates.weight.abs().max().item() == 0.0, \
                "highway_gates weight 应 init 0"
            assert torch.allclose(blk.highway_gates.bias, torch.full_like(blk.highway_gates.bias, 3.0)), \
                "highway_gates bias 应为 3.0"


def test_highway_gates_init_progressive_residual():
    """R32-2: progressive_residual 时 bias 按 1/sqrt(depth) 衰减。"""
    import math
    m = _make_highway_model(num_layers=3, progressive_residual=True)
    for i, blk in enumerate(m.blocks):
        if hasattr(blk, 'highway_gates'):
            if i == 0:
                expected_bias = 3.0
            else:
                expected_bias = 3.0 / math.sqrt(i + 1)
            assert torch.allclose(blk.highway_gates.bias,
                                   torch.full_like(blk.highway_gates.bias, expected_bias)), \
                f"layer {i} highway_gates bias 应为 {expected_bias}"


def test_highway_gates_hybrid_block_keeps_ffn_highway():
    """R32-2: hybrid 块保留 ffn_highway（第一子层用 hybrid_attn_gate/hybrid_ssm_gate）。"""
    from models.transformer import TransformerModel
    m = TransformerModel(vocab_size=100, embedding_dim=64, num_heads=4, num_layers=2,
                          hidden_dim=128, max_seq_length=16,
                          layer_plan='hybrid,hybrid',
                          highway_gate=True)
    for blk in m.blocks:
        if blk.block_type == 'hybrid':
            assert hasattr(blk, 'ffn_highway'), "hybrid 块应保留 ffn_highway"
            assert not hasattr(blk, 'highway_gates'), "hybrid 块不应创建 highway_gates"


def test_highway_gates_numerical_equivalence():
    """R32-2: highway_gates(x).chunk(2) 与 cat([sub1_highway(x), ffn_highway(x)]) 数值等价。"""
    torch.manual_seed(42)
    dim = 64
    legacy_sub1 = nn.Linear(dim, 1)
    legacy_ffn = nn.Linear(dim, 1)
    nn.init.normal_(legacy_sub1.weight, 0, 0.1)
    nn.init.normal_(legacy_sub1.bias, 0, 0.1)
    nn.init.normal_(legacy_ffn.weight, 0, 0.1)
    nn.init.normal_(legacy_ffn.bias, 0, 0.1)

    # 合并后的 Linear
    merged = nn.Linear(dim, 2)
    with torch.no_grad():
        merged.weight.copy_(torch.cat([legacy_sub1.weight, legacy_ffn.weight], dim=0))
        merged.bias.copy_(torch.cat([legacy_sub1.bias, legacy_ffn.bias], dim=0))

    x = torch.randn(2, 8, dim)
    # 旧路径
    legacy_out = torch.cat([legacy_sub1(x), legacy_ffn(x)], dim=-1)  # (B, T, 2)
    # 新路径
    merged_out = merged(x)  # (B, T, 2)
    # chunk 拆分
    legacy_g1 = torch.sigmoid(legacy_out[..., 0:1])
    legacy_g2 = torch.sigmoid(legacy_out[..., 1:2])
    merged_g1 = torch.sigmoid(merged_out[..., 0:1])
    merged_g2 = torch.sigmoid(merged_out[..., 1:2])

    max_diff = (legacy_out - merged_out).abs().max().item()
    assert max_diff < 1e-6, f"highway_gates 数值不等价: max_diff={max_diff:.2e}"
    max_diff_g1 = (legacy_g1 - merged_g1).abs().max().item()
    max_diff_g2 = (legacy_g2 - merged_g2).abs().max().item()
    assert max_diff_g1 < 1e-6 and max_diff_g2 < 1e-6, "sigmoid 后数值不等价"


def test_highway_gates_gradient():
    """R32-2: highway_gates.weight 收到梯度。"""
    m = _make_highway_model(num_layers=2)
    m.train()
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    logits = out["logits"] if isinstance(out, dict) else out
    logits.sum().backward()
    for blk in m.blocks:
        if hasattr(blk, 'highway_gates'):
            assert blk.highway_gates.weight.grad is not None, "highway_gates.weight 无梯度"
            assert blk.highway_gates.weight.grad.abs().sum() > 0, "highway_gates.weight 梯度为零"


def test_highway_gates_backward_end_to_end():
    """R32-2: highway_gate 模型端到端可训练。"""
    m = _make_highway_model(num_layers=2)
    m.train()
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    logits = out["logits"] if isinstance(out, dict) else out
    loss = logits.float().sum()
    loss.backward()
    # 至少一个 highway_gates 有梯度
    has_grad = any(getattr(blk, 'highway_gates', None) is not None
                   and blk.highway_gates.weight.grad is not None
                   for blk in m.blocks)
    assert has_grad, "无 highway_gates.weight 梯度"


def test_highway_gates_forward_output():
    """R32-2: highway_gate 模型前向输出形状正确。"""
    m = _make_highway_model(num_layers=2)
    m.eval()
    x = torch.randint(0, 200, (2, 8))
    with torch.no_grad():
        out = m(x)
    logits = out["logits"] if isinstance(out, dict) else out
    assert logits.shape == (2, 8, 200), f"输出形状错误: {logits.shape}"


# ═══════════════════════════════════════════════════════════════════════
# 组合测试
# ═══════════════════════════════════════════════════════════════════════

def test_qkv12_with_highway_gates_combination():
    """R32 组合：DifferentialAttention + highway_gate 同时启用。"""
    from models.transformer import TransformerModel
    m = TransformerModel(vocab_size=100, embedding_dim=64, num_heads=4, num_layers=2,
                          hidden_dim=128, max_seq_length=16,
                          mixer='diff', highway_gate=True)
    m.train()
    x = torch.randint(0, 100, (2, 8))
    out = m(x)
    logits = out["logits"] if isinstance(out, dict) else out
    assert logits.shape == (2, 8, 100)
    loss = logits.sum()
    loss.backward()
    # 验证两种合并参数都有梯度
    has_qkv12_grad = any(getattr(blk.attn, 'qkv12', None) is not None
                        and blk.attn.qkv12.weight.grad is not None
                        for blk in m.blocks)
    has_highway_gates_grad = any(getattr(blk, 'highway_gates', None) is not None
                                  and blk.highway_gates.weight.grad is not None
                                  for blk in m.blocks)
    assert has_qkv12_grad, "DifferentialAttention qkv12 无梯度"
    assert has_highway_gates_grad, "TransformerBlock highway_gates 无梯度"


def test_qkv12_legacy_checkpoint_compatible():
    """R32-1: 模型用 qkv12 格式，加载旧 qkv/qkv2 checkpoint 不报错（strict=False）。"""
    from models.transformer import TransformerModel
    # 创建旧格式 checkpoint（qkv + qkv2）
    legacy_model = TransformerModel(vocab_size=100, embedding_dim=64, num_heads=4,
                                     num_layers=1, hidden_dim=128, max_seq_length=16,
                                     mixer='diff')
    # 手动构造旧格式 state_dict（模拟转换前）
    legacy_sd = legacy_model.state_dict()
    # 检查新模型默认就有 qkv12
    assert any(k.endswith('.qkv12.weight') for k in legacy_sd), "新模型应有 qkv12"

    # 通过 convert_legacy_state_dict 转换
    # 先模拟旧格式：把 qkv12 拆成 qkv + qkv2
    fake_legacy_sd = {}
    for k, v in legacy_sd.items():
        if k.endswith('.qkv12.weight'):
            prefix = k[:-len('qkv12.weight')]
            # qkv12 是 (5*dim, dim)，拆成 qkv (3*dim, dim) + qkv2 (2*dim, dim)
            fake_legacy_sd[prefix + 'qkv.weight'] = v[:3 * 64]
            fake_legacy_sd[prefix + 'qkv2.weight'] = v[3 * 64:]
        else:
            fake_legacy_sd[k] = v

    # 转换回新格式
    new_sd = DifferentialAttention.convert_legacy_state_dict(fake_legacy_sd)
    # 验证能加载到新模型
    missing, unexpected = legacy_model.load_state_dict(new_sd, strict=False)
    # qkv12 应已转换，不应在 missing 中
    qkv12_missing = [k for k in missing if 'qkv12' in k]
    assert not qkv12_missing, f"qkv12 转换后仍 missing: {qkv12_missing}"


def test_highway_gates_legacy_no_sub1_ffn_highway():
    """R32-2: 非 hybrid 块的 highway_gate 模型只含 highway_gates，不含旧 sub1_highway/ffn_highway。"""
    m = _make_highway_model(num_layers=2)
    sd = m.state_dict()
    highway_keys = [k for k in sd if 'highway' in k.lower()]
    assert highway_keys, "state_dict 应含 highway_gates 参数"
    for k in highway_keys:
        assert 'sub1_highway' not in k, f"state_dict 含旧 sub1_highway: {k}"
        assert 'ffn_highway' not in k, f"非 hybrid 块不应含 ffn_highway: {k}"
        assert 'highway_gates' in k, f"非 highway_gates 的 highway 参数: {k}"


# ─── R32-2 修复：convert_legacy_state_dict（sub1_highway + ffn_highway → highway_gates） ───

def test_highway_gates_convert_legacy_state_dict():
    """R32-2: TransformerBlock.convert_legacy_state_dict 正确转换旧 sub1_highway/ffn_highway → highway_gates。"""
    from models.transformer import TransformerBlock
    torch.manual_seed(42)
    dim = 64

    # 构造旧格式 state_dict（sub1_highway + ffn_highway 分离）
    sub1_w = torch.randn(1, dim)
    sub1_b = torch.randn(1)
    ffn_w = torch.randn(1, dim)
    ffn_b = torch.randn(1)
    legacy_sd = {
        'blocks.0.sub1_highway.weight': sub1_w,
        'blocks.0.sub1_highway.bias': sub1_b,
        'blocks.0.ffn_highway.weight': ffn_w,
        'blocks.0.ffn_highway.bias': ffn_b,
        # 其他无关参数保留
        'blocks.0.ln1.weight': torch.randn(dim),
    }

    new_sd = TransformerBlock.convert_legacy_state_dict(legacy_sd)

    # 旧键应已转换
    assert 'blocks.0.sub1_highway.weight' not in new_sd, "sub1_highway.weight 应已转换"
    assert 'blocks.0.sub1_highway.bias' not in new_sd, "sub1_highway.bias 应已转换"
    assert 'blocks.0.ffn_highway.weight' not in new_sd, "ffn_highway.weight 应已转换"
    assert 'blocks.0.ffn_highway.bias' not in new_sd, "ffn_highway.bias 应已转换"
    # 新键应已创建
    assert 'blocks.0.highway_gates.weight' in new_sd, "highway_gates.weight 应已创建"
    assert 'blocks.0.highway_gates.bias' in new_sd, "highway_gates.bias 应已创建"
    # 其他参数保留
    assert 'blocks.0.ln1.weight' in new_sd, "无关参数应保留"
    # 数值正确：cat([sub1, ffn], dim=0)
    expected_w = torch.cat([sub1_w, ffn_w], dim=0)
    expected_b = torch.cat([sub1_b, ffn_b], dim=0)
    assert torch.equal(new_sd['blocks.0.highway_gates.weight'], expected_w), \
        f"weight 不等: max_diff={(new_sd['blocks.0.highway_gates.weight'] - expected_w).abs().max().item():.2e}"
    assert torch.equal(new_sd['blocks.0.highway_gates.bias'], expected_b), \
        f"bias 不等: max_diff={(new_sd['blocks.0.highway_gates.bias'] - expected_b).abs().max().item():.2e}"


def test_highway_gates_convert_legacy_preserves_hybrid_ffn_highway():
    """R32-2: hybrid 块的 ffn_highway 保持原结构不转换（只转非 hybrid 块的 sub1_highway+ffn_highway）。

    hybrid 块只有 ffn_highway（无 sub1_highway），convert 不应处理。
    """
    from models.transformer import TransformerBlock
    torch.manual_seed(42)
    dim = 64

    # 模拟 hybrid 块的 state_dict：只有 ffn_highway（无 sub1_highway）
    ffn_w = torch.randn(1, dim)
    ffn_b = torch.randn(1)
    legacy_sd = {
        'blocks.0.ffn_highway.weight': ffn_w,  # hybrid 块只有 ffn_highway
        'blocks.0.ffn_highway.bias': ffn_b,
    }

    new_sd = TransformerBlock.convert_legacy_state_dict(legacy_sd)

    # ffn_highway 应保留（无 sub1_highway 配对，不触发合并）
    assert 'blocks.0.ffn_highway.weight' in new_sd, "hybrid 块 ffn_highway.weight 应保留"
    assert 'blocks.0.ffn_highway.bias' in new_sd, "hybrid 块 ffn_highway.bias 应保留"
    assert 'blocks.0.highway_gates.weight' not in new_sd, "hybrid 块不应创建 highway_gates"
    assert torch.equal(new_sd['blocks.0.ffn_highway.weight'], ffn_w), "数值应不变"


def test_highway_gates_legacy_checkpoint_load():
    """R32-2 端到端：旧格式 checkpoint 经 convert 后能加载到新格式模型。

    模拟旧 checkpoint（sub1_highway+ffn_highway 分离）→ 新模型（highway_gates 合并）。
    """
    from models.transformer import TransformerBlock
    torch.manual_seed(42)
    dim = 64

    # 构造旧格式模型（手动模拟，绕过 __init__ 创建 highway_gates）
    # 用 _make_highway_model 创建新格式模型
    m_new = _make_highway_model(num_layers=2)
    # 保存新模型的 highway_gates 权重作为"旧格式"的等价值
    # 实际上我们从新模型反推：split highway_gates → sub1_highway + ffn_highway
    hg_w = m_new.blocks[0].highway_gates.weight.data.clone()
    hg_b = m_new.blocks[0].highway_gates.bias.data.clone()
    # split 为 sub1_highway + ffn_highway（模拟旧 checkpoint）
    legacy_sd = {
        'blocks.0.sub1_highway.weight': hg_w[0:1].clone(),
        'blocks.0.sub1_highway.bias': hg_b[0:1].clone(),
        'blocks.0.ffn_highway.weight': hg_w[1:2].clone(),
        'blocks.0.ffn_highway.bias': hg_b[1:2].clone(),
    }
    # 加上其他必要参数（用新模型的 state_dict 填充）
    full_sd = m_new.state_dict()
    for k, v in full_sd.items():
        if 'highway_gates' not in k and k not in legacy_sd:
            legacy_sd[k] = v.clone()

    # 转换
    new_sd = TransformerBlock.convert_legacy_state_dict(legacy_sd)

    # 加载到新模型
    m_new2 = _make_highway_model(num_layers=2)
    m_new2.load_state_dict(new_sd, strict=False)

    # 验证前向输出一致
    m_new.eval()
    m_new2.eval()
    x = torch.randint(0, 200, (2, 8))
    with torch.no_grad():
        out1 = m_new(x)
        out2 = m_new2(x)
        if isinstance(out1, tuple):
            out1 = out1[0]
        if isinstance(out2, tuple):
            out2 = out2[0]
    assert torch.allclose(out1, out2, atol=1e-6), \
        f"旧 checkpoint 转换后前向不等价: max_diff={(out1 - out2).abs().max().item():.2e}"


def test_highway_gates_convert_legacy_empty_state_dict():
    """R32-2: 空 state_dict 不崩溃，返回空 dict。"""
    from models.transformer import TransformerBlock
    new_sd = TransformerBlock.convert_legacy_state_dict({})
    assert new_sd == {}, "空 state_dict 应返回空 dict"


def test_highway_gates_convert_legacy_no_sub1_highway():
    """R32-2: 只有 ffn_highway（无 sub1_highway）时不转换，ffn_highway 保留。"""
    from models.transformer import TransformerBlock
    ffn_w = torch.randn(1, 64)
    legacy_sd = {
        'blocks.0.ffn_highway.weight': ffn_w,  # 只有 ffn_highway（hybrid 块场景）
    }
    new_sd = TransformerBlock.convert_legacy_state_dict(legacy_sd)
    assert 'blocks.0.ffn_highway.weight' in new_sd, "ffn_highway 应保留"
    assert 'blocks.0.highway_gates.weight' not in new_sd, "不应创建 highway_gates"
    assert torch.equal(new_sd['blocks.0.ffn_highway.weight'], ffn_w), "数值应不变"
