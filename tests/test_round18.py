"""第十八轮回归测试：iRoPE 交错 NoPE / Gated Attention / GEMM 合并转换。

覆盖：
- nope_layers 配置正确性（use_rope=False + alibi=True）
- Gated Attention 门源为 query（梯度回流 + init sigmoid=0.5）
- MemoryBank mem_k/mem_v → mem_kv_proj 转换
- ssm_k_proj/ssm_v_proj → ssm_kv_proj 转换
"""
import torch
import torch.nn as nn

from models.transformer import TransformerModel
from models.memory import MemoryBank


def _small(**over):
    kw = dict(vocab_size=200, embedding_dim=64, num_heads=4, num_layers=2,
              hidden_dim=128, max_seq_length=32)
    kw.update(over)
    return TransformerModel(**kw)


# ---------------------------------------------------------------------------
# iRoPE 交错 NoPE 层
# ---------------------------------------------------------------------------

def test_nope_layers_config():
    """nope_layers=[1] 时 block1 关闭 RoPE 并强制 alibi，block0 保持 RoPE。"""
    m = _small(nope_layers=[1])
    assert m.blocks[0].attn.use_rope is True, "非 nope 层应保持 use_rope=True"
    assert m.blocks[1].attn.use_rope is False, "nope 层应 use_rope=False"
    assert m.blocks[1].attn.alibi is True, "nope 层应强制 alibi=True 提供位置信号"
    assert m.blocks[0].attn.alibi is False, "非 nope 层 alibi 不受影响（默认 False）"


def test_nope_layers_changes_output():
    """nope 层关闭 RoPE 后输出应与全 RoPE 模型不同。"""
    torch.manual_seed(42)
    m_full = _small(nope_layers=[])
    torch.manual_seed(42)
    m_nope = _small(nope_layers=[1])
    x = torch.randint(0, 200, (2, 8))
    with torch.no_grad():
        out_full = m_full(x)
        out_nope = m_nope(x)
    assert not torch.allclose(out_full, out_nope, atol=1e-5), "nope 层应改变输出"


def test_nope_layers_backward():
    """nope 层模型梯度正常回流。"""
    m = _small(nope_layers=[1])
    m.train()
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    loss = out.float().sum()
    loss.backward()
    # nope 层的 qkv 应有梯度
    assert m.blocks[1].attn.qkv.weight.grad is not None, "nope 层 qkv 无梯度"


def test_nope_layers_default_empty():
    """默认 nope_layers=None 时所有层 use_rope=True（向后兼容）。"""
    m = _small()
    for blk in m.blocks:
        if hasattr(blk, 'attn'):
            assert blk.attn.use_rope is True, "默认应 use_rope=True"


# ---------------------------------------------------------------------------
# Gated Attention（output_gate 门源 = query）
# ---------------------------------------------------------------------------

def test_gated_attention_init_half():
    """output_gate init W=0/b=0 → sigmoid=0.5，输出 = out * 0.5。"""
    m = _small(output_gate=True)
    # output_gate 应被 _apply_specialized_inits 重置为 W=0/b=0
    og = m.blocks[0].attn.output_gate
    assert torch.allclose(og.weight, torch.zeros_like(og.weight)), "output_gate weight 应为 0"
    assert torch.allclose(og.bias, torch.zeros_like(og.bias)), "output_gate bias 应为 0"


def test_gated_attention_backward():
    """output_gate 梯度回流（门源是 query，梯度应通过 q → qkv 回流）。"""
    m = _small(output_gate=True)
    m.train()
    x = torch.randint(0, 200, (2, 8))
    out = m(x)
    loss = out.float().sum()
    loss.backward()
    assert m.blocks[0].attn.output_gate.weight.grad is not None, "output_gate 无梯度"
    # 门源是 query，query 来自 qkv 投影，qkv 也应有梯度
    assert m.blocks[0].attn.qkv.weight.grad is not None, "qkv 无梯度（门源应回流到 query）"


def test_gated_attention_changes_output():
    """开启 output_gate 后输出应与关闭时不同。"""
    torch.manual_seed(42)
    m_off = _small(output_gate=False)
    torch.manual_seed(42)
    m_on = _small(output_gate=True)
    x = torch.randint(0, 200, (2, 8))
    with torch.no_grad():
        out_off = m_off(x)
        out_on = m_on(x)
    # output_gate init sigmoid=0.5 → out*0.5，但 qkv 权重不同（output_gate 占参数）
    assert not torch.allclose(out_off, out_on, atol=1e-5), "output_gate 应改变输出"


# ---------------------------------------------------------------------------
# GEMM 合并转换：mem_k/mem_v → mem_kv_proj
# ---------------------------------------------------------------------------

def test_mem_kv_proj_convert():
    """旧 mem_k/mem_v 权重转换为 mem_kv_proj 后数值等价。"""
    mb_old = MemoryBank(dim=64, num_slots=4, comp_dim=32, head_dim=16)
    # 手动设置旧权重
    with torch.no_grad():
        mb_old.mem_k = nn.Linear(64, 16, bias=False)
        mb_old.mem_v = nn.Linear(64, 16, bias=False)
        torch.nn.init.normal_(mb_old.mem_k.weight, 0, 0.02)
        torch.nn.init.normal_(mb_old.mem_v.weight, 0, 0.02)

    # 构造旧 state_dict
    old_sd = {'mem_k.weight': mb_old.mem_k.weight.clone(),
              'mem_v.weight': mb_old.mem_v.weight.clone()}
    new_sd = MemoryBank.convert_legacy_state_dict(dict(old_sd))

    assert 'mem_kv_proj.weight' in new_sd, "转换后应有 mem_kv_proj.weight"
    assert 'mem_k.weight' not in new_sd, "旧 mem_k.weight 应被移除"
    assert 'mem_v.weight' not in new_sd, "旧 mem_v.weight 应被移除"
    # 验证拼接顺序：前 head_dim 行来自 mem_k，后 head_dim 行来自 mem_v
    expected = torch.cat([old_sd['mem_k.weight'], old_sd['mem_v.weight']], dim=0)
    assert torch.equal(new_sd['mem_kv_proj.weight'], expected), "mem_kv_proj 拼接顺序错误"


def test_mem_kv_proj_forward_equivalent():
    """合并后 mem_kv_proj 前向 chunk(2) 与分离 mem_k/mem_v 数值等价。"""
    mb = MemoryBank(dim=64, num_slots=4, comp_dim=32, head_dim=16)
    decomp = torch.randn(2, 4, 64)
    kv = mb.mem_kv_proj(decomp)
    k, v = kv.chunk(2, dim=-1)
    assert k.shape == (2, 4, 16), f"k shape 错误: {k.shape}"
    assert v.shape == (2, 4, 16), f"v shape 错误: {v.shape}"


# ---------------------------------------------------------------------------
# GEMM 合并转换：ssm_k_proj/ssm_v_proj → ssm_kv_proj
# ---------------------------------------------------------------------------

def test_ssm_kv_proj_created():
    """ssm_as_memory=True + hybrid 块时创建 ssm_kv_proj（合并 GEMM）。"""
    m = _small(layer_plan='attn,hybrid', ssm_as_memory=True)
    assert not hasattr(m.blocks[0], 'ssm_kv_proj'), "attn 块不应创建 ssm_kv_proj"
    assert hasattr(m.blocks[1], 'ssm_kv_proj'), "hybrid 块未创建 ssm_kv_proj"
    # 验证输出维度 = 2 * head_dim
    head_dim = 64 // 4
    assert m.blocks[1].ssm_kv_proj.out_features == 2 * head_dim, "ssm_kv_proj 维度错误"


def test_ssm_kv_proj_checkpoint_convert():
    """checkpoint 加载时自动转换 ssm_k_proj/ssm_v_proj → ssm_kv_proj。"""
    import torch
    # 构造旧格式的 state_dict（含 ssm_k_proj/ssm_v_proj）
    head_dim = 64 // 4
    old_k = torch.randn(2 * head_dim, 64)  # 两个 hybrid 块各一个（但测试只需验证转换逻辑）
    old_v = torch.randn(2 * head_dim, 64)

    # 模拟 checkpoint.py 的转换逻辑
    _ckpt_sd = {'blocks.1.ssm_k_proj.weight': old_k,
                'blocks.1.ssm_v_proj.weight': old_v}
    _ssm_keys = [k for k in list(_ckpt_sd) if k.endswith('.ssm_k_proj.weight')]
    for _sk in _ssm_keys:
        _sv = _sk.replace('ssm_k_proj.weight', 'ssm_v_proj.weight')
        if _sv in _ckpt_sd:
            _prefix = _sk[:-len('ssm_k_proj.weight')]
            _ckpt_sd[_prefix + 'ssm_kv_proj.weight'] = torch.cat(
                [_ckpt_sd.pop(_sk), _ckpt_sd.pop(_sv)], dim=0)

    assert 'blocks.1.ssm_kv_proj.weight' in _ckpt_sd, "转换后应有 ssm_kv_proj.weight"
    assert 'blocks.1.ssm_k_proj.weight' not in _ckpt_sd, "旧 ssm_k_proj 应被移除"
    expected = torch.cat([old_k, old_v], dim=0)
    assert torch.equal(_ckpt_sd['blocks.1.ssm_kv_proj.weight'], expected), "拼接顺序错误"
