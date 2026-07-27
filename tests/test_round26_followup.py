"""第二十六轮 follow-up 回归测试：补齐近期合并代码的测试覆盖缺口。

本文件针对自动测试缺口分析发现的 4 类高风险缺口补测，遵循项目现有测试框架与命名规范
（pytest + _small 辅助 + 确定性/隔离/独立可运行）。

覆盖缺口（按风险排序）：
1. [_fluency_batch 边界] commit f17bcfc 把逐序列 .item() 改为 torch.stack(means).tolist()
   单次同步。现有 test_fluency_batch_applies_log_softmax 仅测正常长度序列，未覆盖
   空序列列表 / 单 token 序列 / 空列表序列 / 混合序列——这些边界走 torch.stack 时
   若任一元素非 0-dim 张量会崩溃或静默返回错误形状。IGMCG 候选打分是业务关键流程。

2. [_apply_ngram_fusion 张量温度 clamp 效果] commit 02d0e3e 的 B1 修复支持 (N,) 张量
   温度。现有 test_ngram_fusion_tensor_temperature_clamp 仅验证输出有限，未验证
   clamp 实际生效——若 clamp 写错维度或未应用，输出仍可能有限但数值错误。本文件
   直接比较越界温度与 clamp 边界温度的输出等价性。

3. [_alibi_bias mem_cols + alibi_learnable 梯度流] commit f17bcfc 把 _alibi_bias 的
   mem_cols>0 路径从 bias.clone()+in-place 改为 bias*mask。现有
   test_alibi_memory_columns_zero 仅测 alibi_learnable=False（buffer）的前向值，
   未测 alibi_learnable=True（Parameter）时梯度能否穿过 mask 乘法正确回流到斜率——
   若 mask 广播维度错可能让梯度全零或回流到错误列。

4. [DifferentialAttention 动态掩码因果正确性] commit 02d0e3e 的 B4 修复在 T>max_seq_length
   时动态扩展因果掩码。现有 test_diff_attn_T_exceeds_max_seq_length 仅验证输出有限+
   形状正确，未验证扩展后的掩码仍是上三角（因果）——若 diagonal 写错会让未来信息
   泄漏到过去，输出仍有限但语义错误。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F

from models.transformer import TransformerModel
from models.mixers import SlidingWindowCausalSelfAttention, DifferentialAttention


def _small(**over):
    kw = dict(vocab_size=200, embedding_dim=64, num_heads=4, num_layers=2,
              hidden_dim=128, max_seq_length=32)
    kw.update(over)
    return TransformerModel(**kw)


# ============================================================================
# 缺口 1：_fluency_batch 边界（commit f17bcfc 批量化 .item() 路径）
# ============================================================================

def _build_ngram_fusion_model():
    """构造启用 ngram_fusion 的小模型（复用 test_new_mechanisms._small_ngram 模式）。"""
    from tests.test_new_mechanisms import _small_ngram
    v, ng = _small_ngram()
    m = _small(vocab_size=len(v), ngram_fusion=True, ngram_model=ng)
    return m, v


def test_fluency_batch_empty_sequences_list():
    """缺口1：空序列列表应返回 []（N==0 早返回路径）。

    抓住：commit f17bcfc 的 torch.stack(means).tolist() 在 means=[] 时会崩溃
    （torch.stack 空列表报 RuntimeError）。生产代码有 `if N == 0: return []`
    早返回保护，但无测试覆盖——若早返回被误删，空列表路径会崩溃。
    """
    from scripts.generate import _fluency_batch
    m = _small()
    m.eval()
    result = _fluency_batch(m, [], 'cpu', 0)
    assert result == [], f"空序列列表应返回 []，实际 {result}"


def test_fluency_batch_all_short_sequences():
    """缺口1：所有序列 len(s)<2 时应返回 [0.0, ...]（走 torch.tensor(0.0) 路径）。

    抓住：commit f17bcfc 把 out.append(0.0) 改为 means.append(torch.tensor(0.0,
    device=device))。若误写为 means.append(0.0)（Python float），后续
    torch.stack(means) 会报 TypeError。本测试验证所有短序列路径返回 0.0。
    """
    from scripts.generate import _fluency_batch
    m = _small()
    m.eval()
    # 全部单 token 序列
    result = _fluency_batch(m, [[1], [2], [3]], 'cpu', 0)
    assert len(result) == 3, f"应返回 3 个值，实际 {len(result)}"
    assert all(r == 0.0 for r in result), \
        f"短序列应返回 0.0，实际 {result}"


def test_fluency_batch_empty_list_sequence():
    """缺口1：空列表序列（len(s)==0）应返回 0.0（与 len(s)<2 同路径）。

    抓住：空列表 [] 走 `if s:` 为 False 不填充 batch，再走 `if len(s) < 2`
    为 True（0 < 2）append torch.tensor(0.0)。若逻辑改错（如先检查 len 再
    检查 truthy）可能崩溃。"""
    from scripts.generate import _fluency_batch
    m = _small()
    m.eval()
    result = _fluency_batch(m, [[], []], 'cpu', 0)
    assert result == [0.0, 0.0], f"空列表序列应返回 [0.0, 0.0]，实际 {result}"


def test_fluency_batch_mixed_short_and_long_sequences():
    """缺口1：混合短/长序列应正确返回（torch.stack 要求所有元素同形态）。

    抓住：commit f17bcfc 的 torch.stack(means).tolist() 要求 means 中所有元素
    都是 0-dim 张量。短序列 append torch.tensor(0.0)（0-dim），长序列 append
    lp.gather(...).mean()（0-dim）。若任一路径返回非 0-dim（如 1-dim），
    torch.stack 会成功但 tolist() 返回嵌套列表，破坏 IGMCG 候选排序。

    本测试构造 [短, 长, 空, 长] 混合并验证：
      1) 返回长度 == 输入长度
      2) 短/空序列返回 0.0
      3) 长序列返回有限实数（非嵌套列表）
    """
    from scripts.generate import _fluency_batch
    m = _small()
    m.eval()
    seqs = [[1], [5, 6, 7, 8], [], [10, 11, 12]]
    result = _fluency_batch(m, seqs, 'cpu', 0)
    assert len(result) == 4, f"应返回 4 个值，实际 {len(result)}"
    # 短序列和空序列返回 0.0
    assert result[0] == 0.0, f"短序列应返回 0.0，实际 {result[0]}"
    assert result[2] == 0.0, f"空序列应返回 0.0，实际 {result[2]}"
    # 长序列返回有限浮点数（非 list）
    assert isinstance(result[1], float), f"长序列应返回 float，实际 {type(result[1])}"
    assert isinstance(result[3], float), f"长序列应返回 float，实际 {type(result[3])}"
    import math
    assert math.isfinite(result[1]), f"长序列流畅度应有限，实际 {result[1]}"
    assert math.isfinite(result[3]), f"长序列流畅度应有限，实际 {result[3]}"


def test_fluency_batch_single_sequence():
    """缺口1：单条序列边界（N==1 时 torch.stack 单元素列表）。

    抓住：N==1 时 means 仅 1 个元素，torch.stack 返回 shape (1,) 的张量，
    tolist() 返回单元素列表。若误用 .item() 会返回标量破坏接口契约。"""
    from scripts.generate import _fluency_batch
    m = _small()
    m.eval()
    result = _fluency_batch(m, [[5, 6, 7]], 'cpu', 0)
    assert isinstance(result, list), f"应返回 list，实际 {type(result)}"
    assert len(result) == 1, f"应返回 1 个值，实际 {len(result)}"
    import math
    assert math.isfinite(result[0]), f"流畅度应有限，实际 {result[0]}"


def test_fluency_batch_batched_matches_per_sequence():
    """缺口1：批量化路径与逐序列计算数值等价（核心回归保护）。

    抓住：commit f17bcfc 把逐序列 .item() 改为 torch.stack(means).tolist()。
    虽然数学等价，但若 stack 顺序错或 tolist() 在某些 dtype 下行为不同，
    候选排序会失真。本测试直接比较批量化结果与逐序列手动计算结果。"""
    from scripts.generate import _fluency_batch
    m = _small()
    m.eval()
    seqs = [[5, 6, 7, 8, 9], [10, 11, 12], [13, 14]]
    batched = _fluency_batch(m, seqs, 'cpu', 0)

    # 逐序列手动计算（原实现路径）
    manual = []
    with torch.no_grad():
        for s in seqs:
            if len(s) < 2:
                manual.append(0.0)
                continue
            inp = torch.tensor([s], dtype=torch.long)
            logits = m.forward(inp)
            lp = F.log_softmax(logits[0, :len(s) - 1].float(), dim=-1)
            tgt = torch.tensor(s[1:]).unsqueeze(1)
            manual.append(lp.gather(1, tgt).mean().item())

    assert len(batched) == len(manual), "长度不一致"
    for b, m_val in zip(batched, manual):
        assert abs(b - m_val) < 1e-6, \
            f"批量化与逐序列不一致：batched={b}, manual={m_val}"


# ============================================================================
# 缺口 2：_apply_ngram_fusion 张量温度 clamp 效果（commit 02d0e3e B1 修复）
# ============================================================================

def test_ngram_fusion_tensor_temp_clamp_lower_bound():
    """缺口2：温度低于 0.01 应被 clamp 到 0.01（验证 clamp 实际生效）。

    抓住：现有 test_ngram_fusion_tensor_temperature_clamp 仅验证输出有限，
    未验证 clamp 真的生效。若 clamp 写错维度或未应用，温度 0.001 会让
    z/_t 爆炸但仍可能 finite（如被 nan_to_num 兜底）。本测试直接比较
    temp=0.001 与 temp=0.01 的输出——若 clamp 生效应完全一致。"""
    m, v = _build_ngram_fusion_model()
    m.eval()
    N, T = 2, 4
    inp = torch.randint(0, len(v), (N, T))
    with torch.no_grad():
        out_below = m(inp, temperature=torch.tensor([0.001, 0.001]))
        out_bound = m(inp, temperature=torch.tensor([0.01, 0.01]))
    assert torch.allclose(out_below, out_bound, atol=1e-6), \
        "temp=0.001 应被 clamp 到 0.01，与 temp=0.01 输出不一致"


def test_ngram_fusion_tensor_temp_clamp_upper_bound():
    """缺口2：温度高于 10 应被 clamp 到 10（验证 clamp 上界生效）。

    抓住：温度 20 会让 z/_t 过小，softmax 退化为均匀——若 clamp 未生效，
    输出与 temp=10 不同。本测试比较 temp=20 与 temp=10 的输出。"""
    m, v = _build_ngram_fusion_model()
    m.eval()
    N, T = 2, 4
    inp = torch.randint(0, len(v), (N, T))
    with torch.no_grad():
        out_above = m(inp, temperature=torch.tensor([20.0, 20.0]))
        out_bound = m(inp, temperature=torch.tensor([10.0, 10.0]))
    assert torch.allclose(out_above, out_bound, atol=1e-6), \
        "temp=20.0 应被 clamp 到 10.0，与 temp=10.0 输出不一致"


def test_ngram_fusion_tensor_temp_clamp_mixed_values():
    """缺口2：混合越界值应逐元素 clamp（验证 clamp 沿正确维度）。

    抓住：(N,) 张量温度的 clamp 应逐元素应用。若 clamp 沿错误维度
    （如对整个张量取 min/max），混合越界值会出错。本测试用
    [0.001, 5.0, 20.0] 验证：两端被 clamp，中间不变。"""
    m, v = _build_ngram_fusion_model()
    m.eval()
    N, T = 3, 4
    inp = torch.randint(0, len(v), (N, T))
    with torch.no_grad():
        out_mixed = m(inp, temperature=torch.tensor([0.001, 5.0, 20.0]))
        out_clamped = m(inp, temperature=torch.tensor([0.01, 5.0, 10.0]))
    assert torch.allclose(out_mixed, out_clamped, atol=1e-6), \
        "混合温度 [0.001, 5.0, 20.0] 应 clamp 到 [0.01, 5.0, 10.0]，输出不一致"


def test_ngram_fusion_tensor_temp_affects_output():
    """缺口2：不同合法温度应产生不同输出（验证温度确实参与计算）。

    抓住：若温度参数被意外忽略（如 _t 未传入 log_softmax），所有温度
    输出相同。本测试用两个差异显著的合法温度（0.5 和 5.0）验证输出不同。"""
    m, v = _build_ngram_fusion_model()
    m.eval()
    N, T = 2, 4
    inp = torch.randint(0, len(v), (N, T))
    with torch.no_grad():
        out_low = m(inp, temperature=torch.tensor([0.5, 0.5]))
        out_high = m(inp, temperature=torch.tensor([5.0, 5.0]))
    diff = (out_low - out_high).abs().max().item()
    assert diff > 1e-3, \
        f"温度 0.5 vs 5.0 输出几乎相同（diff={diff:.2e}）——温度可能未参与计算"


# ============================================================================
# 缺口 3：_alibi_bias mem_cols + alibi_learnable 梯度流（commit f17bcfc）
# ============================================================================

def test_alibi_learnable_gradient_through_mem_cols_mask():
    """缺口3：alibi_learnable=True + mem_cols>0 时梯度应穿过 mask 乘法回流到斜率。

    抓住：commit f17bcfc 把 mem_cols>0 路径从 bias.clone()+in-place 赋值改为
    bias*mask。in-place 赋值对 buffer 无梯度影响，但 alibi_learnable=True 时
    bias 是 Parameter，mask 乘法须让梯度正确回流到未屏蔽列（mem_cols 之后）。
    若 mask 广播维度错，梯度可能全零或回流到错误列。

    本测试验证：
      1) 梯度非零（mask 未切断梯度）
      2) 梯度仅在未屏蔽列非零（mask 正确屏蔽了记忆列的梯度）
    """
    torch.manual_seed(42)
    attn = SlidingWindowCausalSelfAttention(dim=32, num_heads=2, alibi=True,
                                            alibi_learnable=True)
    Tq, Tkv, mem_cols = 4, 8, 3
    bias = attn._alibi_bias(Tq, Tkv, device=torch.device('cpu'),
                            start_pos=0, mem_cols=mem_cols)
    assert bias is not None, "alibi=True 时 _alibi_bias 不应返回 None"
    # 反向传播
    bias.sum().backward()
    grad = attn.alibi_slopes.grad
    assert grad is not None, "梯度未回流到 alibi_slopes"
    assert grad.abs().sum() > 0, "梯度全零——mask 可能切断了梯度流"
    # 梯度应等于未屏蔽列的距离和（屏蔽列梯度为 0，因 bias*0=0 对 slopes 无梯度贡献）
    # bias[h,i,j] = -slopes[h] * dist[i,j] * mask[j]
    # d(sum)/d(slopes[h]) = -sum_{i,j} dist[i,j] * mask[j]
    # 对屏蔽列 (j<mem_cols) mask=0，无贡献；对未屏蔽列 (j>=mem_cols) mask=1，有贡献
    # 故每个 head 的梯度应严格为负（-正数和），且非零
    for h in range(2):
        assert grad[h].item() < 0, \
            f"head {h} 梯度应为负（-dist 和），实际 {grad[h].item()}"


def test_alibi_learnable_mem_cols_zero_gradient_for_masked_columns():
    """缺口3：记忆列（前 mem_cols 列）对斜率的梯度贡献应为零。

    抓住：bias*mask 中 mask=0 的列对 bias 的贡献为 0，故对 slopes 的梯度
    也应为 0。若 mask 未正确应用（如广播错位），屏蔽列会有非零梯度贡献。

    方法：固定 Tkv=8，分别计算 mem_cols=3（有 mask）和 mem_cols=0（无 mask）
    的梯度。两者用同一距离矩阵，故差值应严格等于屏蔽列（j<mem_cols）的
    梯度贡献——若 mask 生效，屏蔽列贡献为 0，故 grad_with_mem 应仅来自
    未屏蔽列（j>=mem_cols）。本测试验证差值等于手动计算的屏蔽列贡献。"""
    torch.manual_seed(42)
    attn = SlidingWindowCausalSelfAttention(dim=32, num_heads=2, alibi=True,
                                            alibi_learnable=True)
    Tq, Tkv, mem_cols = 4, 8, 3

    # 含记忆列（mem_cols>0）：梯度仅来自未屏蔽列 j=mem_cols..Tkv-1
    bias_with_mem = attn._alibi_bias(Tq, Tkv, device=torch.device('cpu'),
                                     start_pos=0, mem_cols=mem_cols)
    bias_with_mem.sum().backward()
    grad_with_mem = attn.alibi_slopes.grad.clone()
    attn.alibi_slopes.grad = None

    # 不含记忆列（mem_cols=0，同样 Tkv=8）：梯度来自所有列 j=0..Tkv-1
    bias_full = attn._alibi_bias(Tq, Tkv, device=torch.device('cpu'),
                                 start_pos=0, mem_cols=0)
    bias_full.sum().backward()
    grad_full = attn.alibi_slopes.grad.clone()
    attn.alibi_slopes.grad = None

    # mask 生效时 grad_with_mem 应小于 grad_full（绝对值，因屏蔽列无贡献）
    assert grad_with_mem.abs().sum() < grad_full.abs().sum(), \
        f"含 mask 梯度 |{grad_with_mem.tolist()}| 应小于无 mask 梯度 |{grad_full.tolist()}|"

    # 差值应严格等于屏蔽列（j<mem_cols）的梯度贡献
    # 屏蔽列贡献 = -sum_{i=0..Tq-1} sum_{j=0..mem_cols-1} |i-j|（每个 head 相同）
    qpos = torch.arange(Tq).unsqueeze(1)
    kpos = torch.arange(Tkv).unsqueeze(0)
    dist = (qpos - kpos).abs()
    expected_masked_contrib = -dist[:, :mem_cols].sum().item()
    diff = grad_full - grad_with_mem
    for h in range(2):
        assert abs(diff[h].item() - expected_masked_contrib) < 1e-6, \
            f"head {h} 梯度差值 {diff[h].item()} != 屏蔽列预期贡献 {expected_masked_contrib}" \
            "——mask 可能未正确屏蔽记忆列的梯度"


def test_alibi_learnable_mem_cols_mask_broadcast_correct():
    """缺口3：mask 应沿最后一维（Tkv）正确广播，不破坏 (1,H,Tq,Tkv) 形状。

    抓住：bias 形状 (1,H,Tq,Tkv)，mask = (arange(Tkv) >= mem_cols) 形状 (Tkv,)。
    bias * mask 要求 mask 沿最后一维广播。若 mask 误 reshape 为 (1,1,Tq,Tkv)
    或 (Tkv,1) 会广播错位——前 mem_cols 行（而非列）被清零。
    本测试直接验证 bias 的前 mem_cols 列（沿最后一维）为 0。"""
    torch.manual_seed(42)
    attn = SlidingWindowCausalSelfAttention(dim=32, num_heads=2, alibi=True,
                                            alibi_learnable=True)
    Tq, Tkv, mem_cols = 5, 7, 4
    bias = attn._alibi_bias(Tq, Tkv, device=torch.device('cpu'),
                            start_pos=0, mem_cols=mem_cols)
    # 前 mem_cols 列（最后一维）应为 0
    assert torch.allclose(bias[..., :mem_cols], torch.zeros_like(bias[..., :mem_cols])), \
        f"前 {mem_cols} 列应为 0（记忆列无 ALiBi 偏置），实际含非零值"
    # 第 mem_cols 列之后应非零（主序列有位置偏置）
    assert not torch.allclose(bias[..., mem_cols:], torch.zeros_like(bias[..., mem_cols:])), \
        f"第 {mem_cols} 列之后应非零（主序列有 ALiBi 偏置），实际全零"
    # 验证是列方向屏蔽（非行方向）：不同查询位置 i 的屏蔽列数应相同
    for i in range(Tq):
        row_masked = bias[0, 0, i, :mem_cols].abs().max().item()
        assert row_masked < 1e-7, \
            f"查询位置 {i} 的前 {mem_cols} 列应全零（列方向屏蔽），实际 max={row_masked}"


# ============================================================================
# 缺口 4：DifferentialAttention 动态掩码因果正确性（commit 02d0e3e B4 修复）
# ============================================================================

def test_diff_attn_dynamic_mask_is_strictly_causal():
    """缺口4：T>max_seq_length 时动态扩展的掩码应严格上三角（因果）。

    抓住：commit 02d0e3e 的 B4 修复在 T>max_seq_length 时用
    torch.triu(ones, diagonal=1) 动态构造掩码。若 diagonal 写错（如 0 或 -1），
    掩码会包含对角线或下三角，让当前 token 看到自身或过去 token 的"未来"——
    输出仍有限但语义错误（信息泄漏）。本测试直接构造动态掩码并验证因果性。

    方法：比较 T>max_seq_length 路径与 T==max_seq_length 路径在相同输入
    前缀下的输出——若掩码因果性正确，两者应一致（掩码仅是缓存 vs 重算的差异）。
    """
    torch.manual_seed(42)
    max_seq = 8
    m = _small(mixer='diff', max_seq_length=max_seq)
    m.eval()
    # T=16 > max_seq_length=8（触发动态扩展）
    x_long = torch.randint(0, 200, (1, 16))
    # T=8 == max_seq_length（走 buffer 切片路径）
    x_short = x_long[:, :max_seq]
    with torch.no_grad():
        out_long_prefix = m(x_long)[:, :max_seq, :]
        out_short = m(x_short)
    # 前 max_seq 个 token 的输出应一致（掩码因果性保证：前缀不依赖未来）
    diff = (out_long_prefix - out_short).abs().max().item()
    assert diff < 1e-4, \
        f"动态扩展掩码因果性损坏：前 {max_seq} token 输出不一致（diff={diff:.2e}）" \
        "——可能 mask diagonal 错误导致信息泄漏"


def test_diff_attn_dynamic_mask_matches_manual_construction():
    """缺口4：动态扩展的掩码应与手动构造的上三角掩码数值一致。

    抓住：直接验证 B4 修复的 mask 构造逻辑。若 triu 的 diagonal 参数错，
    掩码值会不同。本测试用 monkey-patch 捕获动态扩展路径生成的掩码输出，
    与手动 torch.triu(ones, diagonal=1) 比较。"""
    torch.manual_seed(42)
    max_seq = 4
    attn = DifferentialAttention(dim=64, num_heads=4, max_seq_length=max_seq)
    attn.eval()

    # T=8 > max_seq_length=4，触发动态扩展
    T = 8
    x = torch.randn(1, T, 64)

    # 捕获动态扩展路径生成的掩码（捕获 triu 的输出，非输入）
    original_triu = torch.triu
    captured = {}

    def _capturing_triu(tensor, diagonal=0):
        result = original_triu(tensor, diagonal)
        if tensor.dim() == 4 and tensor.shape[-1] == T and tensor.shape[-2] == T:
            captured['mask'] = result.clone()
            captured['diagonal'] = diagonal
        return result

    # DifferentialAttention.forward 内部用 torch.triu 构造动态掩码
    torch.triu = _capturing_triu
    try:
        with torch.no_grad():
            attn(x)
    finally:
        torch.triu = original_triu

    assert 'mask' in captured, "未捕获到动态扩展的掩码——可能未触发 T>max_seq_length 路径"
    assert captured['diagonal'] == 1, \
        f"动态掩码 diagonal 应为 1（严格上三角，屏蔽未来），实际 {captured['diagonal']}"
    # 验证掩码是严格上三角：对角线及以下为 False，以上为 True
    mask = captured['mask']
    expected = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    assert torch.equal(mask[0, 0], expected), \
        "动态扩展掩码与手动 torch.triu(ones, diagonal=1) 不一致——因果性可能损坏"


def test_diff_attn_dynamic_mask_no_future_leakage():
    """缺口4：动态扩展掩码下，修改未来 token 不应影响当前 token 输出。

    抓住：这是因果性的端到端验证。若动态掩码非因果（如下三角泄漏），
    修改位置 t 之后的 token 会改变位置 t 的输出。本测试修改 x[0, 5:]
    验证 x[0, :5] 的输出不变。"""
    torch.manual_seed(42)
    max_seq = 4
    m = _small(mixer='diff', max_seq_length=max_seq)
    m.eval()
    x = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        out_orig = m(x)
        # 修改位置 5-7（"未来"）
        x_modified = x.clone()
        x_modified[0, 5:] = torch.randint(0, 200, (3,))
        out_modified = m(x_modified)
    # 位置 0-4 的输出应不变（因果性：不看未来）
    diff = (out_orig[:, :5, :] - out_modified[:, :5, :]).abs().max().item()
    assert diff < 1e-6, \
        f"修改未来 token 影响了当前位置输出（diff={diff:.2e}）——动态掩码因果性损坏"
