# -*- coding: utf-8 -*-
"""2026-09-28 架构审查·零风险批①的**补测试**（对应提交 320a685 中尚无回归的 5 项）。

`320a685` 当批只给 M2/M3/M9/M10 配了 `test_review_batch1.py`（18 项），
以下 5 项当时只靠"改完全量 pytest 仍绿"间接保护，没有直接断言：
  1. `temperature_applied` property 三处调用点统一（本文件 1-3）
  2. `ngram._logprob_cache` 512MB 字节预算（R40 只修了 `_orders_cache`）（4）
  3. `clip_grad_norm_dml` 用 `torch.where` 消除隐式 `.item()` CPU 同步（5）
  4. `chat.py --repetition-penalty` + 默认模型路径存在（6）
  5. `--igmcg-candidates 1` 不再静默缩 0.75x 温度（7）
  6. `prev_outputs` 死三元表达式清理（8，随手项，一并锁死防回潮）

不改任何数值路径；行为类断言用真实对象，"接线是否到位"类只能用源码断言
（脚本入口无单元接口，与 `test_round41.py` 同风格）。
"""
import inspect
import os
import re
import sys
import tempfile
from pathlib import Path

import torch

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from models.ngram import NGramModel
from models.transformer import TransformerModel
from models.data_utils import CharTokenizer

_CORPUS = [
    "中 国 人 民 生 活 幸 福",
    "中 国 梦 想 伟 大 复 兴",
    "人 民 当 家 作 主 权 利",
    "中 国 人 民 共 和 国 万 岁",
]


def _make_ngram(max_order=5):
    v = CharTokenizer(vocab_size=200)
    v.train(_CORPUS)
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    f.write("\n".join(_CORPUS * 5) + "\n")
    f.close()
    ng = NGramModel(v, f.name, max_order=max_order, smoothing=1.0, min_count=1)
    os.unlink(f.name)
    return v, ng


def _read(rel):
    return (_ROOT / rel).read_text(encoding='utf-8')


# ============================================================
# 1-3：temperature_applied 统一为 property
# ============================================================
def test_temperature_applied_reflects_runtime_switch():
    """property 必须同时看"配置开关"和"运行时开关"。

    修复前 generate.py 用 `enabled and _ngram_fusion_active`、transformer.py
    两处只用 `enabled` → `set_ngram_fusion_active(False)` 时 model.generate
    仍以为温度已应用，会**漏除一次 τ**（采样分布整体偏尖）。
    """
    v, ng = _make_ngram()
    m = TransformerModel(vocab_size=len(v), embedding_dim=64, num_heads=4, num_layers=2,
                         hidden_dim=128, max_seq_length=32,
                         ngram_fusion=True, ngram_model=ng,
                         gradient_checkpointing=False)
    m.eval()
    assert m.temperature_applied is True, "融合开着且未运行时关闭 → 应视为温度已应用"

    m.set_ngram_fusion_active(False)
    assert m.temperature_applied is False, "运行时关闭融合后，forward 不再除 τ，property 必须跟随"

    m.set_ngram_fusion_active(True)
    assert m.temperature_applied is True, "重新打开应回到 True"

    # 未启用融合的模型恒 False（forward 不走融合分支）
    plain = TransformerModel(vocab_size=len(v), embedding_dim=64, num_heads=4, num_layers=2,
                             hidden_dim=128, max_seq_length=32,
                             ngram_fusion=False, ngram_model=None,
                             gradient_checkpointing=False)
    assert plain.temperature_applied is False


def test_temperature_applied_matches_forward_branch():
    """property 与 forward 的真实温度分支必须**同式**，否则单一事实来源名不副实。"""
    src = _read('models/transformer.py')
    # property 定义
    prop = re.search(r'def temperature_applied\(self\)[^\n]*\n.*?return ([^\n]+)', src, re.S)
    assert prop, "temperature_applied property 未找到"
    prop_expr = prop.group(1)
    # forward 里唯一决定"主干 logits 是否已被除过 τ"的分支
    fwd = re.search(r'^(\s*)if (self\.ngram_fusion_enabled and [^\n]+):$', src, re.M)
    assert fwd, "forward 的 n-gram 融合分支未找到"
    fwd_expr = fwd.group(2)
    assert 'ngram_fusion_enabled' in prop_expr and '_ngram_fusion_active' in prop_expr
    assert 'ngram_fusion_enabled' in fwd_expr and '_ngram_fusion_active' in fwd_expr, (
        f"property({prop_expr!r}) 与 forward 分支({fwd_expr!r}) 不同式，两处会再次分叉")


def test_temperature_applied_used_at_all_call_sites():
    """三处采样调用点必须走 property，旧的 `getattr(self, 'ngram_fusion_enabled')` 不得残留。"""
    src = _read('models/transformer.py')
    assert 'temperature_applied=getattr(self, \'ngram_fusion_enabled\'' not in src, \
        "transformer.py 仍残留旧式 temperature_applied=getattr(ngram_fusion_enabled)"
    assert src.count('temperature_applied=self.temperature_applied') >= 2, \
        "model.generate 的两处采样调用（sample_step / _decode_one_step）都应传 property"

    gen = _read('scripts/generate.py')
    assert 'model.temperature_applied' in gen, \
        "generate.py 应缓存 model.temperature_applied 后传入批量解码的每个候选"
    assert '_temp_applied' in gen


# ============================================================
# 4：n-gram _logprob_cache 字节预算（R40 漏网）
# ============================================================
def test_ngram_logprob_cache_byte_budget():
    """`_logprob_cache` 必须有字节预算，而不只是 8192 条数上限。

    R40 只给 `_orders_cache` 加了 512MB 预算；`_logprob_cache` 每条是 (V,)
    fp32 向量，V=50000 时单条 200KB × 8192 ≈ 1.6GB。
    """
    v, ng = _make_ngram()
    assert ng._logprob_cache_byte_budget == 512 * 1024 * 1024, "默认字节预算应为 512MB"
    assert ng._logprob_cache_bytes == 0, "新建模型缓存字节计数应为 0"
    assert len(ng._logprob_cache) == 0

    # 正常写入应记账
    ng._vec_for_ctx(None, None, 'cpu')
    assert len(ng._logprob_cache) == 1, "第一次写入应命中缓存槽"
    assert ng._logprob_cache_bytes > 0, "写入后必须累加字节数"
    one_key = next(iter(ng._logprob_cache))
    one_bytes = ng._logprob_cache[one_key].numel() * ng._logprob_cache[one_key].element_size()
    assert ng._logprob_cache_bytes == one_bytes, "记账字节数应等于实际缓存张量大小"

    # 预算压到 0 → 每写即清（与 test_round38 的 _orders_cache 口径一致）
    # 换一个 cache_key 才会走到写入分支（同 key 在 :406 提前 return）
    ng._logprob_cache_byte_budget = 0
    ng._vec_for_ctx(None, 1, 'cpu')
    assert len(ng._logprob_cache) == 1, "预算为 0 时缓存应每写即清（恒 1 条）"
    assert ng._logprob_cache_bytes == one_bytes


# ============================================================
# 5：clip_grad_norm_dml 不得有隐式 .item() 同步
# ============================================================
def test_clip_grad_norm_dml_uses_device_side_where():
    """`if clip_coef < 1.0` 对 0-dim tensor 做 bool 转换即触发 CPU 同步。

    该同步正是 R38 引入本函数要消除的开销（~0.5-2ms/步），用 `torch.where`
    在设备侧选 scale 才不破坏本函数的存在意义。
    """
    from scripts.train import clip_grad_norm_dml
    src = inspect.getsource(clip_grad_norm_dml)

    # 函数体（去掉 docstring）里不得出现标量同步/裸 if 判 tensor
    body = src.split('"""', 2)[-1]
    assert 'torch.where' in body, "应使用 torch.where 在设备侧选 scale"
    assert '.item()' not in body, "clip_grad_norm_dml 不得调用 .item()"
    assert not re.search(r'^\s*if\s+clip_coef\b', body, re.M), \
        "`if clip_coef < 1.0` 会触发隐式 CPU 同步"

    # 数值等价冒烟：min(max_norm/(norm+1e-6),1) 缩放
    p = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    p.grad = torch.full((4,), 3.0)
    tn = clip_grad_norm_dml([p], max_norm=1.0, foreach_norm=False)
    assert tn is not None and torch.isfinite(tn)
    expected = (4 * 9.0) ** 0.5
    assert abs(float(tn) - expected) < 1e-5, f"总范数应为 {expected}，实得 {float(tn)}"
    scale = 1.0 / (expected + 1e-6)
    assert abs(float(p.grad[0]) - 3.0 * scale) < 1e-5, \
        "梯度应被原地乘 min(max_norm/(norm+eps), 1)"


# ============================================================
# 6：chat.py --repetition-penalty 已定义且默认模型路径存在
# ============================================================
def test_chat_has_repetition_penalty_and_valid_default_model():
    """README/MODEL_USAGE_GUIDE/TUNING_GUIDE 三处宣传了 --repetition-penalty，
    argparse 却从未定义 → 传了就 `unrecognized arguments` 退出码 2。"""
    src = _read('scripts/chat.py')
    assert "--repetition-penalty" in src, "chat.py 应定义 --repetition-penalty"
    assert "args.repetition_penalty" in src, "解析结果应真正传进生成调用"

    m = re.search(r"DEFAULT_MODEL\s*=\s*str\(project_root\s*/\s*'([^']+)'\s*/\s*'([^']+)'\)", src)
    assert m, "DEFAULT_MODEL 常量未找到"
    rel = f"{m.group(1)}/{m.group(2)}"
    assert (_ROOT / m.group(1) / m.group(2)).exists(), f"默认模型路径不存在：{rel}"


# ============================================================
# 7：--igmcg-candidates 1 不再静默缩温度
# ============================================================
def test_igmcg_single_candidate_keeps_base_temperature():
    """原式 `base_temp * (0.75 + 0.6*k/max(1,N-1))` 在 N=1 时退化为 0.75x，
    单候选本该"原样用 base_temp"，却静默把温度降了 25%。"""
    src = _read('scripts/generate.py')
    assert re.search(r'if\s+num_candidates\s*<=\s*1\s*', src), \
        "generate_igmcg 应有 N<=1 的短路分支"
    assert re.search(r'temps\s*=\s*\[base_temp\]\s*if\s+num_candidates\s*<=\s*1\s*else', src), \
        "N<=1 必须直接取 [base_temp]，不得走 0.75x 缩放公式"
    # 缩放公式只应在 else 分支（多候选）出现
    assert '0.75 + 0.6 *' in src


# ============================================================
# 8：prev_outputs 死三元表达式（随手清理项，锁死防回潮）
# ============================================================
def test_prev_outputs_dead_ternary_removed():
    """原 `[] if self.cross_layer_routing else []` 两分支同值 = 死代码。"""
    src = _read('models/transformer.py')
    assert 'if self.cross_layer_routing else' not in src, \
        "prev_outputs 的两分支同值三元表达式应回归为无条件 `[]`"
    assert re.search(r'prev_outputs:\s*List\[torch\.Tensor\]\s*=\s*\[\]', src)
