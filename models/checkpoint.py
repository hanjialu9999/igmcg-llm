from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn


# weights_only=True 仅允许的白名单全局符号：均为张量/numpy 反序列化的官方重建函数。
# 只放行这些固定符号，杜绝被恶意 .pt 诱导放行任意全局符号（即 CVE-2026-24747 类绕过）。
def _safe_getattr(dotted: str):
    """逐段 getattr，任意一段缺失都返回 None（兼容 numpy 1.x/2.x 的路径差异）。"""
    parts = dotted.split('.')
    obj = __import__(parts[0])
    for attr in parts[1:]:
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj


def _build_safe_globals():
    import numpy as np
    cands = [
        getattr(torch._utils, '_rebuild_device_tensor_from_numpy', None),
        _safe_getattr('numpy.core.multiarray._reconstruct'),   # numpy 1.x
        _safe_getattr('numpy._core.multiarray._reconstruct'),   # numpy 2.x
        getattr(np, 'ndarray', None),
        getattr(np, 'dtype', None),
    ]
    return [g for g in cands if g is not None]


_SAFE_GLOBALS = _build_safe_globals()
for _g in _SAFE_GLOBALS:
    try:
        torch.serialization.add_safe_globals([_g])
    except Exception:
        pass


def safe_torch_load(path, map_location: str = 'cpu'):
    """以 weights_only=True 加载 checkpoint。

    白名单全局符号（仅官方张量/numpy 重建函数，见 _SAFE_GLOBALS）已在模块导入时放行；
    若仍遇到非白名单的全局符号，说明该文件可能不是可信 checkpoint，直接抛错拒绝加载，
    避免被诱导放行危险全局符号（CVE-2026-24747 类绕过）。"""
    return torch.load(path, map_location=map_location, weights_only=True)


def build_ngram_model(vocab, model_config: Dict[str, Any]):
    """从配置重建统计 n-gram 缓冲（阶段8.1/8.7 神经融合用）。

    避免在 generate.py / train.py 各自重复构建逻辑。配置关闭 ngram_fusion 时返回 None。
    """
    if not model_config.get('ngram_fusion', False):
        return None
    try:
        from models.ngram import NGramModel
        corpus = model_config.get('ngram_corpus', 'data/pretrain_corpus/merged.txt')
        vocab_size = model_config.get('vocab_size', getattr(vocab, 'vocab_size', None))
        model = NGramModel(vocab, corpus, max_order=10, smoothing=1.0,
                           vocab_size=vocab_size)
        print(f"[n-gram 融合] 已从 {corpus} 重建统计缓冲（推理对齐训练分布）")
        return model
    except Exception as e:
        print(f"[n-gram 融合] 重建失败，已跳过：{e}")
        return None


def load_model(model_path, vocab_path, device: str = 'cpu',
               quantize: bool = False, compile_model: bool = False):
    """Load trained model and vocabulary.

    quantize=True 时对 Linear 层做 int8 动态量化（仅 CPU 有效）：用更低带宽的量化权重做
    矩阵乘，降低内存带宽与功耗，对生成质量几乎无损。AMD DML 设备无量化算子支持，会自动跳过。
    compile_model=True 时对模型做 torch.compile（CUDA/CPU 有效）：融合 RMSNorm/RoPE/MatMul 等
    算子在自回归解码上通常带来 1.5~3× 吞吐提升；DML 设备自动跳过。
    """
    # Load vocabulary（复用 config_loader.load_vocab，正确处理 BPE/char 词表的 merges 等字段，
    # 与训练期保存逻辑对称；避免手写 Vocabulary() 重建丢失 bpe/char 信息导致分布错位）
    from models.config_loader import load_vocab, build_model
    vocab = load_vocab(vocab_path)

    # Load model（map_location 用 'cpu'，加载后再由下方 .to(device) 搬运到目标设备）。
    # 注意：DML 设备下若直接用 torch.device('privateuseone:0') 作 map_location，
    # torch.load 内部会调用 torch_directml.device(torch.device) 触发 TypeError，
    # 导致 DML 推理无法加载权重；统一先加载到 CPU 可绕开该问题。
    checkpoint = safe_torch_load(model_path, map_location='cpu')

    # Load config from separate YAML file (for weights_only=True compatibility)
    model_path_obj = Path(model_path)
    config_path = model_path_obj.parent / f"{model_path_obj.stem}_config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            model_config = yaml.safe_load(f)
    else:
        # Fallback for old checkpoints with config embedded
        model_config = checkpoint.get('config', {
            'vocab_size': checkpoint.get('vocab_size', 12000),
            'embedding_dim': 128,
            'num_heads': 4,
            'num_layers': 2,
            'hidden_dim': 256,
            'max_seq_length': 32,
            'dropout': 0.1
        })

    # 复用 build_model（正确传递所有参数：char_merge/memory/ssm/rope 等），避免手动列参数遗漏。
    # strict=False 兼容旧权重（旧 checkpoint 可能缺少 qk_norm/attn_temp/residual_gate 等新增参数）。
    _ngram_model = build_ngram_model(vocab, model_config)
    model = build_model({'model': model_config}, device=device, ngram_model=_ngram_model)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    # 架构型参数（hybrid_mix / ngram_gate）缺失/多余属静默质量风险，主动告警而非 strict 吞掉。
    _arch_keys = ['hybrid_mix', 'ngram_gate']
    _missing = [k for k in model.state_dict().keys() if any(ak in k for ak in _arch_keys)
                and k not in checkpoint['model_state_dict']]
    _extra = [k for k in checkpoint['model_state_dict'].keys() if any(ak in k for ak in _arch_keys)
              and k not in model.state_dict()]
    if _missing:
        print(f"[warn] checkpoint 缺少架构参数（缺失将用随机初始化，结果可能异常）：{_missing}")
    if _extra:
        print(f"[warn] checkpoint 含模型未定义的架构参数（已忽略）：{_extra}")
    # 阶段8.2：推理期静态剪枝——若训练时启用 layer_skip，加载后按阈值把"几乎必跳过"的层
    # 真正移除（实现推理提速，否则 skip 门控只训不用、纯死参数）。阈值由 prune_threshold
    # 控制，默认 0.5；设 <=0 则取消剪枝（全保留）。
    if model_config.get('layer_skip', False):
        _pt = float(model_config.get('prune_threshold', 0.5))
        try:
            _pruned = model.prune_layers(_pt)
            if _pruned:
                print(f"[剪枝] 已静态移除 {len(_pruned)} 层（索引 {_pruned}），推理提速生效")
        except Exception as e:
            print(f"[warn] 推理期剪枝失败，已跳过：{e}")
    model.eval()

    if quantize and getattr(device, 'type', None) != 'dml':
        # 量化返回新模型对象，必须重新赋值；DML 无量化算子支持，已在上面跳过
        try:
            model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        except Exception as e:
            print(f"[warn] int8 动态量化不可用，回退 fp32：{e}")

    if compile_model and getattr(device, 'type', None) != 'dml':
        try:
            model = torch.compile(model, dynamic=True)
        except Exception as e:
            print(f"[warn] torch.compile 不可用，回退 eager 模式：{e}")

    return model, vocab
