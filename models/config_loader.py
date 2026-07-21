from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from models.model_config import ModelConfig
from models.transformer import TransformerModel
from models.constants import MASK_FILL_VALUE, ROPE_BASE

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str = 'configs/pretrain.yaml') -> Dict[str, Any]:
    """Load configuration from a YAML file (resolved relative to project root)."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_model(config: Dict[str, Any], device: Optional[torch.device] = None,
                ngram_model=None) -> TransformerModel:
    """Build a TransformerModel from a loaded config dict (config['model']).

     兼容混合架构：读取 layer_plan / ssm_* / attn_window / attn_rel_bias 等可选字段。
     ngram_model：已构建的统计 NGramModel 实例（阶段8.1 神经融合用），由调用方传入，
     避免在 config_loader 内 import scripts.generate（与 transformer 循环依赖）；默认 None。
    """
    mc = config['model']
    # 阶段8：统一记忆预算——用单一 memory_budget(0,1] 推导 window/记忆槽数/检索 topk，
    # 让三者按比例自平衡（预算小→窗口小+记忆少+检索窄；预算大→反之）。优先级低于各自显式配置。
    budget = mc.get('memory_budget', None)
    if budget is not None:
        budget = float(budget)
        if budget <= 0 or budget > 1.0:
            raise ValueError(f"memory_budget must be in (0, 1], got {budget}")
        if 'attn_window' not in mc:
            mc['attn_window'] = int(round(64 * budget))
        if 'memory_size' not in mc:
            mc['memory_size'] = int(round(64 * budget))
        if 'memory_retrieval_topk' not in mc:
            mc['memory_retrieval_topk'] = int(round(32 * budget))
        if 'memory_retrieval' not in mc:
            mc['memory_retrieval'] = budget > 0.3
    # 用 ModelConfig schema 校验（替代 42 个 mc.get() 散参数）
    model_cfg = ModelConfig.from_dict(mc)
    # 机制组合校验（需在 ModelConfig 之上额外检查 layer_plan 交互）
    _mixer = mc.get('mixer', 'attn')
    if _mixer in ('hybrid', 'attn_linear') and mc.get('layer_plan', None) is not None:
        layer_plan = mc.get('layer_plan', None)
        hybrid_blocks = [p for p in layer_plan.replace(',', ' ').split() if p == 'hybrid']
        if hybrid_blocks:
            import warnings
            warnings.warn(
                "mixer='hybrid' 与 layer_plan 中的 'hybrid' 块组合时，线性注意力 mixer 仅在 "
                "attn 块生效、hybrid(SSM) 块内的 linear_attn/mixer_gate 为死参数（不更新）。"
                "如需全局线性融合，请将对应层改为 'attn'；否则 hybrid 块仅做 attn+ssm。",
                stacklevel=2,
            )
    model = TransformerModel.from_config(model_cfg, ngram_model=ngram_model)
    if device is not None:
        model = model.to(device)
        model.tie_weights()
    return model


def load_vocab(vocab_path: str = 'checkpoints/vocab.json') -> 'BaseTokenizer':
    """Load a BaseTokenizer (CharTokenizer / BPETokenizer) from a saved vocab.json.

    词表 JSON 须带 `char`/`bpe` 标志（build_char_vocab.py / build_bpe_vocab.py /
    save_final_model 均已写出）；缺标志时按字符级 CharTokenizer 重建，避免旧双轨
    残留的 Vocabulary 分支再被使用（统一为单一 BaseTokenizer 系）。
    """
    from models.data_utils import BPETokenizer, CharTokenizer
    path = Path(vocab_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)

    if vocab_data.get('char'):
        vocab = CharTokenizer()
    else:
        # 默认按 BPE 路径（含缺标志的情况）重建，char/bpe 统一走 BaseTokenizer
        vocab = BPETokenizer()
    vocab.word2idx = vocab_data['word2idx']
    vocab.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
    vocab.merges = [tuple(m) for m in vocab_data.get('merges', [])]
    vocab.special_tokens = vocab_data.get('special_tokens', vocab.special_tokens)
    # 重建 byte_tokens（用于 decode 还原字节级 token）
    vocab.byte_tokens = [f'{BPETokenizer.BYTE_PREFIX}{b}' for b in range(256)]
    vocab.vocab_size = len(vocab.word2idx)
    vocab._symbol_cap = vocab.vocab_size - len(vocab.special_tokens) - 256
    return vocab


def load_generation_config(path: str = 'chat_config.json') -> Dict[str, Any]:
    """加载对话（生成）参数；配置文件缺失时回退到默认配置，避免运行时尚未生成配置就崩溃。"""
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        # 默认生成参数（与 tools/dialogue_interactive.py 初始配置一致）
        return {
            'temperature': 0.65,
            'top_k': 40,
            'repetition_penalty': 2.0,
            'min_new_tokens': 10,
            'max_new_tokens': 100,
            'context_rounds': 3,
        }
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return json.load(f)