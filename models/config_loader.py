from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from models.data_utils import Vocabulary
from models.transformer import TransformerModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str = 'configs/pretrain.yaml') -> Dict[str, Any]:
    """Load configuration from a YAML file (resolved relative to project root)."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_model(config: Dict[str, Any], device: Optional[torch.device] = None) -> TransformerModel:
    """Build a TransformerModel from a loaded config dict (config['model']).

     兼容混合架构：读取 layer_plan / ssm_* / attn_window / attn_rel_bias 等可选字段。
    """
    mc = config['model']
    model = TransformerModel(
        vocab_size=mc['vocab_size'],
        embedding_dim=mc['embedding_dim'],
        num_heads=mc['num_heads'],
        num_layers=mc['num_layers'],
        hidden_dim=mc['hidden_dim'],
        max_seq_length=mc['max_seq_length'],
        dropout=mc.get('dropout', 0.0),
        tie_weights=mc.get('tie_weights', True),
        gradient_checkpointing=mc.get('gradient_checkpointing', True),
        layer_plan=mc.get('layer_plan', None),
        ssm_d_state=mc.get('ssm_d_state', 16),
        ssm_d_inner_factor=mc.get('ssm_d_inner_factor', 1),
        ssm_dt_rank=mc.get('ssm_dt_rank', None),
        ssm_conv_kernel=mc.get('ssm_conv_kernel', 3),
        ssm_dt_proj_bias_init=mc.get('ssm_dt_proj_bias_init', 0.1),
        ssm_a_log_init_range=mc.get('ssm_a_log_init_range', [-1, 1]),
        ssm_D_init=mc.get('ssm_D_init', 1.0),
        attn_window=mc.get('attn_window', 0),
        attn_rel_bias=mc.get('attn_rel_bias', False),
        rope_base=mc.get('rope_base', 10000.0),
        rope_max_len=mc.get('rope_max_len', 4096),
        mask_fill_value=mc.get('mask_fill_value', -1e9),
        # 架构增强（默认全开：2026-07-14 起；旧权重门控 init=1.0 仍兼容，但开启后需重新训练以生效）
        qk_norm=mc.get('qk_norm', True),
        attn_temp=mc.get('attn_temp', True),
        residual_gate=mc.get('residual_gate', True),
        hybrid_gate=mc.get('hybrid_gate', True),
    )
    if device is not None:
        model = model.to(device)
        # 使用模型内置的 tie_weights() 方法（to() 已自动调用，双重保险）
        model.tie_weights()
    return model


def load_vocab(vocab_path: str = 'checkpoints/vocab.json') -> Vocabulary:
    """Load a Vocabulary object from a previously saved vocab.json."""
    path = Path(vocab_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)

    if vocab_data.get('bpe'):
        # BPE 词表：用 BPETokenizer 重建并恢复合并规则
        from models.data_utils import BPETokenizer
        vocab = BPETokenizer()
        vocab.word2idx = vocab_data['word2idx']
        vocab.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
        vocab.merges = [tuple(m) for m in vocab_data.get('merges', [])]
        vocab.special_tokens = vocab_data.get('special_tokens', vocab.special_tokens)
        return vocab

    vocab = Vocabulary()
    vocab.word2idx = vocab_data['word2idx']
    vocab.idx2word = vocab_data['idx2word']
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