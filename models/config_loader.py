from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from models.data_utils import Vocabulary
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
        if 'attn_window' not in mc:
            mc['attn_window'] = int(round(64 * budget))
        if 'memory_size' not in mc:
            mc['memory_size'] = int(round(64 * budget))
        if 'memory_retrieval_topk' not in mc:
            mc['memory_retrieval_topk'] = int(round(32 * budget))
        if 'memory_retrieval' not in mc:
            mc['memory_retrieval'] = budget > 0.3
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
        rope_base=mc.get('rope_base', ROPE_BASE),
        # RoPE/注意力缓冲区长度：默认与 max_seq_length 一致（指向同一"模型最大序列长"概念），
        # 仅当显式配置 rope_max_len 时才覆盖——避免未配置时悄悄用 4096 而与 max_seq_length 发散。
        rope_max_len=mc.get('rope_max_len', mc['max_seq_length']),
        mask_fill_value=mc.get('mask_fill_value', MASK_FILL_VALUE),
        # 阶段5：可学习 RoPE 频率 + ALiBi 线性位置偏置（默认关，向后兼容）
        rope_learnable=mc.get('rope_learnable', False),
        alibi=mc.get('alibi', False),
        # 阶段6：可学习跳过层（默认关，向后兼容；开启需重训）
        layer_skip=mc.get('layer_skip', False),
        # 阶段6：可学习滑动窗口（默认关，向后兼容；开启需重训）
        learn_window=mc.get('learn_window', False),
        window_base=mc.get('window_base', 64),
        # 阶段7：token mixer 选择（attn | linear | attn_linear，默认 attn 向后兼容）。
        # 旧字符串 'hybrid' 等价于 'attn_linear'（attn+线性注意力并行），由 TransformerBlock 归一。
        mixer=('attn_linear' if mc.get('mixer', 'attn') == 'hybrid' else mc.get('mixer', 'attn')),
        linear_attn_feature=mc.get('linear_attn_feature', 'relu'),
        linear_attn_head_dim=mc.get('linear_attn_head_dim', None),
        # 架构增强（默认全开：2026-07-14 起；旧权重门控 init=1.0 仍兼容，但开启后需重新训练以生效）
        qk_norm=mc.get('qk_norm', True),
        attn_temp=mc.get('attn_temp', True),
        residual_gate=mc.get('residual_gate', True),
        hybrid_gate=mc.get('hybrid_gate', True),
        hybrid_single_gate=mc.get('hybrid_single_gate', False),
        char_merge=mc.get('char_merge', False),
        char_merge_kernel=mc.get('char_merge_kernel', 3),
        char_merge_dropout=mc.get('char_merge_dropout', 0.0),
        # 阶段2 可学习压缩记忆 + 阶段3 可学习检索/稀疏 + 阶段4 可学习遗忘
        memory_size=mc.get('memory_size', 0),
        memory_comp_dim=mc.get('memory_comp_dim', 32),
        memory_retrieval=mc.get('memory_retrieval', False),
        memory_sparse_topk=mc.get('memory_sparse_topk', 0),
        memory_forget=mc.get('memory_forget', False),
        memory_product_key=mc.get('memory_product_key', False),
        memory_retrieval_full=mc.get('memory_retrieval_full', False),
        memory_retrieval_topk=mc.get('memory_retrieval_topk', 32),
        # 阶段8.1/8.7：n-gram 神经融合 + IGMCG 2.0（默认关，向后兼容；开启需传入 ngram_model 实例）
        ngram_fusion=mc.get('ngram_fusion', False),
        ngram_model=ngram_model,
        ngram_gate_scale=float(mc.get('ngram_gate_scale', 1.0)),
        igmcg=bool(mc.get('igmcg', False)),
    )
    # 机制组合校验：mixer='attn_linear'（旧名 'hybrid'，attn+线性注意力并行）仅在
    # block_type='attn' 的层真正融合线性注意力；若 layer_plan 含 'hybrid'（SSM×注意力混合块），
    # 该块不会调用 linear_attn/mixer_gate，导致二者成为永不更新的死参数（占显存与 checkpoint 体积）。
    # 发出告警避免静默无效训练。
    if mc.get('mixer', 'attn') in ('hybrid', 'attn_linear') and mc.get('layer_plan', None) is not None:
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

    if vocab_data.get('bpe') or vocab_data.get('char'):
        # BPE / 字符级词表：用 BPETokenizer 或 CharTokenizer 重建
        from models.data_utils import BPETokenizer, CharTokenizer
        if vocab_data.get('char'):
            vocab = CharTokenizer()
        else:
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

    vocab = Vocabulary()
    vocab.word2idx = vocab_data['word2idx']
    vocab.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
    # 恢复 special_tokens（save_final_model 写出；旧 vocab.json 无此键则用默认）
    _st = vocab_data.get('special_tokens')
    if _st is not None:
        vocab.special_tokens = list(_st)
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