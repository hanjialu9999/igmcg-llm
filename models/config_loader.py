import json
from pathlib import Path

import torch
import yaml

from models.data_utils import Vocabulary
from models.transformer import TransformerModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path='config/config.yaml'):
    """Load configuration from a YAML file (resolved relative to project root)."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_model(config, device=None):
    """Build a TransformerModel from a loaded config dict (config['model'])."""
    mc = config['model']
    model = TransformerModel(
        vocab_size=mc['vocab_size'],
        embedding_dim=mc['embedding_dim'],
        num_heads=mc['num_heads'],
        num_layers=mc['num_layers'],
        hidden_dim=mc['hidden_dim'],
        max_seq_length=mc['max_seq_length'],
        dropout=mc['dropout'],
    )
    if device is not None:
        model = model.to(device)
    return model


def load_vocab(vocab_path='checkpoints/vocab.json'):
    """Load a Vocabulary object from a previously saved vocab.json."""
    path = Path(vocab_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)

    vocab = Vocabulary()
    vocab.word2idx = vocab_data['word2idx']
    vocab.idx2word = vocab_data['idx2word']
    return vocab


def load_generation_config(path='chat_config.json'):
    """Load generation (dialogue) parameters from chat_config.json."""
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return json.load(f)
