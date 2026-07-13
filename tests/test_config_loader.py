import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config_loader import load_config, build_model, load_vocab, load_generation_config


def test_load_config():
    """Test loading YAML configs."""
    config = load_config('configs/pretrain.yaml')
    assert 'model' in config
    assert 'training' in config
    assert config['model']['vocab_size'] == 12000
    # layer_plan is optional, defaults to None (all attn)
    layer_plan = config['model'].get('layer_plan')
    assert layer_plan is None or layer_plan == 'attn,ssm,attn,ssm,attn,ssm'
    
    hybrid_config = load_config('configs/config_hybrid.yaml')
    assert hybrid_config['model']['layer_plan'] == 'attn,ssm,attn,ssm,attn,ssm'
    print("✅ test_load_config passed")


def test_build_model_from_config():
    """Test building model from config."""
    config = load_config('configs/pretrain.yaml')
    model = build_model(config, device='cpu')
    assert model is not None
    assert model.vocab_size == config['model']['vocab_size']
    assert model.embedding_dim == config['model']['embedding_dim']
    assert model.layer_plan == ['attn'] * 6
    print("✅ test_build_model_from_config passed")


def test_load_vocab():
    """Test loading vocabulary from JSON."""
    vocab = load_vocab('checkpoints/vocab.json')
    assert vocab is not None
    assert len(vocab.word2idx) > 0
    assert len(vocab.idx2word) > 0
    assert vocab.pad_idx == 0
    print("✅ test_load_vocab passed")


def test_load_generation_config():
    """Test loading generation config from JSON."""
    config = load_generation_config('chat_config.json')
    assert 'temperature' in config
    assert 'top_k' in config
    assert 'repetition_penalty' in config
    assert isinstance(config['temperature'], float)
    print("✅ test_load_generation_config passed")


if __name__ == '__main__':
    test_load_config()
    test_build_model_from_config()
    test_load_vocab()
    test_load_generation_config()
    print("\n🎉 All config tests passed!")