import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer import (
    TransformerModel,
    MambaSSM,
    RotaryEmbedding,
    SlidingWindowCausalSelfAttention,
)
from models.config_loader import load_config, build_model


def test_model_creation():
    """Test TransformerModel creation with default (attn-only) config."""
    config = load_config('configs/pretrain.yaml')
    model = build_model(config, device='cpu')
    assert model is not None
    assert sum(p.numel() for p in model.parameters()) > 0
    # tie_weights should work
    assert model.embedding.weight is model.output_head.weight
    print("✅ test_model_creation passed")


def test_hybrid_model_creation():
    """Test TransformerModel with hybrid (attn+ssm) layer_plan."""
    config = load_config('configs/config_hybrid.yaml')
    model = build_model(config, device='cpu')
    assert model is not None
    assert 'ssm' in model.layer_plan
    # tie_weights should work
    assert model.embedding.weight is model.output_head.weight
    print("✅ test_hybrid_model_creation passed")


def test_forward_pass():
    """Test forward pass without cache."""
    config = load_config('configs/pretrain.yaml')
    model = build_model(config, device='cpu')
    model.eval()
    x = torch.randint(0, config['model']['vocab_size'], (2, 10))
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 10, config['model']['vocab_size'])
    print("✅ test_forward_pass passed")


def test_forward_with_cache():
    """Test forward pass with KV-cache (attn-only model)."""
    config = load_config('configs/pretrain.yaml')
    model = build_model(config, device='cpu')
    model.eval()
    x = torch.randint(0, config['model']['vocab_size'], (2, 5))
    with torch.no_grad():
        logits, past = model(x, use_cache=True)
    assert logits.shape == (2, 5, config['model']['vocab_size'])
    assert past is not None
    assert len(past) == config['model']['num_layers']
    # Each layer: (attn_kv, None) for attn-only
    assert past[0][0] is not None
    assert past[0][1] is None
    print("✅ test_forward_with_cache passed")


def test_hybrid_forward_with_cache():
    """Test hybrid model forward with cache returns both attn KV and SSM states."""
    config = load_config('configs/config_hybrid.yaml')
    model = build_model(config, device='cpu')
    model.eval()
    x = torch.randint(0, config['model']['vocab_size'], (2, 10))
    with torch.no_grad():
        out, past = model(x, use_cache=True)
    assert out.shape == (2, 10, config['model']['vocab_size'])
    assert len(past) == 6
    # Layer 0 is attn -> (attn_kv, None)
    # Layer 1 is ssm -> (None, ssm_state)
    assert past[0][0] is not None  # attn KV
    assert past[1][1] is not None  # SSM state
    print("✅ test_hybrid_forward_with_cache passed")


def test_generate_basic():
    """Test basic generation."""
    config = load_config('configs/pretrain.yaml')
    model = build_model(config, device='cpu')
    model.eval()
    tokens = [2]  # BOS
    generated = model.generate(tokens, max_length=5, device='cpu')
    assert len(generated) > 1
    assert len(generated) <= len(tokens) + 5
    print("✅ test_generate_basic passed")


def test_generate_hybrid():
    """Test generation with hybrid model (SSM KV-cache)."""
    config = load_config('configs/config_hybrid.yaml')
    model = build_model(config, device='cpu')
    model.eval()
    tokens = [2]
    generated = model.generate(tokens, max_length=5, device='cpu')
    assert len(generated) > 1
    assert len(generated) <= len(tokens) + 5
    print("✅ test_generate_hybrid passed")


def test_rope_instance_cache():
    """Test RotaryEmbedding uses instance-level cache."""
    rope = RotaryEmbedding(64)
    q = torch.randn(1, 8, 10, 64)
    k = torch.randn(1, 8, 10, 64)
    # First call - populate cache
    q1, k1 = rope(q, k, start_pos=0, max_len=2048)
    # Second call - should hit instance cache
    q2, k2 = rope(q, k, start_pos=0, max_len=2048)
    # Results should be identical
    assert torch.allclose(q1, q2)
    assert torch.allclose(k1, k2)
    # Cache should be in instance
    assert hasattr(rope, '_cache')
    assert len(rope._cache) > 0
    print("✅ test_rope_instance_cache passed")


def test_mamba_ssm_incremental():
    """MambaSSM 增量解码必须与全量重算一致（捕获因果卷积 / conv 状态错位回归）。"""
    ssm = MambaSSM(dim=64, d_state=16)
    ssm.eval()
    # Prefill
    x = torch.randn(1, 4, 64)
    with torch.no_grad():
        y1, state, conv_state = ssm(x, use_cache=True)
    # 增量步：在 past_state / past_conv_state 之上处理下一个 token
    x_step = torch.randn(1, 1, 64)
    with torch.no_grad():
        y2, new_state, new_conv = ssm(x_step, past_state=state, past_conv_state=conv_state, use_cache=True)
    # 全量重算（prefill + step 拼接）应当等于增量 step 的输出
    x_full = torch.cat([x, x_step], dim=1)
    with torch.no_grad():
        y_full, _, _ = ssm(x_full, use_cache=False)
    assert torch.allclose(y2, y_full[:, -1:], atol=1e-5), (
        f"增量解码与全量重算不一致：max diff = {(y2 - y_full[:, -1:]).abs().max().item()}")
    # 形状与状态维度校验
    assert y1.shape == (1, 4, 64)
    assert y2.shape == (1, 1, 64)
    assert state.shape == new_state.shape == (1, 64, 16)
    assert conv_state is not None and new_conv is not None
    assert conv_state.shape == new_conv.shape == (1, 64, ssm.conv_kernel - 1)
    print("✅ test_mamba_ssm_incremental passed")


def test_tie_weights_after_to():
    """Test tie_weights works after .to(device)."""
    config = load_config('configs/pretrain.yaml')
    model = build_model(config, device='cpu')
    # Move to CPU again (simulates device change)
    model.to('cpu')
    model.tie_weights()
    assert model.embedding.weight is model.output_head.weight
    print("✅ test_tie_weights_after_to passed")


def test_attention_mask_device_change():
    """Test attention mask cache handles device changes."""
    attn = SlidingWindowCausalSelfAttention(64, 8, max_seq_length=32)
    attn.eval()
    x = torch.randn(1, 10, 64)
    # First forward on CPU
    with torch.no_grad():
        _ = attn(x, use_cache=False)
    # 验证实际使用的 _bias_cache（传给 SDPA 的 float attn_mask）已构建且设备正确
    assert attn._bias_cache is not None, "_bias_cache 应在前向后构建"
    assert attn._bias_cache.device == x.device, (
        f"_bias_cache 设备 {attn._bias_cache.device} != 输入设备 {x.device}")
    print("✅ test_attention_mask_device_change passed")


if __name__ == '__main__':
    test_model_creation()
    test_hybrid_model_creation()
    test_forward_pass()
    test_forward_with_cache()
    test_hybrid_forward_with_cache()
    test_generate_basic()
    test_generate_hybrid()
    test_rope_instance_cache()
    test_mamba_ssm_incremental()
    test_tie_weights_after_to()
    test_attention_mask_device_change()
    print("\n🎉 All tests passed!")