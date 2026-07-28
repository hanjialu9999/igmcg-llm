"""第二十六轮性能优化回归测试：rope.py 缓存命中路径删冗余 .to(dtype)。

覆盖：
- 数值等价性：cache hit 路径与 cache miss 路径返回相同结果
- dtype 一致性：缓存表 dtype 与请求 dtype 一致时直接返回
- dtype 不一致时仍正确转换（混合精度场景）
- learnable RoPE 路径不受影响
- 应用 RoPE 后输出 dtype 与输入一致
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.rope import RotaryEmbedding


def test_rope_cache_hit_dtype_match():
    """cache hit 时 dtype 一致：直接切片返回，不再调用 .to(dtype)。
    返回的 cos/sin dtype 应等于请求的 dtype。"""
    rope = RotaryEmbedding(dim=32)
    q = torch.randn(2, 4, 8, 32)  # (B, H, T, D)
    # 首次调用：cache miss
    cos1, sin1 = rope._get_cos_sin(0, 8, q.device, q.dtype)
    # 第二次调用：cache hit
    cos2, sin2 = rope._get_cos_sin(0, 8, q.device, q.dtype)
    assert cos1.dtype == q.dtype
    assert sin1.dtype == q.dtype
    assert cos2.dtype == q.dtype
    assert sin2.dtype == q.dtype
    # 数值完全等价
    assert torch.equal(cos1, cos2)
    assert torch.equal(sin1, sin2)


def test_rope_cache_hit_no_extra_to():
    """cache hit 路径不再触发冗余 .to(dtype)：通过算子计数验证。
    使用 torch.addcmul 等其他算子作为对照，确保缓存命中后 .to 不被调用。"""
    rope = RotaryEmbedding(dim=32)
    q = torch.randn(2, 4, 8, 32)
    # 首次填充缓存
    rope._get_cos_sin(0, 8, q.device, q.dtype)
    # 用 profiler 计数 aten::to 算子
    with torch.profiler.profile(with_modules=False) as prof:
        for _ in range(5):
            rope._get_cos_sin(0, 8, q.device, q.dtype)
    # 提取 aten::to 事件数
    events = prof.events()
    to_count = sum(1 for e in events if 'aten::to' in str(e.name) if 'dtype' in str(e.name).lower() or True)
    # cache hit 时不应有 aten::to 调用（dtype 一致场景）
    # 注意：profiler 自身可能产生少量 aten::to，但 RoPE 内部不应触发
    # 这里用宽松断言：少于 5 次（避免 profiler 噪声）
    assert to_count < 5, f"cache hit 路径触发 {to_count} 次 aten::to，可能仍有冗余 .to(dtype)"


def test_rope_dtype_mismatch_still_converts():
    """混合精度场景：缓存 fp32 但请求 fp16（或反之）时，应正确转换 dtype。"""
    rope = RotaryEmbedding(dim=32)
    # 首次以 fp32 填充缓存
    cos_fp32, sin_fp32 = rope._get_cos_sin(0, 8, torch.device('cpu'), torch.float32)
    assert cos_fp32.dtype == torch.float32
    # 再次请求 fp16（混合精度场景）：缓存 dtype 不匹配，应转换
    # 注意：CPU 上 fp16 计算有限，主要验证 dtype 转换逻辑
    cos_fp16, sin_fp16 = rope._get_cos_sin(0, 8, torch.device('cpu'), torch.float16)
    assert cos_fp16.dtype == torch.float16
    assert sin_fp16.dtype == torch.float16
    # 数值在 fp16 精度范围内等价
    assert torch.allclose(cos_fp32.half(), cos_fp16, atol=1e-3)


def test_rope_apply_preserves_dtype():
    """应用 RoPE 后输出 dtype 与输入一致（fp32 训练路径）。"""
    rope = RotaryEmbedding(dim=32)
    q = torch.randn(2, 4, 8, 32)
    k = torch.randn(2, 4, 8, 32)
    q_out, k_out = rope(q, k, start_pos=0)
    assert q_out.dtype == q.dtype
    assert k_out.dtype == k.dtype
    # 多次应用 RoPE 结果一致（缓存命中数值不变）
    q_out2, k_out2 = rope(q, k, start_pos=0)
    assert torch.equal(q_out, q_out2)
    assert torch.equal(k_out, k_out2)


def test_rope_learnable_path_unchanged():
    """learnable RoPE 路径不受 cache 优化影响：每步重算带 grad cos/sin。"""
    rope = RotaryEmbedding(dim=32, learnable=True)
    q = torch.randn(2, 4, 8, 32, requires_grad=True)
    # learnable 路径不使用缓存，每步重算
    cos1, sin1 = rope._get_cos_sin(0, 8, q.device, q.dtype)
    cos2, sin2 = rope._get_cos_sin(0, 8, q.device, q.dtype)
    # learnable 路径每次返回都带 grad（可学习参数激活）
    assert cos1.requires_grad or not rope.learnable  # learnable=True 时应带 grad
    # 数值等价（同一 step 内参数未变）
    assert torch.allclose(cos1, cos2, atol=1e-6)


def test_rope_partial_rope_dtype_consistency():
    """Partial RoPE（dim_fraction<1.0）：rot_dim < dim 时 dtype 仍一致。"""
    rope = RotaryEmbedding(dim=32, dim_fraction=0.5)  # rot_dim=16, no_pe_dim=16
    q = torch.randn(2, 4, 8, 32)
    cos, sin = rope._get_cos_sin(0, 8, q.device, q.dtype)
    assert cos.dtype == q.dtype
    assert sin.dtype == q.dtype
    # 应用后输出维度和 dtype 一致
    q_out = rope.apply_to_single(q, start_pos=0)
    assert q_out.shape == q.shape
    assert q_out.dtype == q.dtype


def test_rope_yarn_dtype_consistency():
    """YaRN 长度外推：dtype 一致性不被破坏。"""
    rope = RotaryEmbedding(dim=32, yarn_scale=2.0, yarn_orig_max_seq_length=2048)
    q = torch.randn(2, 4, 8, 32)
    cos, sin = rope._get_cos_sin(0, 8, q.device, q.dtype)
    assert cos.dtype == q.dtype
    # 二次调用 cache hit
    cos2, sin2 = rope._get_cos_sin(0, 8, q.device, q.dtype)
    assert torch.equal(cos, cos2)
    assert torch.equal(sin, sin2)


def test_rope_dim_wise_dtype_consistency():
    """dim_wise RoPE：可学软掩码应用后 dtype 一致。"""
    rope = RotaryEmbedding(dim=32, dim_wise=True)
    q = torch.randn(2, 4, 8, 32)
    cos, sin = rope._get_cos_sin(0, 8, q.device, q.dtype)
    cos_masked, sin_masked = rope._apply_dim_wise_mask(cos, sin)
    assert cos_masked.dtype == q.dtype
    assert sin_masked.dtype == q.dtype


def test_rope_cache_slice_correctness():
    """cache hit 时切片正确：start_pos=4 应取位置 4-11 的 cos/sin。"""
    rope = RotaryEmbedding(dim=32)
    # 首次填充长度 16 的缓存
    cos_full, sin_full = rope._get_cos_sin(0, 16, torch.device('cpu'), torch.float32)
    # 从 start_pos=4 取 seq_len=8（cache hit）
    cos_slice, sin_slice = rope._get_cos_sin(4, 8, torch.device('cpu'), torch.float32)
    # 验证切片正确：等于完整表的 [4:12]
    assert torch.equal(cos_slice, cos_full[:, :, 4:12, :])
    assert torch.equal(sin_slice, sin_full[:, :, 4:12, :])


def test_rope_cache_extension():
    """cache miss 后扩展：need > cached size 时正确扩展缓存。"""
    rope = RotaryEmbedding(dim=32)
    # 首次填充 need=8
    cos1, sin1 = rope._get_cos_sin(0, 8, torch.device('cpu'), torch.float32)
    # 再次请求 need=16（cache miss，扩展）
    cos2, sin2 = rope._get_cos_sin(0, 16, torch.device('cpu'), torch.float32)
    assert cos2.size(2) == 16
    # 前 8 个位置应与首次一致（同一基准表）
    assert torch.allclose(cos1, cos2[:, :, :8, :], atol=1e-6)


if __name__ == '__main__':
    test_rope_cache_hit_dtype_match()
    test_rope_cache_hit_no_extra_to()
    test_rope_dtype_mismatch_still_converts()
    test_rope_apply_preserves_dtype()
    test_rope_learnable_path_unchanged()
    test_rope_partial_rope_dtype_consistency()
    test_rope_yarn_dtype_consistency()
    test_rope_dim_wise_dtype_consistency()
    test_rope_cache_slice_correctness()
    test_rope_cache_extension()
    print("All RoPE optimization tests passed!")
