from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict
from models.constants import MASK_FILL_VALUE, ROPE_BASE


@dataclass
class SSMConfig:
    """SSM 参数组。"""
    d_state: int = 16
    d_inner_factor: int = 1
    dt_rank: Optional[int] = None
    conv_kernel: int = 3
    dt_proj_bias_init: float = 0.1
    a_log_init_range: List[float] = field(default_factory=lambda: [-1, 1])
    D_init: float = 1.0
    ssm_type: str = 'standard'

    def __post_init__(self):
        assert self.d_state > 0, f"ssm_d_state must be > 0, got {self.d_state}"
        assert self.d_inner_factor > 0, f"ssm_d_inner_factor must be > 0, got {self.d_inner_factor}"
        assert self.conv_kernel >= 1, f"ssm_conv_kernel must be >= 1, got {self.conv_kernel}"
        assert self.ssm_type in ('standard', 'cast'), f"ssm_type must be 'standard' or 'cast', got '{self.ssm_type}'"
        assert len(self.a_log_init_range) == 2, f"a_log_init_range must have 2 elements"


@dataclass
class AttnConfig:
    """注意力参数组。"""
    window: int = 0
    rel_bias: bool = False
    qk_norm: bool = True
    attn_temp: bool = True
    rope_learnable: bool = False
    alibi: bool = False
    learn_window: bool = False
    window_base: int = 64
    mixer: str = 'attn'
    linear_attn_feature: str = 'relu'
    linear_attn_head_dim: Optional[int] = None
    # 第十一轮新特性
    linear_correction: bool = False  # 线性注意力作"修正项"补主注意力（而非凸组合替代）
    pe_gate: bool = False            # 位置编码选择性门控（per-head 可学强度）
    # 第十五轮新特性
    rope_dim_fraction: float = 1.0   # Partial RoPE：仅前 dim*fraction 维度加 RoPE（1.0=全维，向后兼容）
    output_gate: bool = False        # 注意力输出门控（消除 Attention Sink）
    zero_centered_norm: bool = False # Zero-Centered RMSNorm（先去均值再归一化，防 Massive Activation）
    delta_alpha_init: float = -2.0   # GatedDeltaNet 衰减门偏置初值（sigmoid≈0.12，弱遗忘起步）
    delta_beta_init: float = 2.0     # GatedDeltaNet 输入门偏置初值（sigmoid≈0.88，强写入起步）
    # 第十七轮新特性
    use_mla_kv: bool = False         # MLA 风格 KV 潜空间压缩（cache 只存潜向量，长序列内存降 2*dim/kv_latent_dim 倍）
    kv_latent_dim: Optional[int] = None  # MLA 潜空间维度（None=默认 dim，压缩 2x；更小值压缩更多）

    def __post_init__(self):
        _VALID = {'attn', 'linear', 'linear2d', 'attn_linear', 'hybrid_linear2d', 'diff', 'gated_delta'}
        if self.mixer == 'hybrid':
            self.mixer = 'attn_linear'
        if self.mixer not in _VALID:
            raise ValueError(f"未知 mixer='{self.mixer}'，可选 {_VALID}")
        # MLA 仅支持标准 attn 系 mixer：DifferentialAttention 和 AxialLinearAttention
        # （hybrid_linear2d）不接受 use_mla_kv 参数，传入会静默忽略 MLA 配置
        if self.use_mla_kv and self.mixer not in {'attn', 'attn_linear'}:
            raise ValueError(
                f"use_mla_kv=True 仅支持 mixer in {{'attn','attn_linear'}}，"
                f"当前 mixer='{self.mixer}' 不支持 MLA KV 压缩")


@dataclass
class MemoryConfig:
    """记忆参数组。"""
    size: int = 0
    comp_dim: int = 32
    retrieval: bool = False
    sparse_topk: int = 0
    forget: bool = False
    product_key: bool = False
    retrieval_full: bool = False
    retrieval_topk: int = 32

    def __post_init__(self):
        if self.size > 0:
            assert self.comp_dim > 0, f"memory_comp_dim must be > 0 when memory_size > 0"


@dataclass
class ModelConfig:
    """模型配置 schema（替代 42 个 mc.get() 散参数）。

    用法：
        cfg = ModelConfig.from_dict(config['model'])
        model = TransformerModel.from_config(cfg)

    或直接构造：
        cfg = ModelConfig(vocab_size=200, embedding_dim=64, ...)
        model = TransformerModel.from_config(cfg)
    """
    # 必填
    vocab_size: int = 0
    embedding_dim: int = 0
    num_heads: int = 0
    num_layers: int = 0
    hidden_dim: int = 0
    max_seq_length: int = 0

    # 通用
    dropout: float = 0.0
    tie_weights: bool = True
    gradient_checkpointing: bool = True
    layer_plan: Optional[str] = None
    rope_base: float = ROPE_BASE
    rope_max_len: Optional[int] = None  # None → 用 max_seq_length
    mask_fill_value: float = MASK_FILL_VALUE

    # 架构增强
    layer_skip: bool = False
    learn_window: bool = False
    window_base: int = 64
    hybrid_single_gate: bool = False
    residual_gate: bool = True
    hybrid_gate: bool = True

    # 字符合并
    char_merge: bool = False
    char_merge_kernel: int = 3
    char_merge_dropout: float = 0.0

    # 共享
    share_attn_proj: bool = False
    share_ffn: bool = False
    share_norm: bool = False

    # 第十一轮新特性
    cross_layer_routing: bool = False  # 跨层稀疏路由信息流动（DenseNet 风格 top-k 跳跃连接）
    cross_layer_topk: int = 2          # 跨层路由每层检索的前层数量
    qat_bits: int = 0                  # 量化感知训练位宽（0=关闭，8=int8 量化噪声模拟）
    ssm_as_memory: bool = False        # SSM 输出作隐式记忆注入 hybrid 块注意力（需 hybrid 层）
    # 第十二轮新特性
    cross_ssm_transfer: bool = False   # 层间 SSM 状态传递（hybrid 块间传递 SSM 信息，需 hybrid 层）
    progressive_residual: bool = False # 渐进式残差（浅层残差大保留信息，深层残差小激进变换）
    # 第十三轮新特性：跨层协作深化
    layer_film: bool = False           # 跨层 FiLM 调制（浅层输出→γ,β 调制深层 input，init 恒等）
    highway_gate: bool = False         # 动态残差门控（input-dependent gate 替代静态 residual_gate）
    # 第十四轮新特性：跨层协作再深化
    input_highway: bool = False        # 输入全局高速公路（embedding 输出经门控注入每层，init 弱注入）
    layer_contrastive: bool = False    # 层间对比绑定（相邻层余弦相似度损失，防深层遗忘浅层特征）
    shared_alibi: bool = False         # ALiBi 跨层共享（所有层共用同一组斜率，减参+一致位置建模）
    # 第十七轮新特性
    fuse_swiglu: bool = False          # SwiGLU w1/w3 合并为 w13（减少 GEMM 调用，默认关向后兼容）

    # n-gram
    ngram_fusion: bool = False
    ngram_gate_scale: float = 1.0
    igmcg: bool = False

    # 子配置
    ssm: SSMConfig = field(default_factory=SSMConfig)
    attn: AttnConfig = field(default_factory=AttnConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    def __post_init__(self):
        assert self.vocab_size > 0, f"vocab_size must be > 0"
        assert self.embedding_dim > 0, f"embedding_dim must be > 0"
        assert self.num_heads > 0, f"num_heads must be > 0"
        assert self.num_layers > 0, f"num_layers must be > 0"
        assert self.hidden_dim > 0, f"hidden_dim must be > 0"
        assert self.max_seq_length > 0, f"max_seq_length must be > 0"
        assert self.embedding_dim % self.num_heads == 0, \
            f"embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({self.num_heads})"
        if self.rope_max_len is None:
            self.rope_max_len = self.max_seq_length

    @classmethod
    def from_dict(cls, mc: Dict[str, Any]) -> ModelConfig:
        """从 config['model'] 字典构建 ModelConfig（替代 42 个 mc.get()）。"""
        ssm = SSMConfig(
            d_state=mc.get('ssm_d_state', 16),
            d_inner_factor=mc.get('ssm_d_inner_factor', 1),
            dt_rank=mc.get('ssm_dt_rank', None),
            conv_kernel=mc.get('ssm_conv_kernel', 3),
            dt_proj_bias_init=mc.get('ssm_dt_proj_bias_init', 0.1),
            a_log_init_range=mc.get('ssm_a_log_init_range', [-1, 1]),
            D_init=mc.get('ssm_D_init', 1.0),
            ssm_type=mc.get('ssm_type', 'standard'),
        )
        attn = AttnConfig(
            window=mc.get('attn_window', 0),
            rel_bias=mc.get('attn_rel_bias', False),
            qk_norm=mc.get('qk_norm', True),
            attn_temp=mc.get('attn_temp', True),
            rope_learnable=mc.get('rope_learnable', False),
            alibi=mc.get('alibi', False),
            learn_window=mc.get('learn_window', False),
            window_base=mc.get('window_base', 64),
            mixer=mc.get('mixer', 'attn'),
            linear_attn_feature=mc.get('linear_attn_feature', 'relu'),
            linear_attn_head_dim=mc.get('linear_attn_head_dim', None),
            linear_correction=mc.get('linear_correction', False),
            pe_gate=mc.get('pe_gate', False),
            rope_dim_fraction=mc.get('rope_dim_fraction', 1.0),
            output_gate=mc.get('output_gate', False),
            zero_centered_norm=mc.get('zero_centered_norm', False),
            delta_alpha_init=mc.get('delta_alpha_init', -2.0),
            delta_beta_init=mc.get('delta_beta_init', 2.0),
            use_mla_kv=mc.get('use_mla_kv', False),
            kv_latent_dim=mc.get('kv_latent_dim', None),
        )
        memory = MemoryConfig(
            size=mc.get('memory_size', 0),
            comp_dim=mc.get('memory_comp_dim', 32),
            retrieval=mc.get('memory_retrieval', False),
            sparse_topk=mc.get('memory_sparse_topk', 0),
            forget=mc.get('memory_forget', False),
            product_key=mc.get('memory_product_key', False),
            retrieval_full=mc.get('memory_retrieval_full', False),
            retrieval_topk=mc.get('memory_retrieval_topk', 32),
        )
        return cls(
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
            rope_base=mc.get('rope_base', ROPE_BASE),
            rope_max_len=mc.get('rope_max_len', None),
            mask_fill_value=mc.get('mask_fill_value', MASK_FILL_VALUE),
            layer_skip=mc.get('layer_skip', False),
            learn_window=mc.get('learn_window', False),
            window_base=mc.get('window_base', 64),
            hybrid_single_gate=mc.get('hybrid_single_gate', False),
            residual_gate=mc.get('residual_gate', True),
            hybrid_gate=mc.get('hybrid_gate', True),
            char_merge=mc.get('char_merge', False),
            char_merge_kernel=mc.get('char_merge_kernel', 3),
            char_merge_dropout=mc.get('char_merge_dropout', 0.0),
            share_attn_proj=mc.get('share_attn_proj', False),
            share_ffn=mc.get('share_ffn', False),
            share_norm=mc.get('share_norm', False),
            cross_layer_routing=mc.get('cross_layer_routing', False),
            cross_layer_topk=int(mc.get('cross_layer_topk', 2)),
            qat_bits=int(mc.get('qat_bits', 0)),
            ssm_as_memory=bool(mc.get('ssm_as_memory', False)),
            cross_ssm_transfer=bool(mc.get('cross_ssm_transfer', False)),
            progressive_residual=bool(mc.get('progressive_residual', False)),
            layer_film=bool(mc.get('layer_film', False)),
            highway_gate=bool(mc.get('highway_gate', False)),
            input_highway=bool(mc.get('input_highway', False)),
            layer_contrastive=bool(mc.get('layer_contrastive', False)),
            shared_alibi=bool(mc.get('shared_alibi', False)),
            fuse_swiglu=bool(mc.get('fuse_swiglu', False)),
            ngram_fusion=mc.get('ngram_fusion', False),
            ngram_gate_scale=float(mc.get('ngram_gate_scale', 1.0)),
            igmcg=bool(mc.get('igmcg', False)),
            ssm=ssm,
            attn=attn,
            memory=memory,
        )
