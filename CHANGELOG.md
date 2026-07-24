# 变更日志 (CHANGELOG)

手工维护的主要变更记录，便于对照 Git 提交历史（`git log`）。

## 约定
- 每个条目以提交哈希（如 `43c7d27`）为标题，并标注父提交与推送状态。
- 按时间倒序排列（最新在最上方）。
- 仅记录对训练 / 推理链路有实质影响的修复、功能与破坏性变更；纯文档微调通常合并记录。
- 提交信息风格：中文主题行 + 空行 + 要点式正文。
- 状态标记：`已推送` = 已 `git push` 到 `origin/main`；`本地` = 仅本地提交待推送。

## `bb239c0`（已推送，第十九轮：KDA 逐通道衰减 + YaRN 长度外推 + 审查修复）

- feat: **KDA 逐通道衰减（Kimi Delta Attention）**——`models/mixers.py` `GatedDeltaNet` 新增 `channel_wise` 参数，α/β 门控从标量（per-head）升级为逐通道向量（per-head per-dim），`alpha_proj`/`beta_proj` 输出维度从 `num_heads` → `num_heads*head_dim`。`_compute_gates` 通道模式返回 `(B,H,T,D)`，`Diag(α_t)·S` 让每个通道独立遗忘率，提升长程检索精度。init W=0（向后兼容，weight=0 时与标量模式行为等价）。config: `gated_delta_channel_wise`。灵感：Kimi K3 KDA（arXiv:2510.26692）。与 MemoryBank per-slot forget 对称。
- feat: **YaRN 长度外推**——`models/rope.py` `RotaryEmbedding` 新增 `yarn_scale`/`yarn_beta`/`yarn_orig_max_seq_length` 参数。`yarn_scale>1.0` 时通过三段式非均匀频率缩放计算 `inv_freq`：高频维度（短波长）保持外推，低频维度（长波长）插值缩放。辅助函数 `_yarn_find_correction_dim`/`_yarn_find_correction_range`/`_yarn_linear_ramp_mask`。与 Partial RoPE 正交叠加。`SlidingWindowCausalSelfAttention` 同步支持 YaRN 参数（传给 RotaryEmbedding）。config: `yarn_scale`/`yarn_beta`/`yarn_orig_max_seq_length`。灵感：YaRN（arXiv:2309.00071）。
- fix: **MLA + nope_layers 场景 k 的 RoPE 应用不一致（CRITICAL，审查发现）**——NoPE 层（`use_rope=False`）的 k 在 MLA 路径 `attend` 中仍被 RoPE 旋转（q 无 RoPE/k 有 RoPE 导致点积语义错误）。修复：`models/mixers.py` MLA 路径 k 的 RoPE 应用添加 `if self.use_rope` 条件判断。
- fix: **shared_alibi_enabled 标志未考虑 nope_layers_set（MEDIUM，审查发现）**——`shared_alibi=True + alibi=False + nope_layers=[1]` 时 `shared_alibi_enabled` 为 False，设备迁移后 alibi_slopes 共享关系丢失。修复：`models/transformer.py` `shared_alibi_enabled = shared_alibi and (alibi or bool(self.nope_layers_set))`。
- fix: **nope_layers 索引越界无校验（LOW，审查发现）**——`nope_layers` 包含超出 `num_layers` 的索引时未报错。修复：`models/transformer.py` 添加越界校验，`_invalid = {i for i in self.nope_layers_set if i < 0 or i >= _n_layers}; if _invalid: raise ValueError(...)`。
- fix: **SlidingWindowCausalSelfAttention 不接受 yarn_scale 参数（构建阻断）**——`RotaryEmbedding` 初始化引用了 `yarn_scale` 等变量但 `__init__` 签名未声明，导致 NameError。修复：签名添加 `yarn_scale`/`yarn_beta`/`yarn_orig_max_seq_length` 参数。
- test: **新增 tests/test_round19.py（20 项）**——KDA 通道/标量参数维度/门控形状/前向/输出差异/梯度/cache parity/专用初始化保持；YaRN 向后兼容/inv_freq 缩放/前向输出/Partial RoPE 正交/模型集成/梯度；MLA+NoPE k 不旋转/shared_alibi 标志/nope_layers 越界校验；KDA+YaRN 组合。pytest **384 passed / 1 skipped / 1 xfailed**（+20）。

## `e5d87d5`（已推送，第十八轮：iRoPE 交错 NoPE + Gated Attention + GEMM 合并优化）

- feat: **iRoPE 交错 NoPE 层**——`nope_layers` 配置指定层关闭 RoPE，位置信号由 ALiBi 提供（NoPE 层强制 `alibi=True`）。`SlidingWindowCausalSelfAttention` 新增 `use_rope` 参数，`project_and_norm` 中 `use_rope=False` 时跳过 RoPE 应用。灵感：LLaMA 4 iRoPE（3:1 交错 RoPE/NoPE）。NoPE 层理论上能学到任意长度外推。默认空列表（向后兼容）。config: `nope_layers=[1,5,...]`。
- feat: **Gated Attention（output_gate 门源升级 out→query）**——`models/mixers.py` `SlidingWindowCausalSelfAttention.forward` 中 output_gate 的门源从注意力输出 `out` 升级为 `query`（`(B,H,T,D) → (B,T,dim)` 后投影）。论文关键：门必须由 query 产生才触发 query-dependent 稀疏 + value 通路非线性，BOS 注意力下沉 46.7%→4.8%。ckpt 路径（`_run_attn_mixer`）同步更新。init W=0/b=0 → sigmoid=0.5 半通起步（向后兼容）。灵感：NeurIPS 2025 Best Paper。
- feat: **GEMM 合并优化**——(1) `TransformerBlock.ssm_k_proj`/`ssm_v_proj` 合并为 `ssm_kv_proj`（`Linear(dim, 2*head_dim)`，前向 chunk 拆分）；(2) `MemoryBank.mem_k`/`mem_v` 合并为 `mem_kv_proj`。减少 DML 小算子启动税。`checkpoint.py` 自动检测旧格式并转换（`mem_kv_proj.weight = cat([mem_k, mem_v], dim=0)`）。`MemoryBank.convert_legacy_state_dict` 静态方法。
- perf: **inject_memory 清理 2 处冗余 .to(device)**——`retrieval_gate` 是 Parameter 已在设备；比较结果 `(mlogits < thr)` 继承操作数设备。消除热路径冗余设备检查。
- test: **新增 tests/test_round18.py（11 项）**——iRoPE 配置/输出/梯度/默认空；Gated Attention init/梯度/输出；GEMM 合并转换/前向/checkpoint。pytest **364 passed / 1 skipped / 1 xfailed**（+11）。

## `08d4c09`（第十七轮审查后续修复：cache 协议与增量解码一致性）

- fix: **CRITICAL #1 RNN-state mixers 增量解码 start_pos 永远返回 1**——LinearAttention/GatedDeltaNet/AxialLinearAttention 增量路径只存当前 token 的 k（T=1），导致 BlockState.start_pos 始终为 1，RoPE 位置全部错位。新增 `_accum_kv()` 辅助函数累积 k/v。
- fix: **CRITICAL #2 MLA + hybrid_linear2d 配置校验通过但实现静默忽略 MLA**——校验集收紧为 `{'attn','attn_linear'}`。
- fix: **MEDIUM #3 MLA + attn_linear hybrid 增量解码 2.2% 偏差**——`_accum_kv` 检查维度匹配后再累积。
- fix: **GatedDeltaNet 增量解码输出布局**（多余 transpose）+ **MambaSSM conv_state 首步形状不足**（左补零）。
- test: 新增 `tests/test_cache_parity.py`（6 passed + 1 xfail）。AxialLinearAttention 2D→1D 退化标 xfail（已知架构取舍）。

## `bc55c3d`（已推送，第十七轮：MLA KV 压缩 + SwiGLU 合并 + 审查修复）

- feat: **MLA 风格 KV 潜空间压缩**——`models/mixers.py` `SlidingWindowCausalSelfAttention` 新增 `use_mla_kv`/`kv_latent_dim` 参数。K/V 拼接后下投影到低维潜空间 `c_kv`（`kv_compress: 2*dim → kv_latent_dim`），cache 只存潜向量；attend 时单次 GEMM 上投影还原 K+V（`kv_decompress: kv_latent_dim → 2*dim`，chunk 拆分）。RoPE 在解压后对 k 应用（位置 0..T_total-1），q 在 project_and_norm 中应用（位置 start_pos），保证旋转信息不被压缩-还原破坏。cache 内存降 `2*dim/kv_latent_dim` 倍。config: `use_mla_kv`/`kv_latent_dim`。灵感：DeepSeek-V3 MLA。
- feat: **SwiGLU w1/w3 合并优化**——`models/mixers.py` `SwiGLU` 新增 `fuse_swiglu` 参数，将 `w1`/`w3` 合并为单个 `w13` Linear（输出 `2*hidden_dim`），前向时 `chunk(2, dim=-1)` 拆分（view 零拷贝）。减少一次 GEMM 调用，DML 小算子启动税敏感时有益。`convert_legacy_state_dict` 静态方法处理旧 checkpoint 权重转换（`w13.weight = cat([w1.weight, w3.weight], dim=0)`）。`checkpoint.py` `load_model` 自动检测格式不匹配并转换。默认关（向后兼容旧 checkpoint 的 w1/w2/w3 格式）。config: `fuse_swiglu`。
- fix: **DifferentialAttention present 只存当前 token 导致增量解码历史丢失（CRITICAL pre-existing）**——`models/mixers.py` `DifferentialAttention.forward` 的 `present` 原存 `(k1, k2, v)`（仅当前 token），而非累积的 `(k1_full, k2_full, v_full)`。增量解码第三步起历史完全丢失（max_diff=0.33）。修复：present 改存 `k1_full`/`k2_full`/`v_full` 的转置（累积历史）。
- fix: **DifferentialAttention cache 布局 (B,T,H,D) 与 BlockState.start_pos 不兼容（MEDIUM pre-existing）**——原 cache 为 `(B,T,H,D)` 布局，`BlockState.start_pos` 取 `size(2)` 返回 `num_heads` 而非 `seq_len`。混合层计划（diff 块后跟 attn 块）下后续块 `start_pos` 错误。修复：cache 统一为 `(B,H,T,D)` 布局（present 存 transpose 后的 k1/k2/v）。
- fix: **SwiGLU convert_legacy_state_dict endswith 对顶层模块不匹配（MEDIUM）**——`k.endswith('.w1.weight')` 对顶层 SwiGLU（key=`"w1.weight"`，无前导点）不匹配，导致转换失败。修复：改用 `split('.')` + `parts[-2:] == ['w1', 'weight']` 精确匹配，兼容顶层和嵌套模块。同时预扫描修复 w3 在 w1 之前迭代时的残留键问题。
- fix: **checkpoint 加载不自动转换 SwiGLU 格式（MEDIUM）**——`checkpoint.py` `load_model` 用 `strict=False` 静默忽略 w13 缺失/w1-w3 多余，导致 fuse_swiglu=True 模型加载旧 checkpoint 时 FFN 随机初始化。修复：load 前检测格式不匹配并自动调用 `convert_legacy_state_dict`。
- fix: **MLA + mixer='diff' 静默忽略 MLA 配置（LOW）**——`_build_attn_mixer` 的 diff 分支不传 `use_mla_kv`，配置不报错但 MLA 被忽略。修复：`AttnConfig.__post_init__` 校验 `use_mla_kv` 仅与 `mixer in {'attn','attn_linear','hybrid_linear2d'}` 组合。
- perf: **MLA 解压 GEMM 合并（2→1）**——原 `kv_decompress_k`/`kv_decompress_v` 两次独立 GEMM 合并为单个 `kv_decompress`（输出 `2*dim`，chunk 拆分），减少一次 GEMM launch。与项目已有 `fuse_swiglu` 同思路。
- perf: **_sync_window 推理期跳过 DML CPU 同步税**——`float(self.log_window)` 在 DML 上触发 GPU→CPU 同步。推理期参数冻结，首次同步后跳过（`_window_synced` 标志）；训练期每步同步（参数在变）。
- test: **新增 6 个回归测试**——DifferentialAttention cache 布局/start_pos 正确性/增量解码一致性、MLA+diff 配置校验、convert_legacy_state_dict 预扫描、_sync_window 推理期跳过、checkpoint 自动转换 SwiGLU。pytest **347 passed / 1 skipped**（+6）。
- smoke: **双配置 CPU 训练验证**——config_smoke_mla.yaml（MLA kv_latent_dim=128）Val 6.5847；config_smoke_fuse_swiglu.yaml（SwiGLU 合并）Val 6.1352。

## `837d821`（已推送，基于 `4e9e8f3`，第十六轮：Gate 抽象统一 + 代码债清理）

- refactor: **Gate 抽象统一**——新建 `models/gates.py`，定义 `GateConfig` dataclass + 6 个工具函数（`apply_direct`/`apply_sigmoid_scalar`/`apply_linear_gate`/`convex_combine_scalar`/`convex_combine_linear`/`apply_correction`）。`TransformerBlock.__init__` 的 6 个散落 bool 门控参数（`residual_gate`/`hybrid_gate`/`skip`/`hybrid_single_gate`/`linear_correction`/`highway_gate`）收口为单一 `gate_cfg: GateConfig` 参数，__init__ 签名从 20 个参数精简到 15 个。`forward` 中散乱的 `if gate is not None` 分支统一用工具函数替代。**参数注册方式不变，state_dict 100% 兼容**。
- test: **新增 13 个 Gate 抽象回归测试**——覆盖 GateConfig 默认值/from_kwargs 兼容、6 个工具函数的 None passthrough 和数值正确性、gate_cfg 构造与默认构造的 state_dict key 一致性、highway_gate 互斥约束、全特性前向+反向。pytest **317 passed / 1 skipped**（+13）。
- review: **子代理独立审查确认无 bug**——行为等价性（apply_direct/convex_combine/apply_correction 与原内联表达式完全一致）、state_dict 兼容性（参数注册方式未变）、专用初始化正确引用 gate_cfg 属性。

## `（本地，基于 `d21464b`，待推送，第十五轮：新架构特性 + 性能优化 + 代码债清理）

- feat: **Gated DeltaNet（delta rule + α/β 门控）**——`models/mixers.py` 新增 `GatedDeltaNet` 类（继承 `LinearMixerBase`），递推改为 `S_t = α_t·S_{t-1} + β_t·(v_t - S_{t-1}·k_t)⊗k_t`（gated delta rule），替代 LinearAttention 的纯加法更新。α/β 门控 per-head per-token 由输入 x 经线性+sigmoid 决定（init W=0, bias α=-2/β=2 → sigmoid 0.12/0.88 弱遗忘强写入起步）。k L2 归一化保证 delta rule 谱半径<1。config: `mixer='gated_delta'`。灵感：Gated DeltaNet (ICLR 2025) + Qwen3-Next + Kimi KDA。
- feat: **Partial RoPE（仅前 k% 维度加 RoPE）**——`models/rope.py` `RotaryEmbedding` 新增 `dim_fraction` 参数（默认 1.0 全维，向后兼容）。`rot_dim = max(2, int(dim*fraction)//2*2)`（向下取偶），后段 `no_pe_dim` 维度不旋转（NoPE 纯内容维度）。`_rope_apply` 拆分前段旋转+后段透传。config: `rope_dim_fraction`。灵感：Qwen3-Next（前 25%）+ MLA Decoupled RoPE。
- feat: **Output Gating（注意力输出门控）**——`models/mixers.py` `SlidingWindowCausalSelfAttention` 新增 `output_gate` Linear（dim→dim），attend 输出后 `out = out * sigmoid(W·out + b)`（init W=0, b=0 → sigmoid 0.5 半通起步）。消除 Attention Sink / Massive Activation。`_run_attn_mixer` ckpt 路径补应用（ckpt 绕过 forward 直接调 attend，须显式补 output_gate）。config: `output_gate`。灵感：Qwen3-Next Output Gating。
- feat: **Zero-Centered RMSNorm**——`models/norms.py` `RMSNorm` 新增 `zero_centered` 选项（默认 False），先去均值 `x = x - mean(x)` 再 rms 归一化。防止 norm 权重异常增大（Massive Activation），DML 数值稳定性提升。`TransformerModel` 正确传递到 ln1/ln2/ln_f（修复 bug：原构造器接受 `zero_centered_norm` 但未传给 `TransformerBlock`）。config: `zero_centered_norm`。灵感：Qwen3-Next Zero-Centered RMSNorm。
- fix: **GatedDeltaNet/output_gate 专用初始化被 _init_weights 覆盖（HIGH）**——`_apply_specialized_inits` 遍历 blocks 重置 alpha_proj/beta_proj（weight=0, bias=alpha_init/beta_init）和 output_gate（weight=0, bias=0），恢复专用初始化设计意图。
- fix: **zero_centered_norm 未传递给 TransformerBlock（HIGH）**——`TransformerModel.__init__` 接受 `zero_centered_norm` 参数但构造 blocks 时未传递，导致所有 RMSNorm 恒用默认 False。修复：显式传 `zero_centered_norm=zero_centered_norm`。同时修复 `ln_f` 也使用 zero_centered。
- fix: **output_gate 在 ckpt 路径未应用（HIGH）**——`_run_attn_mixer` 的 ckpt+attend 路径绕过 `SlidingWindowCausalSelfAttention.forward`（output_gate 在 forward 中应用），导致训练时 output_gate 参数无梯度。修复：ckpt 路径后显式补 `h = h * sigmoid(output_gate(h))`。
- fix: **AxialLinearAttention super().__init__() 位置参数错位（HIGH）**——`AxialLinearAttention.__init__` 用位置参数调 `super().__init__()`，`shared_qkv` 被传到 `rope_dim_fraction` 位置导致 `TypeError: float() argument must be a string or a real number, not 'NoneType'`。修复：改用关键字参数。
- fix: **Partial RoPE 测试期望错误（LOW）**——`test_partial_rope_no_pe_dim_passthrough` 用 cos=sin=0 期望 identity，但 RoPE identity 条件是 cos=1,sin=0。修复测试。
- perf: **validate() GPU 累加 loss**——`scripts/train.py` `validate()` 原每 batch `loss.item()` 同步 DML→CPU，改为 GPU 张量累加仅在末尾 `.item()` 一次，消除 N-1 次同步税。
- refactor: **MambaSSMWithCAST forward 去重**——提取 `_compute_dA_and_xb` 方法到 `MambaSSM`，`MambaSSMWithCAST` 只覆盖此方法（添加 CAST A_delta），删除 ~35 行重复 forward 代码。同时修复 CAST 旧 forward 缺失 `conv_kernel=1` 防御检查（基类有 `keep = max(conv_kernel-1, 0)` 保护，CAST 旧代码直接 `-(conv_kernel-1)` 在 kernel=1 时返回全长序列）。
- smoke: **双配置 CPU 训练验证**——
  - config_smoke_features_v4.yaml（Partial RoPE + Output Gating + Zero-Centered RMSNorm + 全套跨层协作）：225 步 Train 8.83 / Val 14.19 / ~1300 tok/s。
  - config_smoke_gated_delta.yaml（Gated DeltaNet + Partial RoPE + Zero-Centered RMSNorm）：225 步 Train 6.15 / Val 5.58 / ~1696 tok/s。GatedDeltaNet 比 std attn 快 30%+ 且 Val Loss 显著更低（delta rule 检索能力优于纯加法线性注意力）。
- test: **新增第十五轮回归测试**——20+ 测试覆盖 GatedDeltaNet（参数创建/初始化/前向/cache parity/梯度回流）、Partial RoPE（rot_dim 计算/passthrough/向后兼容/输出变化/cache parity）、Output Gating（参数创建/初始化/输出变化/梯度回流/cache parity）、Zero-Centered RMSNorm（参数标记/计算正确性/输出变化/cache parity）。pytest **299 passed / 1 skipped**。

### 已识别但未实施的代码债（待后续清理）
- ~~15+ 种 gate 机制（residual_gate/hybrid_gate/highway_gate/skip_gate/input_highway_gate 等）分散实现~~ → **第十六轮已清理**（GateConfig dataclass + 6 工具函数）
- ~~MambaSSM vs MambaSSMWithCAST 的 forward 几乎逐字重复~~ → **第十五轮已清理**（提取 `_compute_dA_and_xb` 方法）
- ~~SwiGLU 三个 Linear（w1/w2/w3）可合并为两个（LLaMA 风格 w13 chunk）~~ → **第十七轮已实施**（fuse_swiglu opt-in + convert_legacy_state_dict + checkpoint 自动转换）
- MemoryConfig 与 AttnConfig 的 retrieval 字段已统一到 MemoryConfig（第十四轮已清理 AttnConfig 侧）
- GatedDeltaNet 全量训练 for 循环可优化为 chunk-wise parallel（DeltaNet 原论文做法），当前 T≤64 开销可控

## `（本地，基于 `d21464b`，待推送，第十四轮续：死代码清理 + 性能优化 + 架构创新调研）

- refactor: **移除 AxialLinearAttention 死代码**——`models/mixers.py` 移除 `pos_aware_feat`/`enable_pos_aware_feat`/`_feat_pos`/`pos_feat_alpha`。原代码注释自承"前向输出实际不变；保留接口以备未来接入"，是误导性占位特性（声称支持但实际不工作）。`tests/test_decode_merge_parity.py` 的 `test_axial_linear_pos_aware_feat` 替换为 `test_axial_linear_basic_forward`（验证死代码已移除 + 基本前向正常）。
- refactor: **移除 AttnConfig 未使用字段**——`models/model_config.py` 移除 `AttnConfig.retrieval_full`/`retrieval_topk`。grep 确认无任何代码读取 `attn_cfg.retrieval*`（仅 `MemoryConfig.retrieval*` 被使用），是历史遗留双份字段。`from_dict` 同步移除重复赋值。
- perf: **移除 3 处冗余 `.to(device)`**——`models/mixers.py` L235（`dist` 已由 `device=device` 的 arange 生成）/ L237（`alibi_slopes` 是 buffer，`model.to(device)` 时已移动）/ L287（`drop` 是 `rlogits < thr` 的比较结果，rlogits 已在 device）。DML 上 `.to()` 有 CPU 同步税，热路径避免。
- docs: **移除 ARCHITECTURE_PLAN.md 中 pos_aware_feat 引用**——同步删除"位置感知特征映射"章节。
- research: **架构创新调研报告**——3 个子代理并行（bug 审查/性能分析/架构创新），架构代理联网搜索 2025-2026 主流 LLM 架构（Mamba2/Mamba3/Jamba/Zamba2/RWKV-7/DeepSeek-V3 MLA/Qwen3-Next/Kimi KDA/Gated DeltaNet/CLSA/MSA），结合项目内代码分析，输出 8 个可实施的新架构想法（详见下方"未来架构想法"章节）。

### 未来架构想法（按优先级排序，默认关 opt-in，待后续轮次实施）

1. **★★★★★ Gated DeltaNet 替换 LinearAttention**——灵感：Gated DeltaNet (ICLR 2025) + Kimi KDA + ByteDance 混合线性注意力横评。现状 LinearAttention 用纯加法 `S_t = S_{t-1} + v_t k_t^T` 无 delta rule 无遗忘门，检索能力弱。方案：递推改为 `S_t = α_t·S_{t-1} + β_t·v_t k_t^T - β_t·(S_{t-1} k_t) k_t^T`（gated delta rule）。新增 `mixer='gated_delta'`，默认关。
2. **★★★★★ Partial RoPE（仅前 k% 维度加 RoPE）**——灵感：Qwen3-Next（前 25%）+ MLA Decoupled RoPE。现状 `RotaryEmbedding` 对全 head_dim 旋转。方案：加 `rope_dim_fraction` 参数（默认 1.0），仅前 `head_dim*fraction` 维度旋转，后段 NoPE。长度外推更稳，与 ALiBi+pe_gate 正交。
3. **★★★★ MLA 风格 KV 潜空间压缩**——灵感：DeepSeek-V3 MLA。现状 MemoryBank 已有 compress/decompress 但仅用于固定槽记忆，未用于 KV cache。方案：新增 `use_mla_kv`，K/V 投影后压到 `kv_latent_dim`，cache 只存潜向量。长序列 KV cache 内存降 4-8x。
4. **★★★★ CrossLayerRouter 升级为 CLSA 共享路由索引**——灵感：CLSA (Microsoft 2026) + YOCO。现状每层独立打分 top-k，开销随层数线性增长。方案：引入 indexer 层算一次 token-level top-k，后续层复用。长上下文解码 7.6x 加速。
5. **★★★★ SSM 升级非对角状态转移（RWKV-7 风格）**——灵感：RWKV-7 "Goose" + Mamba2 SSD。现状 MambaSSM 用对角 dA，channel 独立演化。方案：引入非对角项 `G_t = diag(w_t) - κ_t(a_t·κ_t)`，增强跨 channel 混合。新增 `ssm_type='rwkv7'`。
6. **★★★ Output Gating + Zero-Centered RMSNorm**——灵感：Qwen3-Next。现状注意力输出无门控。方案：attend 输出后加 `output_gate=sigmoid(W·x)` 消除 Attention Sink；RMSNorm 加 `zero_centered` 选项 `x/rms(x-mean)`。DML 上提升数值稳定性。
7. **★★★ Intra-layer Hybrid（heads 拆半并行）**——灵感：Meta 混合架构系统分析 (arxiv 2510.04800)。现状 hybrid 块是 attn+ssm 加法并行 2x 算力。方案：新增 `block_type='intra_hybrid'`，heads 拆半（前半 attn 后半 ssm）concat。算力 ≈1x 保留混合建模。
8. **★★★ MemoryBank 可微分 top-k + 文档级 RoPE**——灵感：MSA (EverMind 100M token) + Gated DeltaNet 内容寻址。现状硬 top-k 不可微。方案：sparsemax/soft-topk 替代；记忆槽携带文档位置编码。超长上下文检索精度提升。

### 已识别但未实施的代码债（待后续清理）
- 15+ 种 gate 机制（residual_gate/hybrid_gate/highway_gate/skip_gate/input_highway_gate 等）分散实现，建议统一为通用 Gate 抽象（kind∈{static,dynamic,affine}）
- MambaSSM vs MambaSSMWithCAST 的 forward 几乎逐字重复，可提取 MambaSSMBase
- SwiGLU 三个 Linear（w1/w2/w3）可合并为两个（LLaMA 风格 w13 chunk），但破坏 checkpoint 兼容，需 opt-in 或 state_dict 转换
- MemoryConfig 与 AttnConfig 的 retrieval 字段已统一到 MemoryConfig（本轮已清理 AttnConfig 侧）

## `（本地，基于 `2d154bf`，待推送，第十四轮：跨层协作再深化 + input_highway 增量解码 bug 修复）

- feat: **输入全局高速公路（input_highway）**——`models/transformer.py` embedding 输出 x0 经门控注入每层：`gate=sigmoid(W·x+b)`（init b=-3，sigmoid≈0.05 弱注入），`x = x + gate * proj(x0)`。让每层都能直接访问原始输入信息，避免深层变换后信息丢失。config: `input_highway`。8 个回归测试（参数创建/恒等初始化/输出变化/梯度回流/cache parity/x0 跨步缓存/shape 对齐/与 cross_layer 组合）。
- feat: **层间对比绑定（layer_contrastive）**——训练期累积相邻层 (1 - cos_sim) 损失到 `self._contrastive_loss`，训练循环以 0.01 权重加入主 loss。detach _prev_layer_out 使梯度只回流到当前层（推当前层向上一层靠拢，不反向）。eval 时不计算。防深层过度偏离浅层特征。config: `layer_contrastive`。3 个回归测试（损失计算/梯度回流/cache parity）。
- feat: **ALiBi 跨层共享（shared_alibi）**——所有注意力层共用同一组 alibi_slopes buffer（减参+一致位置建模）。在 blocks 创建后统一绑定第一层的 alibi_slopes 到所有层。config: `shared_alibi`。3 个回归测试（斜率共享/buffer 数量减少/cache parity）。
- fix: **input_highway 增量解码 x0 缓存 bug（HIGH）**——原实现 `x0 = x` 每步重算，增量解码后续步 src 只 1 token，input_highway 注入错误内容（单 token embedding 而非完整 prompt）。修复：首步缓存 `_cached_x0`（detach 避免长生命周期图），后续步用缓存值。
- fix: **input_highway 增量解码 shape 不匹配 bug（CRITICAL）**——后续步 x shape [B,1,D] 与 x0 shape [B,T_prompt,D] 不匹配，直接 broadcast 让 x 被放大到 [B,T_prompt,D]，破坏 cross_layer_routing 的 `torch.stack(prev_outputs)` shape 一致性，导致 `RuntimeError: stack expects each tensor to be equal size`。修复：后续步取 x0 mean-pool 到 [B,1,D] 与当前 x 对齐（语义：注入 prompt 全局摘要到当前生成 token）。
- fix: **shared_alibi .to(device) 后共享被打破（MEDIUM）**——PyTorch _apply 遍历每个 module 独立处理 buffer，会打破 alibi_slopes 的对象共享（数值仍正确，但失去减参优势）。修复：重写 `to()` 方法，在设备迁移后重新绑定共享关系。
- perf: **input_highway_proj 重复计算优化**——原实现每层调用 `self.input_highway_proj(x0)`，x0 在循环中不变。改为循环外预计算一次 `x0_proj = self.input_highway_proj(x0)`，4 层模型节省 3 次 Linear 计算。
- refactor: **测试辅助函数补 import**——`tests/test_new_mechanisms.py` 补 `import torch.nn as nn`（test_input_highway_param_created 用 isinstance(..., nn.Identity)/nn.Linear 但 nn 未导入）。
- refactor: **test_shared_alibi_reduces_params 改为统计 buffer**——alibi_slopes 是 buffer 非 parameter，原测试用 `parameters()` 统计永远相等。改为 `named_buffers()` 统计 alibi_slopes 总 numel。
- smoke: **全特性 CPU 训练验证**——config_smoke_features_v3.yaml 全特性开（v2 全特性 + input_highway + layer_contrastive + shared_alibi）。225 步训练：Train 8.58 / Val 10.10 / ~1.21s/step（CPU）。修复 input_highway bug 后 Val Loss 从 12.28→10.10（降 2 点，证明 bug 修复对训练质量有显著正向影响）。3 个 prompt 生成不崩溃（"中国人民"/"今天天气"/"科学技术"），内容质量受限于数据量和训练轮次。
- test: **新增回归测试**——第十四轮三特性共 14 个测试 + 3 个增量解码 bug 回归测试（input_highway x0 缓存/shape 对齐/与 cross_layer 组合 + shared_alibi survives to_device）。pytest **275 passed / 1 skipped**。

### 教训补充（本轮）
- **input_highway 增量解码须缓存 x0**：训练时 x0 = embedding(src) 是整条序列，但增量解码后续步 src 只 1 token，x0 = embedding(单 token) 会让 input_highway 注入错误内容。须首步缓存完整 x0，后续步用缓存值。
- **input_highway 增量解码须 shape 对齐**：即使缓存了 x0，后续步 x shape [B,1,D] 与 x0 shape [B,T_prompt,D] 不匹配，直接 broadcast 会让 x 被放大破坏后续层（特别是 cross_layer_routing 的 stack）。须 mean-pool x0 到 [B,1,D] 对齐。
- **shared_alibi 须在 .to(device) 后重新绑定**：PyTorch _apply 遍历每个 module 独立处理 buffer，会打破对象共享（数值仍正确但失去减参优势）。须重写 to() 方法在设备迁移后重新绑定。
- **性能优化：循环外预计算不变量**：input_highway_proj(x0) 中 x0 在层循环中不变，应循环外计算一次，避免每层重复 Linear。

## `2d154bf`（已推送，基于 `7f26bd4`，第十二/十三轮：跨层协作深化 + 冗余合并 + 全特性训练验证）

- feat: **层间 SSM 状态传递（cross_ssm_transfer）**——`models/transformer.py` TransformerModel forward 中 hybrid 块间传递 SSM 信息：前一个 hybrid 块输出经 cross_ssm_proj（init weight=0，弱注入）投影后加到下一个 hybrid 块输入。让 SSM 的序列理解在层间流动，而非每层独立重算。config: `cross_ssm_transfer`（需 hybrid 层）。
- feat: **渐进式残差（progressive_residual）**——残差门控值随层数按 1/√(depth) 衰减：浅层 gate≈1（保留信息），深层 gate≈0.5（激进变换）。平衡浅层信息保留与深层特征抽象。与 highway_gate 组合时 bias 按 3/√(depth) 衰减。config: `progressive_residual`。
- feat: **跨层 FiLM 调制（layer_film）**——浅层输出经 layer_film_projs[i] 产生 (γ,β)，对深层输入仿射调制 `x = x*(1+tanh(γ)) + β`。init γ=β=0（恒等，向后兼容）。tanh 限制 γ∈(-1,1) 防止深层堆叠数值爆炸。config: `layer_film`。
- feat: **动态残差门控（highway_gate）**——用 input-dependent gate `sigmoid(W·x+b)` 替代静态标量 residual_gate。init W=0, b=3.0（sigmoid≈0.95，平滑过渡）。逐 token 动态门控，比静态标量更灵活。与 residual_gate 互斥（highway_gate=True 时不创建 sub1_gate/ffn_gate，避免 dead params）。config: `highway_gate`。
- fix: **专用初始化被 _init_weights 覆盖**——cross_ssm_proj/cross_router/layer_film_projs/highway_gate 的专用初始化（weight=0/bias=-3 等）被通用 _init_weights（N(0,0.02)/zeros）覆盖。新增 `_apply_specialized_inits` 方法在 _init_weights 后重新应用专用初始化。
- fix: **progressive_residual + highway_gate 冲突**——highway_gate=True 时仍创建 sub1_gate/ffn_gate（dead params），且 progressive_residual 未作用于 highway_gate 的 bias。修复：highway_gate=True 时不创建静态 gate，并在 _apply_specialized_inits 中按 3/√(depth) 缩放 highway bias。
- fix: **KV 缓存嵌套 bug**——_run_attn_mixer 返回 ((k,v,linear_S,z), None, None) 三元组导致 BlockState.attn_kv 嵌套，增量解码崩溃。修复：_run_attn_mixer 只返回 (k,v,linear_S,z)。
- refactor: **_run_ssm 辅助方法提取**——TransformerBlock.forward 中 3 处重复的 SSM ckpt/non-ckpt 调用分支（ssm 块 / hybrid+ssm_as_memory / hybrid 并行）合并为 `_run_ssm(xn, past_state, past_conv_state, use_cache, ckpt)` 单一入口。
- refactor: **测试辅助函数参数冲突修复**——`_small_hybrid`/`_small_igmcg` 用 dict 合并模式替代直接传参，允许调用者覆盖 layer_plan/vocab_size 等默认键（原模式 `layer_plan='attn,hybrid', **over` 在调用者也传 layer_plan 时 TypeError）。
- smoke: **全特性 CPU 训练验证**——config_smoke_features_v2.yaml 全特性开（alibi+pe_gate+attn_linear+linear_correction+cross_layer+ssm_as_memory+QAT 8bit + cross_ssm_transfer+progressive_residual+layer_film+highway_gate）。225 步训练：Train 8.5452 / Val 8.1032 / ~1.3s/step（CPU）。模型可生成中文但质量受限于数据量和训练轮次。
- test: **新增回归测试**——KV 缓存嵌套 bug 回归、跨层 SSM 传递、渐进式残差、layer_film、highway_gate、专用初始化覆盖、progressive_residual+highway_gate 组合。pytest **259 passed / 1 skipped**（较基线 238 + 21）。

### 教训补充（本轮）
- **专用初始化须在 _init_weights 后重新应用**：PyTorch 的 apply(_init_weights) 会遍历所有子模块，包括之后创建的专用参数。若专用初始化在 apply 之前执行，会被通用 N(0,0.02) 覆盖。模式：__init__ 末尾 apply(_init_weights) → 然后 _apply_specialized_inits() 重应用专用初始化。
- **highway_gate 与 residual_gate 互斥须显式处理**：highway_gate=True 时不创建 sub1_gate/ffn_gate，否则这些参数成为 dead params（定义但不使用），既浪费内存又让 progressive_residual 失效（修改了不用的参数）。
- **测试辅助函数用 dict 合并而非直接传参**：`def f(**over): return g(k=v, **over)` 在 over 也含 k 时 TypeError；应改为 `kw={'k':v}; kw.update(over); return g(**kw)`。

## `7f26bd4`（已推送，基于 `cbb2df8`，第十一轮续：SSM 作记忆 + ngram 8层防护 + DML 全特性训练 + 性能优化）

- feat: **SSM 输出作隐式记忆（ssm_as_memory）**——`models/transformer.py` TransformerBlock hybrid 块新增 SSM-first 顺序：先算 SSM，把 ssm_h mean-pool 投影为单记忆槽（ssm_k_proj/ssm_v_proj → head_dim），注入注意力 mem_kv。让注意力能"查到"SSM 的序列理解，实现 SSM→Attention 的信息流。仅 hybrid 块生效；config: `ssm_as_memory`。5 个回归测试（参数创建/输出变化/梯度回流/cache parity/非 hybrid noop）。
- fix: **CrossLayerRouter DML 兼容**——原 route() 用 torch.gather + advanced indexing，在 DML 上报 "scatter doesn't allow partially modified dimensions"。三次重写最终用 mask+einsum 完全避免 gather/scatter：`mask = (scores >= threshold).float()` + `torch.einsum('bn,bntd->btd', gates, prev_stack)`。CPU + DML 均通过。
- feat: **n-gram 爆炸防护 5→8 层**——`models/transformer.py` _apply_ngram_fusion 新增 3 层防护：(0) ngram_ord nan_to_num 兜底（防 -inf/NaN 传播）；(0b) ngram_order_logits clamp [-10,10]（防 softmax 饱和失去多阶信息）；(3b) gate clamp [0,10]（防浮点误差使 gate·ngram_vec 超限）。3 个新回归测试。
- perf: **ngram NaN/Inf 检查优化**——原 `if torch.isnan(fused).any() or torch.isinf(fused).any()` 有两次全量 reduce + CPU 同步税，改为直接 `torch.nan_to_num`（一次操作，DML 原生 kernel，正常情况下 no-op）。
- perf: **pe_gate 冗余 .to(device) 删除**——`models/mixers.py` log_pe_gate 是 Parameter 已在设备上，`.to(bias.device)` 是冗余操作。
- perf: **QAT init std() 在 CPU 计算**——`models/qat.py` 原 `all_w.std()` 在 DML 上回退 CPU（aten::std.correction 不支持），改为直接在 CPU 上计算（`.detach().cpu()` 后 std），省一次 DML→CPU 回退。
- exp: **QAT 权重缓存尝试与回退**——尝试缓存权重量化整数值 r（round+clip 结果），用 mod.weight._version 检测变化。但 LSQ 可学步长下 _qat_scale 每步被优化器更新，r=round(w/s) 随 s 变化，缓存 r 会用过时 scale 导致 Val Loss 升高（9.22→12.70）。回退到原始 _fake_quant。教训：LSQ 可学步长下权重量化缓存不适用。
- smoke: **全特性 DML 训练验证（3 次）**——config_smoke_features_v2.yaml 全特性开（alibi+pe_gate+attn_linear+linear_correction+cross_layer+ssm_as_memory+QAT 8bit）。3 次训练结果：Train 8.37-9.48 / Val 9.22-10.94 / ~893-935ms/step。Val Loss 略升因新增 ngram 防护层限制模型灵活性（稳定性 trade-off）。速度 ~1.07-1.12 it/s，主要瓶颈在 QAT 双重量化（每 Linear 两次 _fake_quant）和 ssm_as_memory 串行化。
- test: **3 个 ngram 防护回归测试**——test_ngram_ord_nan_inf_sanitized / test_ngram_order_logits_clamped / test_ngram_gate_clamped。pytest **238 passed / 1 skipped**（较基线 235 + 3）。
- review: **2 个子代理审查**——架构审查（合并点+性能+合理性）和 bug 审查。无严重 bug，架构合理。合并建议：_small_* 测试辅助函数可整合（低优先级）；性能建议已部分实施。

### 教训补充（本轮）
- **QAT 权重缓存在 LSQ 下不适用**：_qat_scale 每步被优化器更新，r=round(w/s) 随 s 变化，缓存 r 会用过时 scale 导致 Val Loss 升高（9.22→12.70）。回退到原始 _fake_quant。
- **DML advanced indexing scatter 不稳定**：`base[k-1, idx] = vals` 在 DML 上可能报 "partially modified dimensions"（但隔离测试中又通过了）。CrossLayerRouter 的 gather/scatter 已改为 mask+einsum（DML 原生支持）。
- **ngram 防护需多层**：8 层防护从 ngram_ord 兜底到最终 logits clamp，缺任何一层都可能在极端输入下爆炸。

## `cbb2df8`（已推送，基于 `76a431d`，第十一轮 4 新特性 + product_key 回归 + 6 想法评估）

- feat: **线性注意力修正模式（linear_correction）**——`models/transformer.py` TransformerBlock 新增修正模式：主注意力 h 为基础，线性注意力 lh 提供"修正项"（lh - h），`h = h + sigmoid(correction_gate) * (lh - h)`。correction_gate init -1.0（sigmoid≈0.27）平滑过渡。相比原凸组合（mg·h+(1-mg)·lh），修正模式保留主注意力主体地位，线性注意力仅补充差异。config: `attn.linear_correction`。
- feat: **位置编码选择性门控（pe_gate）**——`models/mixers.py` SlidingWindowCausalSelfAttention 新增 per-head 可学强度控制 ALiBi 位置偏置：`pe_strength = 1.0 + tanh(log_pe_gate)`，init 0 → 1.0（精确向后兼容），范围 (0,2)。让模型自决每个头对位置信息的依赖。config: `attn.pe_gate`（需 alibi=True）。
- feat: **跨层稀疏路由（cross_layer_routing）**——`models/transformer.py` 新增 CrossLayerRouter 类（DenseNet 风格 top-k 跳跃连接）。每层 j 拥有路由器 Linear(D,1)，对前 j 层输出打分，选 top-k 个最高分前层，按 sigmoid(score) 加权累加注入当前层输入 x（残差+稀疏+选择性+输入相关）。init bias=-3（sigmoid≈0.05，弱注入不破坏预训练）。config: `cross_layer_routing` + `cross_layer_topk`。
- feat: **量化感知训练 QAT（qat_bits）**——`models/qat.py` 新模块：LSQ 风格可学习步长伪量化。前向 round+clip 量化权重+激活，反向 STE(x) + LSQ(scale) 梯度直通。共享可学习步长（per-tensor，1 标量参数）。eval 时恒等（推理无开销）。monkey-patch Linear.forward 不破坏 state_dict。`scripts/train.py` 集成：config.model.qat_bits>0 时启用。config: `qat_bits`（0=关闭，4/8=对应位宽）。
- test: **product_key/forget 跨块 divergence 回归测试**——`tests/test_new_mechanisms.py` 新增 4 测试：(1) product_key 默认关；(2) 非 product_key 无 forget 跨块 parity；(3) forget_gate 跨块 divergence 文档化（新发现：forget 衰减乘法不满足交换律）；(4) product_key 跨块 divergence 文档化。`models/memory.py` write() 注释补充 forget 跨块 divergence 根因分析。
- test: **18 个新特性单元测试**——t3 linear_correction（4 测试：参数创建/输出变化/梯度回流/cache parity）；t4 pe_gate（4 测试：参数创建/输出变化/梯度回流/init 向后兼容）；t5 cross_layer_routing（5 测试：参数创建/输出变化/梯度回流/cache parity/单层 noop）；t6 QAT（5 测试：参数注册/eval 恒等/训练改变输出/梯度回流/disable 恢复）。pytest **225 passed / 1 skipped**（较基线 203 + 22）。
- eval: **6 个创新想法可行性评估**——(1) Memory-Gated MoE：中高可行性，高收益，需专门训练调参（未来工作）；(2) 跨层记忆层级：已部分被 cross_layer_routing 覆盖；(3) SSM 状态作隐式记忆：高可行性低成本（~40行），但 inject_memory 形状耦合需谨慎（待实施）；(4) n-gram 触发定向检索：中可行性，n-gram+memory 耦合（未来工作）；(5) DifferentialMemory：中可行性，数值不稳定风险（未来工作）；(6) 自适应计算深度：已部分被 layer_skip + prune_layers 覆盖。
- smoke: **全特性训练验证**——4 新特性全开（linear_correction+pe_gate+cross_layer_routing+QAT 8bit）小模型 30 步训练：loss 5.30→5.29（下降），cache parity diff=0.00e+00（完美一致），QAT status 正确（scale=0.000157, qmax=127）。

## `（本地，基于 `f7bed96`，待推送，第十轮 fused SDPA + forget parity + 去重 + 创新想法）

- perf: **DML fused SDPA 全面启用（~22x 训练提速）**——`models/mixers.py` 原注释称 DML fused SDPA 崩溃，实测 torch 2.4.1 + torch_directml 0.2.5 已稳定可用。关键发现：DML bool attn_mask 语义与 PyTorch 标准相反（True=允许≠禁止），但 float attn_mask 与 is_causal=True 正确。全路径改用 fused SDPA（纯因果用 is_causal=True，有偏置用 float attn_mask）。训练速度 1500ms/step → 68ms/step（~22x），删除 .item() 同步后 54.9ms/step。pytest 203 passed。
- fix: **forget_gate train/infer parity**——`models/memory.py` 原 write() 每次 write() 调用施加一次 forget 衰减，训练（1 次 write T token = 1 次衰减）与增量解码（T 次 write = T 次衰减）衰减次数不同导致 divergence。修复：forget 衰减改为按 token 数施加（product_key 逐 token 循环内衰减；非 product_key 向量化 f^T·slots_0 + Σ f^{T-1-t}·update_t）。parity 测试：diff=4.47e-08（浮点误差级别，原显著发散）。
- refactor: **EnhancementsMixin 合并 set_enhancements_active**——SlidingWindowCausalSelfAttention / LinearMixerBase / DifferentialAttention 三类原各自重复实现 set_enhancements_active，统一到 EnhancementsMixin（nn.Module + mixin 多继承）。删除 3 份重复（~25 行）。
- refactor: **_pad_mem_bias 收口**——4 处重复的 `F.pad(mem_bias, (0, Tkv - mem_cols))` 提取到模块级 `_pad_mem_bias(mem_bias, Tkv, mem_cols)` 函数。
- refactor: **死代码清理**——删除 `_manual_attention`（fused SDPA 全面启用后不再被调用的 fallback）。
- perf: **logprob_orders_incremental 向量化**——`models/ngram.py` 原逐 (b,t) `.to(device)` 赋值，改为 stack + index_select 批量填充（仿 logprob_orders_matrix 模式）。消除 DML 启动税。
- refactor: **train/finetune 用 safe_torch_load**——`scripts/train.py`、`scripts/train_finetune.py` 原 `torch.load(weights_only=True)` 改为 `safe_torch_load`（带全局白名单的安全加载）。
- refactor: **generate.py 用 apply_cpu_threads**——原 `torch.set_num_threads` 改为 `apply_cpu_threads`（含边界保护）。
- docs: **记忆因果写速度评估**——逐 token 因果写（消除 train/infer memory divergence）开销 14x（0.21ms→3.20ms/write），不值得实施。当前 memory.write 的 train/infer 差异为已知 trade-off（记忆 K/V 对所有 query 共享，训练时 query t 可见 token t 自己的写入）。文档标注此限制。
- docs: **product_key 跨块顺序 divergence 标注**——product_key 模式下训练 block-major（block0 写全部 T token → block1）与增量 token-major（各 block 逐 token）slots 演化路径不同。opt-in 功能默认关，建议仅在单层或 train/infer 同序时使用。
- review: **多轮 bug 审查**——3 轮审查 + 核对：1 真 bug（forget_gate parity，已修）；1 性能优化（.item() 同步，已删）；其余误报（sampling/state/CJK Ext G 过滤/CAST 增量 A_delta 均确认正确）。
- exp: **可共享内容调研**——10 项发现：MambaSSMWithCAST.forward 重复（待 P1）、ngram 双路径骨架可抽（P2）、LinearAttention RNN 步可抽（P2）、AMP 决策可抽（P2）、DifferentialAttention mask 预分配可优化（P3）、optimizer state 迁移可抽（P3）等。已实施 P0/P1，余项按机会成本决定。

### 新架构创新想法（待实施评估）

1. **记忆驱动的动态路由 (Memory-Gated MoE)**：把 MemoryBank 的 M 个 slots 作为 "soft experts"，每个 query 根据与 slots 的相似度（现有 mlogits）动态路由到不同 FFN 计算路径。slots 既是压缩历史又是 expert 路由键，一物两用。相比标准 MoE 的固定 expert，记忆路由是内容驱动且随上下文演化的。

2. **跨层记忆层级**：不同层共享记忆但用不同 comp_dim（低层 comp_dim=8 细粒度、高层 comp_dim=32 语义），形成层级记忆金字塔。低层捕获局部模式、高层捕获全局主题，类似人脑短期/长期记忆分化。

3. **SSM 状态作为隐式记忆**：MambaSSM 的 h_t (B, d_inner, d_state) 本质是压缩历史，可把它的 state 作为额外记忆源注入 attention 的 KV（与 MemoryBank 并列）。SSM 状态是"连续演化"记忆，MemoryBank 是"离散槽"记忆，两者互补。

4. **n-gram 触发的定向检索**：n-gram 命中时不仅做概率融合，还触发对记忆槽的定向检索（用 n-gram 上下文作为 query 检索相关 slots）。让符号先验（n-gram）与神经记忆（MemoryBank）深度耦合，而非仅在 logits 层相加。

5. **DifferentialMemory：记忆差分检测信息更新**：用记忆的"旧版"（上一轮 slots）vs"新版"（本轮 slots）做差分，检测信息更新位置。差分大的位置 = 新信息注入点，可路由更多计算资源。借鉴 DifferentialAttention 的差分思想到记忆系统。

6. **自适应计算深度（基于记忆饱和度）**：用 MemoryBank 的 slots 饱和度（norm 或熵）作为 skip gate 信号——slots 未饱和时跳过深层计算（信息少），饱和时走完整深度。让模型根据"已压缩了多少信息"动态决定计算量。

## `（本地，基于 `f7bed96`，待推送，第九轮全面审查修复 + 性能优化）

- fix: **M3 DifferentialAttention+memory 形状崩溃**——`models/mixers.py` DifferentialAttention.forward 原 scores 先算（B,H,T,Tkv）后注入 memory，导致 `scores + mem_bias` 形状 (B,H,T,Tkv)+(B,H,T,M) 不符。重写为「先注入 K/V 扩展为 M+Tkv、再算 scores」顺序（与 SlidingWindowCausalSelfAttention 一致），mem_bias 右侧补零到 Tkv_full 后广播相加。新增 3 回归测试（前向不崩/训练模式不崩/cache parity）。
- fix: **M1 model.generate 温度丢失（ngram_fusion 路径）**——`models/sampling.py` `_decode_one_step` 在 `temperature_applied=True` 时曾把 τ 盖成 1.0 传给 forward，导致后续步温度丢失、采样分布首步冷后续步热。改为始终把真实 τ 传给 forward（由 forward 内部决定是否应用）。影响：τ=1.0 无变化，τ≠1.0 后续步采样分布正确。新增 2 回归测试（parity/τ≠1.0 输出不同于 τ=1.0）。
- fix: **DifferentialAttention 缺 set_enhancements_active 方法**——SEL 增强调度训练末尾对每个 block 调用，但 DifferentialAttention 未实现该方法导致崩溃。补齐方法 + forward 内 QK-Norm/temp 接入 `self._rt` 运行时开关（原仅静态 `self.qk_norm_enabled` 检查，SEL 设 false 时不生效）。
- perf: **logprob_orders_matrix 向量化（DML 提速 63x）**——`models/ngram.py` 原 step 3 用 `for i in range(U): mask = (flat_idx == i); flat_out[mask] = uniq_vecs[i]` 逐唯一上下文填充，DML 上每次都是 GPU 同步（U 大时极慢，83s/it）。改为 `torch.stack(uniq_vecs) + index_select` 一次性搬回 (B,T,V,K)，DML 上 83s/it→1.5s/it（63x），CPU 上 0.37s/it（无变化，内存峰值 U*V*K≤48MB 可接受）。
- perf: **ngram.py 传输优化（批量 .to(device) + 设备缓存）**——`logprob_orders_matrix` step 2 原对每个缓存命中上下文逐张 `.to(device)`，DML 上每张小传输有固定启动税（warmup 后 U_cached 大时累积开销可观）。改为 `torch.stack` + 单次 `.to(device)` + 按位置回填。`_compute_logprob_orders` 新增 `_ensure_dev_caches`：首次调用时一次性把 `uni_prob` + 全部 `_ngram_tensors` 传到 device 并缓存，后续直接查表无需逐次 `.to(device)`。数值完全一致，pytest 203 passed。
- fix: **B1-B4/M2/M4-M7/MINOR**——模型加载路径/transformer 层 bug 修复（详见之前各轮）。
- chore: **死脚本/死配置清理**——删 6 个死脚本（improve_data/convert_statements_to_qa/reformat_data/analyze_datasets/check_files/check_ckpt）+ 13 个零引用 yaml 配置。
- fix: **监控脚本 regex + argparse**——4 个监控脚本 regex 从 `Epoch \[(\d+)/(\d+)\].*训练损失` 修为实际 train.py 输出格式 `Epoch (\d+)/(\d+) \| Train Loss`，加 argparse 与 config-based epoch 读取。
- fix: **repetition_penalty 一致性**——3 处不一致（1.2/1.4/2.0）统一为 2.0（tuning 实验值），删除 5 个 yaml 的死 `generation:` 节。
- docs: 8 个 README/guide 修复（README.md/models/README.md/MODEL_USAGE_GUIDE/experiments/tools/scripts/tuning/AGENT_MEMORY）。
- test: **新增 16+5=21 测试**（utils/checkpoint 基础 + BUG-9/10 回归 + M3×3 + M1×2 + 向量化 parity）。pytest **203 passed / 1 skipped**（较基线 181 + 22）。
- exp: **DML 全特性训练验证**——`config_smoke_features.yaml`（diff+memory+ngram_fusion+qk_norm+attn_temp+residual_gate）4000 行 1 epoch：Val Loss 5.74，生成输出有中文短语结构（"大家用客户是中国际内"、"全国特学习"）。83s/it 优化后 1.5s/it。

## `（本地，基于 `fb4e8e2`，待推送）

- exp: **20M 级大模型速度扫描**（最优基线 hd16/mem32/ngram+window 保持，仅放大 ed/nl）：ed512_nl6(14.8M/24.2k) / ed512_nl8(18.7M/18.2k) / **ed640_nl6(20.6M/20.9k tok/s，~20M 最快)** / ed768_nl6(27.2M/18.3k) / ed512_nl10(22.7M/15.7k)。结论：放大 embedding_dim 比加深层数更划算（nl10 仅 15.7k）；**ed640_nl6 为 20M 容量档推荐**。质量验证：ed640_nl6 单 epoch val_loss=8.98（4000 行小数据欠拟合），故 **4.3M 质量档仍是默认甜点**，20M 档需多 epoch/大数据。
- chore: **清理旧 checkpoint 释放 ~415MB**——删除 `checkpoints_50mb_dml` / `checkpoints_baseline_dml` / `checkpoints_cmp_{enh,sel,selv2}_full` / `logs_dml` / `logs_smoke_8k`；保留 `checkpoints/`（默认）、`checkpoints_full_dml/`（当前最优配置产出）、`checkpoints_smoke_4k/`（pytest 依赖）。

## `（本地，基于 `fb4e8e2`，待推送，第四轮训练超参扫描）

- exp: **训练超参扫描**（4.3M 质量档）：**epochs 是质量第一杠杆**（方向性）。⚠️ **更正**：前几轮（§24~§27）扫描脚本把文件路径字符串误传给 `TextDataset`（应为 `load_data()` 的文本行列表），导致只在 19 个「路径字符」序列上训练，所有此前 val_loss 绝对值（8.02/5.21/2.80/3.61 等）均作废；**速度数字（tok/s）不受影响**；方向性结论仍有效。真实数据（4000 序列）重测：1ep val=5.19 → 3ep val=4.13。
- 生成验证（3 epoch 真实数据）：`中国的首都是`→"真实习近日，由于什么那么一身后，由《三星》），北京北..."；能产出真实中文词/短语。3 epoch 明显优于早前 1 epoch 碎片化。
- config: `config_full_dml.yaml` 改 `epochs: 3` / `sgd_learning_rate: 0.3` / `warmup_steps: 0.0`（并修一处重复 batch_size 行）。
- chore: 删 `checkpoints/final_model.pt`+`vocab.json`（7/11 架构 overhaul 前、无 config、与现代码不兼容，84MB）；保留 `checkpoints/`(空) / `checkpoints_full_dml/` / `checkpoints_smoke_4k/`(pytest 依赖)。
- 验证：pytest 104 passed（1 skipped）。

## `（本地，基于 `fb4e8e2`，待推送，第三轮特性开关扫描 + rope bug 修复）

- exp: **特性开关扫描**（4.3M 质量档基线，单 epoch val_loss + tok/s）：char_merge=true(vl5.96 vs 6.32 提质量)、complexity_reward=true(λ0.05, vl5.87 vs 6.09 提质量)、alibi=true(vl6.18 vs 6.51 略提)、ngram_fusion(true 中性略慢)、learn_window(true 质量略降但提速特性)。
- **BUG 修复：`rope_learnable=true` 多步训练崩溃**（`RuntimeError: backward through graph a second time`）。根因：`RotaryEmbedding._get_cos_sin` 把带 grad 的 cos/sin 张量存入实例缓存，第二步 backward 复用第一步已释放的图。修复：缓存只存「无 grad 基准表」（inv_freq buffer 算 + detach），可学路径每步按 `rope_log_scale` 重算 cos/sin（梯度回流已验证非零）。修复后 `rope_learnable=true` 多步训练不崩、参数真正被训练。
- test: 新增 `test_learnable_rope_multistep_training_no_graph_reuse`（多步训练不崩 + rope_log_scale 收到非零梯度）。pytest 105 passed。
- 组合：20M(mem0) 全开安全增强(alibi+complexity+char_merge+ngram) val 9.90→8.83 明确改善。
- config: `config_full_dml.yaml` 新增 `alibi: true` + `complexity_reward: true`（配 `training.complexity_lambda: 0.05`）+ `rope_learnable: true`（bug 修复后开启）。
- 验证：pytest 105 passed + Python 编译。

## `（本地，基于 `fb4e8e2`，待推送，第二轮 20M 维度扫描）

- exp: **20M 维度扫描找新合适点**（固定 ed640_nl6，扫 head_dim/num_heads/memory/hidden_dim/window + 组合）：
  - head_dim：16(17.3k/vl12.5) > 24 > 32(14.5k/12.7) → **hd16 仍最优**。
  - num_heads：8(15.9k/vl12.4 最佳) > 16 > 4(17.2k/13.9 最差)，nh8 质量略优但慢 ~10%，不划算。
  - **memory_size 关键反转**：mem0(20.4k/**vl8.5**) >> mem16≈mem32≈mem64(vl13.4~13.6)。→ **20M 大模型在 4000 行小数据下严重过拟合 memory 机制，关掉 memory 质量暴涨、速度也更快**；与 4.3M 档（mem32 最优）结论相反。
  - hidden_dim 加大只增参减速、质量不升；window 下 w64 > w32 > w16。
  - **20M 最优配置修正**：`ed640/nl6/num_heads=4/memory_size=0/hd16/w64` → 20.3M / ~21k tok/s / val ~8.7（比 mem32 版 13.6 大幅改善）。**核心：配置最优值随规模变化——小模型靠 memory，大模型有限数据下关 memory**。
- docs: `config_full_dml.yaml` 注释更新 20M 容量档为 mem0 版。

## `（本地，基于 `d941095`，待推送）

- feat: **LinearAttention 可配置 head_dim**——新增 `linear_attn_head_dim` 参数（三级透传：LinearAttention / TransformerBlock / config_loader）；qkv 投影改为 `3*num_heads*head_dim`，proj 改为 `num_heads*head_dim→dim`。默认 `16`（AMD 780M iGPU DML 上比原默认 64 **快 1.75x** 且质量持平：中间张量 33.6MB→2MB 解除内存带宽瓶颈）。
- exp: **DML 配置扫描（速度 + 1-epoch val_loss 双指标）** 定 Pareto 最优：
  - `linear_attn_head_dim`：16(61k/7.99) > 32(53k/8.01) > 64(35k)。→ 定 16。
  - `memory_size`：32 质量最优(6.05)，0 最快(73k/6.10)。→ 定 32（特性+质量双赢）。
  - `embedding_dim`：256 为"不退化"最大维(4.3M/61k/6.02)，320+ 过拟合崩(8.03/9.37)。→ 定 256。
  - `num_layers`：4 质量最优(5.89) 且速度可接受(62k)。→ 定 4。
  - `ngram_fusion`：开(true) 轻微质量+特性增益、速度几乎不变。→ **默认开**。
  - `learn_window`：单独开 +15% 提速（64k→93k 量级）。→ **默认开**。
  - `layer_skip`：训练期负收益（58k/6.35），属推理静态剪枝特性，训练期反添开销。→ **保持关**（推理时 `prune_layers` 生效）。
- config: **`configs/config_full_dml.yaml` 更新为扫描最优**（ed256/nl4/mem32/hd16/ngram_fusion+learn_window 开）；端到端验证 4.28M params / **91.6k tok/s / val_loss 6.20**（对比最初 hd64 729 tok/s → **提速 128x**）。
- chore: 清理扫描实验脚本（`_sweep*.py` / `_exp_*.py` / 临时结果 json）。
- 验证：pytest 103 passed + `py_compile` 全过。

## `（本地，基于 `2b6fafb`）

- fix: **LinearAttention elu→relu（DML 兼容）**——`LinearAttention._feat()` 默认 `elu(x)+1`，DML 不支持 `aten::elu.out` → 每步 CPU 回退（~100ms 固定税）。改默认 `feature='relu'`（`relu(x)+1e-6`），新增 config `linear_attn_feature` 可配置；`TransformerModel.__init__` 新增 `linear_attn_feature` 参数透传至 `attn_kwargs`；`TransformerBlock` 构建 LinearAttention 时取 `attn_kwargs['linear_attn_feature']`，并通过 `attn_only` 过滤避免泄漏至 `SlidingWindowCausalSelfAttention`。
- test: 新增 `test_linear_attention_relu_feature`（验证 relu 特征映射非负 + 输出形状）、`test_hybrid_block_no_leak_attn_kwargs`（验证 hybrid block 构建不报 TypeError + linear_attn.feature='relu'）。pytest 103 passed。
- chore: 清理旧 checkpoint 目录（`checkpoints_50mb*`, `checkpoints_dml`, `checkpoints_hybrid*`, `checkpoints_ngram_smoke`, `checkpoints_smoke`）；保留 `checkpoints`/`checkpoints_full_dml`/`checkpoints_baseline_dml`/`checkpoints_smoke_4k`/`checkpoints_cmp_*_full`。

## `2b6fafb`（已推送，基于 `690e57a`）

- fix: **第四轮子代理审查修复**（3 子代理并行审查 + 证据锚定验证）：
  - BUG-7: `_generate_candidates_batch` 未重置 `_ngram_last_ids`，多次调用 `generate_igmcg` 时前一序列的 n-gram 上下文污染当前生成。修复：初始 forward 前 `model._ngram_last_ids = None`。
  - BUG-8: `train.py` 互斥校验 `enhancement_off_prob > 0` 返回 bool，`bool is not None` 恒 True → 任何单一策略配置都误报冲突。修复：改为 `sum([schedule is not None, off_prob > 0, curriculum is not None])`。
  - BUG-9: `train.py` resume 时 `optimizer.load_state_dict` 直接赋值 CPU 张量，首步 `optimizer.step()` 在 CUDA/DML 上崩溃。修复：加载后遍历 state 迁移至 `device`。
  - BUG-10: `save_checkpoint` 未保存 GradScaler state，fp16 resume 丢失已调整的 scale factor 导致梯度溢出。修复：新增 `scaler` 参数 + resume 后恢复。
- test: 新增 2 个回归测试（`test_ngram_last_ids_reset_across_candidates` + `test_enhancement_mutex_check`）。
- 验证：pytest 98 passed + `py_compile` 全过。

## `81c9949`（已推送，基于 `9d62be6`）

- fix: **KV 缓存 present 修复（BUG-6）**——原第三轮修复将 `present=(k,v)` 改为 `present=(k_token,v_token)`（仅存原始 token KV），意图避免 memory 进入缓存。但 `k_token` 在 past_kv 拼接前赋值，导致 `present` 只存当前 token 的 KV（形状始终 `[B,H,1,D]`），下一步的 `past_kv` 丢失全部历史 → 增量路径第 3 步起注意力仅看当前 token，全量/增量 max_diff≈0.09。修复：`present=(k,v)` 移至 past_kv 拼接之后、memory 拼接之前，存储累积的 token KV（不含 memory）。
- fix: **MambaSSM D_init 还原**——`_init_weights` 中 `nn.init.ones_(self.D)` 后补充 `self.D.data.mul_(self.D_init)`，确保 D 按构造函数传入的 `d_init` 初始化而非固定 1.0。
- docs: n-gram 滚动缓冲注释"末 2 token"→"末 ctx_len token"。
- 验证：pytest 96 passed + generation pipeline 12 passed。

## `9d62be6`（已推送，基于 `f5e8ba0`）

- fix: **第三轮子代理审查修复**（用户架构分析逐条验证 + 三轮子代理审查结果）：
  - M1: `_init_weights` 跳过 `char_merge.gate` bias 零初始化，保留 `gate_bias_init=-1.0` 设计意图。
  - M2: `_full_retrieval_bias` keep mask 改为 per-query `(Tq,Treal)` 窗口掩码 `keep[q,k] = (q-k<=window) & (k<=q)`，修复早期 query 窗口内 key 被 top-k 丢弃。
  - M3: `sample_step` 统一为加性频率惩罚（与 IGMCG 路径一致），修复两条路径惩罚不一致。
  - M4: LinearAttention 增加 `z_all=cumsum(kf)` 分母累积 + present 扩展为4元组 `(k, v, S_final, z_final)`，修复 cumsum 分母缺失。
  - MINOR-1: `S_final` 死三元表达式清理。
- 验证：pytest 96 passed + generation pipeline 12 passed。

## `f5e8ba0`（已推送，基于 `9d62be6` 之前的主干）

- feat: **8.9 架构升级**（n-gram max_order 3→10、LinearAttention 向量化、MemoryBank per-slot 遗忘门、重复惩罚改加性、RoPE 绑定、CharMerge 初始化、src_mask 清理）：
  - n-gram max_order 3→10：`NGramModel` 泛化存储 `self.ngrams[order][context_tuple]`；`_compute_logprob_orders` / `logprob_orders_matrix/incremental` 泛化上下文窗口；`_ngram_last_ids` 缓冲长度 = `max_order-1`；`train.py`/`generate.py`/`load_model` 统一 `max_order=10`。
  - ARCH-2 intuition 透传修复：`_generate_candidates_batch` 新增 `intuition` 参数 + `(N,7)` 广播。
  - LinearAttention 向量化：`for t in range(T)` → `torch.cumsum`（分子+分母）。
  - MemoryBank per-slot 遗忘门：`nn.Parameter(torch.zeros(1))` → `nn.Parameter(torch.zeros(M))`。
  - 重复惩罚改加性：`generate.py` 乘性 → 加性 `logit -= penalty × freq`。
  - RoPE max_len 绑定：`TransformerBlock` 接收 `rope_max_len`。
  - CharMerge gate.bias 初始化：`gate_bias_init=-1.0`，`sigmoid(-1)≈0.27`。
  - BUG-5 src_mask 删除。
- 验证：pytest 96 passed + generation pipeline 12 passed。

## 结构清理（本地）

- refactor: 实验脚本 `experiments/` 下 26 个临时/一次性脚本移入 `archive_unused/experiments_legacy/`（保留 `_bench_enh` / `_bench_speed` / `_cmp_sel_full` / `_smoke_gen_compare` / `_smoke_8k_gen` / `_run_train` / `_run_train_cpu` 共 7 个常用脚本）。`_bench_speed.py` 改为通过环境变量 `BENCH_MODEL` / `BENCH_VOCAB` 指定权重，去掉对已删除权重的硬编码路径。
- docs: `QUICK_START.md` 内容并入 `README.md`「完整快速开始」章节后删除。
- refactor: 数据脚本统一单入口 `scripts/data_manager.py`（子命令 `merge` / `stats` / `vocab` / `sample` / `to-jsonl`）；`scripts/merge_data.py` 与 `scripts/process_data.py` 改为其兼容薄包装，更新 README / scripts/README / docs/DATA_USAGE / data/README 引用。
- feat: 新增架构增强机制（默认关闭、向后兼容旧权重）：可学习遗忘 MemoryBank(memory_forget)、可学 RoPE+ALiBi(rope_learnable/alibi)、全上下文检索(memory_retrieval_full/topk)、可学滑动窗口(learn_window/window_base)、选择性跳过层(layer_skip)、线性注意力 mixer(mixer=attn/linear/hybrid)、计算复杂度奖励(training.complexity_lambda)、统一记忆预算(memory_budget)。修复：MemoryBank 首步设备对齐、mask_fill_value 透传、generate.py bf16 检测、window==0 主序列因果泄漏；`train_finetune.py` 读取 config`training` 的优化器/学习率/轮数；`requirements.txt` torch 上界 `<2.5`→`<2.2`；新增 `tests/test_new_mechanisms.py`（53 passed）。

## `7000f7f`（本地，基于 `3ec3107`）

- feat: **架构增强 2.0 在 50MB 数据上落地 + DML 训练大幅提速**。阶段1 学习型分词(char_merge) + 阶段2 可学习压缩记忆(MemoryBank, 64 槽) + 阶段3 可学习检索门控/稀疏注意力 已接入 `models/transformer.py`（bead576/be6f91f/3ec3107，此前未推送），本次补齐 50MB 探流程配置与提速优化并统一提交。
- perf: DML(iGPU) 训练提速约 **3x**（50MB 由 ~30 分 → ~11 分）。关键优化：
  1. `gradient_checkpointing` 关（小模型反向重算纯浪费，+47%）；
  2. `attn_window` 128→64、训练 `max_seq_length` 256→128——记忆做全局、短窗口做局部，且更低复杂度（seq128 时注意力计算约 4x 下降）；
  3. `SlidingWindowCausalSelfAttention.attend` 训练路径**静态偏置掩码加 `_bias_cache` 缓存**（窗口/因果掩码不再每步每头重建 arange/zeros/cat），并以权重设备为权威统一 DML 设备别名(`privateuseone` vs `:0`)，消除每步 `.to()` 拷贝与 `_build_masks` 失效重建；
  4. `MemoryBank.reset/get_kv/write` 热路径移除 `.to(dev)` 拷贝；
  5. **FFN `hidden_dim` 1024→768**（+47% tok/s，仅少 13% 参数，FFN 4x 扩张对字符级小模型过度配置）；
  6. `batch_size` 8→24（tok/s 在 bs16 后持平、bs32 OOM，24 为甜点；DML 实际可用显存远小于 8GB 标称，超则 TDR 设备重置）；
  7. **新增 SGD 优化器支持**（`scripts/train.py` 优化器工厂读 `training.optimizer` ∈ {adamw/sgd/adam}），去 AdamW 每步 ~96ms 的 CPU `lerp` 回退税（DML 缺该 kernel）——SGD 更新留在 GPU。修复 SGD 调度基准 lr 被 `learning_rate`(5e-4) 覆盖回 AdamW 量级的 bug（`opt_base_lr` 跟随优化器实际初始 lr）。
- config: 新增/调整 `configs/config_char_50mb_dml.yaml`（seq128/window64/ckpt OFF/hid768/bs24/SGD lr0.1/memory 64+稀疏32）、`configs/config_char_50mb.yaml`、`configs/config_char_50mb_cpu.yaml`；新增 `scripts/gen_50mb.py`（UTF-8 生成验证，绕开 GBK 终端显 `?`）。`pytest tests/` 27 passed。
- 验证：50MB DML 训练 `Speed: 3925 tok/s`、`Loss 325→73`（batch 60→150 平滑下降）；重复惩罚 1.7 将生成重复从 50+ 次压到 9 次。注：上下文训练 256(位置)+窗口64+记忆64；RoPE 支持长度外推（生成 300 token 不崩）。**全部提交未推送**（代理 `192.168.1.13:8080` 不稳定，需有效系统代理才能 push）。
  > ⚠️ **Loss 数字需重训**：该训练结果在 CharMerge 非因果泄露 + 滑动窗口因果泄露 + 记忆每步 reset 三重 bug 叠加下产出，loss 数字有水分。吞吐数字（tok/s）仍有效。需在 `74454a8`/`66159e8`/`9b76cce` 修复后重训才能获得可信 loss。

## `42323fb`（已推送，基于 `5db434b`）

- exp: 增强 vs 基线 受控对比扩展至 **全量 `merged.txt`（39700 行）**；SEL 改用 **SELv2 8 段掩码**（1 全开 + 1 全关极端 + 6 局部；attn_temp 仅全关段关、平时恒开）取代旧 8 段（无全关）与常开 ENH。同参数（1 epoch / batch32 / seq64 / lr3e-3 / seed42 / test_split0.1）DML fp32。`configs/config_cmp_{enh,sel,selv2}_full.yaml`，`experiments/_cmp_sel_full.py` 复现，`run_full_cmp.ps1` 一键三模型顺序训练。
- 结果：**ENH Val 5.3762 / SEL旧 5.4492 / SELv2 5.4969**——Val 仍常开 ENH 最低、SELv2 最高（差距与 20k 同量级）。鲁棒性 self-loss：ENH 在温和设置（T=0.8 / K=30 / K=100）最优；**SELv2 在极端采样温度最优（T=0.5=2.70、T=1.4=4.14）**，SEL旧在 T=1.1/K=10 最优。结论：**数据量越大 SEL 越接近常开 ENH；且 SELv2（含全关极端）在分布外/极端采样下泛化最好**，印证"SEL 泛化更好"的直觉——20k 时信号太弱且噪声大（SELv2 处处最差），全量下才显形。原始数据见 `experiments/cmp_sel_full.txt`（生成原文亦在其中）。
- feat: **默认训练方式改为 SELv2**——`configs/pretrain.yaml`（train.py 默认配置）与 `configs/config_dml_full.yaml` 新增 `training.enhancement_schedule`（SELv2 8 段）。今后 `python scripts/train.py`（无 `--config`）即按 SELv2 分段选择性增强训练；`validate` 仍强制全开。
- chore: 清理 20k 对比产物——删除 `configs/config_cmp_enh_20k.yaml` / `config_cmp_sel_20k.yaml`、`experiments/cmp_20k.txt` / `_cmp_20k.py` 及相关 20k 权重（`checkpoints_cmp_enh_20k/` / `checkpoints_cmp_sel_20k/` / `checkpoints_cmp_sel_old20k/`）；`.gitignore` 仅保留全量对比权重目录（`checkpoints_cmp_{enh,sel,selv2}_full/`、`logs_cmp_*_full/`）。以全量对比产物取代旧 20k 产物。

## `2741989`（已推送，基于 `7df19b6`）

- exp: 增强 vs 基线 受控对比扩展至 20000 行（`data/pretrain_corpus/merged_sample_20k_top.txt` = merged.txt 前 20000 行，确定性）。SEL 改用**新版 8 段掩码**（`enhancement_schedule`：attn_temp 恒开；qk_norm/residual_gate 段间切换、永不同时关、偏向全开 4/8），取代旧 4 段。`configs/config_cmp_enh_20k.yaml` / `config_cmp_sel_20k.yaml`，同参数（1 epoch / batch32 / seq64 / lr3e-3 / seed42 / test_split0.1）DML fp32。
- 结果：ENH(常开) **Val 6.2188 / ~3278 tok/s**，SEL(8 段选择性交替) **Val 6.2808 / ~3471 tok/s**——质量差距从 8000 行的 ~0.12 缩到 **0.062（几乎持平）**，SEL 训练快约 7%，但**推理无提速**（两权重解码均全开：top-k ~38、IGMCG ~10.8 tok/s）。鲁棒性 self-loss：8 档中 ENH 6 档略优、SEL 2 档略优，差距 ~0.1–0.3，基本打平。结论：数据量越大 SEL 越接近常开 ENH，仅省训练时间、不省推理；两套 20k 权重均保留作对照。原始数据见 `experiments/cmp_20k.txt`（`experiments/_cmp_20k.py` 复现）。
- chore: 清理旧的 8000 行对比产物——删除 `configs/config_cmp_enh.yaml` / `config_cmp_sel.yaml`、`experiments/_robust_enh_sel.py` / `robust_enh_sel.txt` / `cmp_4way.txt`，及本地 `checkpoints_cmp_enh/` / `checkpoints_cmp_sel/` 权重；`.gitignore` 仅保留 20k 两项（`checkpoints_cmp_enh_20k/` / `checkpoints_cmp_sel_20k/`）。`pytest` 27 passed（本次未改训练链路）。

## `99055cb`（已推送，基于 `aa9d757`）

- test: 鲁棒性探针 `experiments/_robust_enh_sel.py`——对 ENH/SEL 在 `temperature∈{0.5,0.8,1.1,1.4}` 与 `top_k∈{10,30,100}`（固定 `repetition_penalty=1.4`）下生成，计算 **self-loss**（模型对自身生成续写的 cross-entropy）作稳健性代理，并逐提示词记录原文。结果：**ENH 在全部设置下 self-loss 均略低于 SEL（更自洽/稳健）**，差距很小（约 0.1–0.3 nat）；该 8000 行/1 epoch 规模下两者都偏弱。配合肉眼观察：ENH≈SEL > (已删)ALT，与 Val 排序（ENH 7.10 < SEL 7.22 < ALT 7.35）一致。结论：常开 ENH 鲁棒性略优，SEL 接近且训练更快。原始数据见 `experiments/robust_enh_sel.txt`（该 8k 探针脚本与结果已于 `2741989` 随 20k 对比清理；20k 对照见 `experiments/cmp_20k.txt`）。
- chore: 清理无用的 BASE/ALT 产物——删除 BASE/ALT 模型（`checkpoints_cmp_base/`、`checkpoints_cmp_alt/`）、`configs/config_cmp_base.yaml`、`experiments/_cmp_enh_base.py`+`experiments/cmp_enh_base.txt`（BASE 专属，已被 `cmp_4way.txt` 覆盖）、`experiments/_cmp_4way.py`（引用已删模型、失效）；删除 `logs_cmp_*` 训练日志。现仅保留 ENH 与 SEL 模型及其配置（`config_cmp_enh.yaml`/`config_cmp_sel.yaml`）。`experiments/cmp_4way.txt` 作为历史四方对比快照保留。**（注：上述 8k 的 ENH/SEL 配置、`cmp_4way.txt` 与 `_robust_enh_sel.py` 已随 20k 对比在 `2741989` 清理，现仓库仅保留 20k 版本 `config_cmp_{enh,sel}_20k.yaml` 与 `cmp_20k.txt`。）**

## `290b790`（已推送，基于 `3f8c2c8`）

- feat: `set_enhancements_active` 由仅全开/全关（`bool`）升级为**按开关粒度 dict**（如 `{"qk_norm": False, "residual_gate": True}`）；`TransformerModel`/`TransformerBlock`/`SlidingWindowCausalSelfAttention` 各自只更新自身键。`scripts/train.py` 新增 `training.enhancement_schedule`（分段掩码列表，按 `batch_idx % len` 循环切换；缺省键补 `True`），与旧式整体随机 `enhancement_off_prob` 互斥（schedule 优先）。多段循环仅切几个布尔开关，**无额外开销**。
- 设计：**attn_temp 开销可忽略且恒有益 → 始终开**；qk_norm/residual_gate/hybrid_gate 有非忽略开销 → 在 4 段掩码间循环切换，模型**永不完全“全关”**（始终至少保留 attn_temp，且在分段间只切换部分增强）。`validate` 仍强制全开。
- 三方→四方对比（复用 BASE/ENH/ALT 权重，仅新训 SEL；8000 行×1 epoch，DML fp32，含②）：SEL **Val 7.2254 / ~3413 tok/s** vs ALT **Val 7.3543 / ~3596**；SEL 明显优于 ALT 且接近 ENH(7.10)。**结论：弃用 ALT、采用 SEL**（分段选择性比整体 50% 全关更能保住增强收益）。
- 入库：`configs/config_cmp_sel.yaml`、`experiments/_cmp_4way.py`、`experiments/cmp_4way.txt`（4 组含原文）；删除 `config_cmp_alt.yaml` / `experiments/_cmp_3way.py` / `experiments/cmp_3way.txt`（被 4 方对比取代）；`checkpoints_cmp_alt/` / `checkpoints_cmp_sel/` 加入 `.gitignore`。`pytest` 27 passed。（上述 8k 的 `config_cmp_sel.yaml`/`cmp_4way.txt` 已于 `2741989` 清理，现仅保留 20k 版本。）

## `3f8c2c8`（已推送，基于 `b5da36a`/`23a09ff`）

### 优化②：重构梯度检查点边界，消除加速开销
- perf: `SlidingWindowCausalSelfAttention.forward` 拆为 `project_and_norm`（廉价：QKV 投影 + QK-Norm + 温度 + RoPE）与 `attend`（重算力：scores/softmax/proj）；`TransformerBlock.forward` 仅对 `attn.attend`/`ssm`/`ffn` 做 `checkpoint`，`ln1`/`ln2`/门控在重算区外只跑一次；同步移除模型层 `forward` 对整块展开的重复包裹，新增 `set_gradient_checkpointing`。
- 根因消除：`gradient_checkpointing=True` 原本在反向重算整个 block 前向，使廉价增强算子执行两次（约 +28%）；② 后增强训练开销大幅下降。
- 实测（DML 微基准）：`all` 增强单步 147.8 → **132.5 ms**（约 -10%）。训练吞吐（8000 行×1 epoch）：ENH ~2374 → **~3424 tok/s**（+44%），与 BASE 差距由 ~28% 缩至 ~8%。

### 交替增强训练（按批次开关增强）
- feat: `TransformerModel.set_enhancements_active(bool)` / `TransformerBlock.set_enhancements_active` / `SlidingWindowCausalSelfAttention` 的 `_enh_active` 运行时开关；关闭时跳过 QK-Norm/温度/门控（恒等）。
- `scripts/train.py`：新增 `training.enhancement_off_prob`（缺省 0），`train_epoch` 按批次 `random.random() >= prob` 切换增强开关；`validate` 强制开。`configs/config_cmp_alt.yaml`（enh + `enhancement_off_prob: 0.5`）。
- A/B/ALT 三方对比（8000 行×1 epoch，DML fp32，含②）：
  - 训练 Val Loss / 吞吐：BASE 7.8300 / ~3735；ENH 7.1021 / ~3424；ALT 7.3543 / ~3596 tok/s。
  - 结论：交替（off=0.5）比常开快约 +5% 训练，但 Val Loss 回升到 7.35，**介于 BASE 与 ENH 之间**，属速度/质量折中（非免费提速）；② 已把“常开”开销压到很小，质量优先推荐常开。
  - 生成解码（推理时 ALT 同常开）：top-k BASE 40.1 / ENH 35.4 / ALT 36.2；IGMCG BASE 12.5 / ENH 10.8 / ALT 10.7 tok/s。
  - 结果入库：`experiments/cmp_3way.txt`（含原文）+ 新增 `experiments/_cmp_3way.py`；权重不入库（可由配置+数据复现）。

## `b5da36a`（已推送，基于 `f9452ed`）

### 架构增强默认开启 + 性能定位/微优
- feat: `config_loader.build_model` 的四开关（`qk_norm`/`attn_temp`/`residual_gate`/`hybrid_gate`）默认值改为 `True`，新训练默认即开增强，无需手动开启。
- 向后兼容：`scripts/generate.py` 的 `load_model` 仍从各 checkpoint 的 `*_config.yaml` 读取开关、缺省关，旧权重（开启增强前训练）仍可加载（修复 `test_quantization_load` 回归）。
- 微优：`attn_temp` 融合为单次标量乘法 `q *= exp(-0.5*log_temp)`，免去额外 `sqrt`。
- 性能定位：新增 `experiments/_bench_enh.py`（DML 微基准）定位增强训练变慢根因——`gradient_checkpointing=True` 反向重算使廉价增强算子执行两次，单步约 +28%；关掉检查点后基线本身快约 22%。增强对比结果 `experiments/cmp_enh_base.txt` 入库（权重较大不入库，可由 config+data 复现）。

## `f9452ed`（已推送，基于 `462a849`）

### 修复：load_model 透传架构增强标志 + 增强/基线 A/B 对比
- fix: `scripts/generate.py` 的 `load_model` 重建 `TransformerModel` 时未传入 `qk_norm`/`attn_temp`/`residual_gate`/`hybrid_gate`，导致开启增强训练的 checkpoint 因 state_dict 多出子层参数而无法加载；现从 `*_config.yaml` 读取并透传，向后兼容旧权重。
- 新增受控 A/B 对比配置与脚本：`configs/config_cmp_base.yaml`（四开关全关）、`configs/config_cmp_enh.yaml`（四开关全开）、`experiments/_cmp_enh_base.py`（同数据多提示词生成对比，结果写 `experiments/cmp_enh_base.txt`）。
- 实测（8000 行×1 epoch，DML fp32）：ENH Val Loss 7.22 vs BASE 7.72；训练吞吐 ~2370 vs ~3270 tok/s（慢约 28%）；生成解码开销可忽略（top-k/IGMCG 约 0.95–0.98x）；增强版文本略更连贯。

## `462a849`（已推送，基于 `adfd63d`）

### 性能 / 可观测性：训练速度日志 + 8000 行冒烟
- `scripts/train.py`：每 10 batch 打印 `Speed: xxx tok/s`（累计 token 数 / 耗时）；新增 `training.show_progress` 配置开关（关闭则走打印分支，配合 `python -u` 实时落盘），便于记录训练速度。
- `configs/config_smoke_8k.yaml`：8000 行小训练配置（`data/pretrain_corpus/merged_sample_8k.txt` 由 `merged.txt` 随机采 8000 行）。
- `experiments/_smoke_8k_gen.py` + `smoke_8k_gen.txt`：IGMCG 开 / 关生成速度与原文对比（DML 上基线 top-k ~40 tok/s、IGMCG ~12 tok/s，约 0.31x）。

## `adfd63d`（已推送，基于 `92f669e`）

### 功能：可配置架构增强（默认关、向后兼容）
- `SlidingWindowCausalSelfAttention`：可选 **QK-Norm**（RMSNorm）与**可学习每层注意力温度**。
- `TransformerBlock`：**门控残差**（`residual_gate`）与**混合路径门控**（`hybrid_gate`：hybrid 块内 attn/ssm 两路各自可学习门控）。
- 全部经 `config['model']` 的 `qk_norm` / `attn_temp` / `residual_gate` / `hybrid_gate`（默认 False）控制；`config_loader.py` 读取。
- 新增单测覆盖带全增强的 hybrid 模型前向 / 增量 / 生成。注意：开启后需重新训练才能学到有效门控值，旧权重仍可加载（门控默认 1.0 ≈ 普通残差）。

## `92f669e`（已推送，基于 `ce82e2b`）

### 性能：IGMCG 批量化 + n-gram 缓存，并补测试
- `scripts/generate.py` 的 `_generate_candidates_batch` 改为真正单次 batched 前向（batch=N，候选独立 KV/SSM 状态）。
- `NGramModel.logprob_vector` 加按上下文 `(w2,w1)` 的缓存（抽离 `_compute_logprob`）。
- 新增 `tests/test_generation_pipeline.py`（11 个测试：data_utils 边界、IGMCG 生成、n-gram、设备探测、config_loader 兜底、学习率调度、梯度累积、量化加载）；`pytest` 现 27 passed（15 + 11 + 1）。
- 审查建议中两项判定无需改：Windows 上 `num_workers` 由 `create_dataloader` 故意置 0；CPU/DML 走手动注意力是有意优化。

## `ce82e2b`（已推送，基于 `001623e`）

### 修复：加固 checkpoint 加载安全 + 若干边界缺陷
- `scripts/generate.py` 的 `_safe_torch_load` 改为**固定白名单**（仅放行张量 / numpy 重建符号），关闭 CVE-2026-24747 的绕过路径（torch 因 `torch-directml` 约束不能升 2.10，缓解放在加载侧）。
- `models/data_utils.py`：coverage 空 `word_freq` 除零；`scripts/train_finetune.py`：空 dataloader `avg_loss` 除零。
- `scripts/train.py`：warmup 步数钳制不超过总有效步数；`models/device.py`：DML 探测各分支打印日志；`scripts/chat.py`：历史文件 open 失败降级。

## `001623e`（已推送，基于 `2402bce`）

### 修复：DML checkpoint 加载失败 + 冒烟训练对比脚本
- `models/utils.py` 新增 `_cpu_offload()`（张量搬 CPU、numpy 数组 / 标量转 torch / 原生类型），应用到 `save_checkpoint` / `save_final_model`；`scripts/train.py` 内联最终保存改用 `_cpu_offload(model.state_dict())`。
- `scripts/generate.py` 的 `load_model` 改 `_safe_torch_load()`（遇到可信官方全局符号自动 `add_safe_globals` 后重试，跨 torch/numpy 版本稳定）。
- 新增 `configs/config_smoke_4k.yaml` 与 `experiments/_smoke_gen_compare.py`（baseline top-k vs IGMCG 多候选打分）；单轮 Val Loss ≈ 8.44；`tests` 15 passed。

## `2402bce`（已推送，基于 `1ac8036`）

### 重构：消除重复 import / 重复配置，系统化错误处理与清理遗留项
- 合并散落根目录的临时脚本到 `experiments/`（`_*.py`）/ `tools/` / `scripts/`；统一错误处理装饰器 `cli_guard`。
- 测试目录约定整理：`tests/` 为正式 pytest（纳入 git），`test/` 为本地沙箱（gitignore）。

## `1ac8036`（已推送）

### 修复：SSM 增量解码与多项生成 / 训练缺陷（F1–F6）
- SSM KV-Cache 增量解码状态错位修复（`past_state` 维度 / 选择性扫描衔接）。
- IGMCG KV-Cache 污染修复：每个候选独立 `past`。
- RoPE 缓存线程安全：实例级 `_cache` + `RLock`，可选类级共享缓存。
- `loss_sum` 内存泄漏修复：Python float 累加。
- `tie_weights` DML 重绑定：重写 `to()` 自动调用 `tie_weights()`。

## `5162f86`（已推送，基于 `ef10f6a`）

### 修复：生成参数与安全增强
- **min_length 计算修正**：移除 `len(token_ids) + 2` 导致输入越长强制生成越长的反直觉行为，改为固定默认 `min_length=3` 并可配置。
- **top_k 边界修正**：`top_k <= 0` 禁用，`top_k >= vocab_size` 视为全词表（原实现 `top_k == vocab_size` 时错误跳过过滤）。
- **EOS 惩罚参数化**：硬编码 `-5.0` → 可配置参数 `eos_penalty`（默认 `-5.0`）。
- **generate() 新增参数**：`min_length`、`eos_penalty`，默认值从配置读取。

### 修复：类型注解与工程质量
- 核心 4 模块（`transformer.py` / `config_loader.py` / `data_utils.py` / `device.py`）添加完整类型注解（`from __future__ import annotations` + `typing`）。
- 合并 5 个重复数据合并脚本 → 统一 `merge_data.py`（argparse + 去重 + 词表构建 + 配置自动更新）。
- 新增 15 单测覆盖核心模块（Transformer / Config / RoPE / SSM step / tie_weights / 注意力掩码设备迁移）。
- 新增 CI/CD 流水线：test / lint / security 三阶段，含 `weights_only=False` 扫描。

### 修复：语法错误（IndentationError）
- `scripts/diagnose.py`、`scripts/tuning/showcase_optimal_params.py`、`tune_temperature.py`、`tune_topk.py` 修正模块级缩进错误（此前编辑误加 4 空格导致无法运行）。

## `ef10f6a`（已推送）

### 修复：安全漏洞 + 核心架构
- **全项目 `torch.load(weights_only=True)`**：防 pickle RCE，checkpoint 分离存储（`.pt` 仅张量 + `*_config.yaml` 配置），涉及 20+ 脚本。
- **SSM KV-Cache 增量推理**：`MambaSSM` 新增 `past_state` / `_forward_step` / `_selective_scan(past_state)`，混合架构现可用 KV-cache O(L) 解码（原回退全量 O(L²)）。
- **IGMCG 多候选独立 KV-Cache**：`_generate_candidates_batch` 每候选维护独立 `past`，消除跨候选状态污染。
- **RoPE 缓存线程安全**：模块级全局 `_ROPE_CACHE` → `RotaryEmbedding` 实例级 `_cache` + `RLock`，可选类级共享缓存（`enable_shared_cache()`）。
- **loss_sum 内存泄漏修复**：`torch.zeros(())` 张量累加 → Python float `loss_sum += loss.detach().item()`。
- **tie_weights DML 重绑定**：重写 `TransformerModel.to()` 自动调用 `tie_weights()`，保证 `.to(device)` 后权重共享生效。

## `a36f037`（已推送）

### 修复：DML 推理崩溃
- 根因：`torch.load(..., map_location=device)` 在 DML 设备对象上触发 `TypeError`；生成路径用 `torch.inference_mode()` 在 DML 报 `Cannot set version_counter for inference tensor`。
- 修复：所有推理脚本统一 `map_location='cpu'` 再 `.to(device)`；生成路径 `inference_mode` → `no_grad`（`models/transformer.py` + `scripts/generate.py` + `experiments/*`）。
- 验证：AMD 780M (DML) 上跑通 generate（IGMCG 与基础双轨）、diagnose、tune_topk。

### 修复：诊断/调参脚本可用性
- `scripts/diagnose.py` 与 `scripts/tuning/*.py` 新增 `--model` / `--vocab` / `--device` 参数。
- 修复 `scripts/tuning/*.py` 缺失的 `sys.path` 注入。
- 修正此前编辑引入的模块级缩进错误。

### 实验脚本
- `experiments/_diag_igmcg.py`、`experiments/_gen_opt_test.py` 同步 `inference_mode` → `no_grad`。

## `43c7d27`（已推送，基于 `7590280`）
...（后续保持原有内容不变）

## （本地，2026-07-18，可学习字符词表落地）
- feat: CharTokenizer 零 OOV 字符词表落地，解决词级词表 UNK 泛滥（原 vocab5000 仅覆盖 21.85%，生成 68% 是 <???>）。词表存盘 checkpoints_full_dml/vocab_char.json（size=5000，永久复用），配套 inal_model_char.pt + inal_model_config.yaml。编码全语料 UNK 占比 = 0.000%。
- fix: CharTokenizer._is_valid_char 过滤 CJK 扩展 A(U+3400-4DBF)/增补 B+(U+20000+) 生僻汉字；原漏过滤导致 C4 类语料建词表混入扩展区汉字，生成全成生僻乱码。
- chore: 旧词级词表(造成 UNK 根因) checkpoints_full_dml/vocab.json + 模型移至 unused_vocab/vocab_wordlevel_old.json（仅留 json 档案）；删 rchive_unused/checkpoints_backup/（~3GB 旧模型）+ 临时脚本/log；pytest 104 passed / 1 skipped。
- 结论：字符级零 OOV 已达成；纯字符级在 4000 行小数据下生成偏生僻字（数据量瓶颈，非词表 bug），需更大清洗语料才有干净常用字生成。
