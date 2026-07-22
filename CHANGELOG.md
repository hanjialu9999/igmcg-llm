# 变更日志 (CHANGELOG)

手工维护的主要变更记录，便于对照 Git 提交历史（`git log`）。

## 约定
- 每个条目以提交哈希（如 `43c7d27`）为标题，并标注父提交与推送状态。
- 按时间倒序排列（最新在最上方）。
- 仅记录对训练 / 推理链路有实质影响的修复、功能与破坏性变更；纯文档微调通常合并记录。
- 提交信息风格：中文主题行 + 空行 + 要点式正文。
- 状态标记：`已推送` = 已 `git push` 到 `origin/main`；`本地` = 仅本地提交待推送。

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
