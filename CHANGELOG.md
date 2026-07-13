# 变更日志 (CHANGELOG)

手工维护的主要变更记录，便于对照 Git 提交历史（`git log`）。

## 约定
- 每个条目以提交哈希（如 `43c7d27`）为标题，并标注父提交与推送状态。
- 按时间倒序排列（最新在最上方）。
- 仅记录对训练 / 推理链路有实质影响的修复、功能与破坏性变更；纯文档微调通常合并记录。
- 提交信息风格：中文主题行 + 空行 + 要点式正文。
- 状态标记：`已推送` = 已 `git push` 到 `origin/main`；`本地` = 仅本地提交待推送。

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
