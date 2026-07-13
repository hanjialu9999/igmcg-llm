# 变更日志 (CHANGELOG)

手工维护的主要变更记录，便于对照 Git 提交历史（`git log`）。

## 约定
- 每个条目以提交哈希（如 `43c7d27`）为标题，并标注父提交与推送状态。
- 按时间倒序排列（最新在最上方）。
- 仅记录对训练 / 推理链路有实质影响的修复、功能与破坏性变更；纯文档微调通常合并记录。
- 提交信息风格：中文主题行 + 空行 + 要点式正文。
- 状态标记：`已推送` = 已 `git push` 到 `origin/main`；`本地` = 仅本地提交待推送。

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
