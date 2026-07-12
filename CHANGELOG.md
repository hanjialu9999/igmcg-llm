# 变更日志 (CHANGELOG)

记录主要修复与功能变更，便于对照提交历史。

## `43c7d27`（已提交，待 push，基于 `7590280`）

### 修复：DML 设备推理崩溃
- 根因：`torch.load(..., map_location=device)` 在 DML 设备对象（`privateuseone:0`）上会触发 `torch_directml.device(torch.device)` 的 `TypeError`；且生成路径使用 `torch.inference_mode()`，在 DML 后端前向时会报 `RuntimeError: Cannot set version_counter for inference tensor`。
- 修复：
  - 所有推理脚本（generate.py / diagnose.py / tuning/* / tools/* / dialogue_interactive.py）统一 `torch.load(..., map_location='cpu')`，加载后再 `.to(device)` 搬运到目标设备。
  - 生成路径的 `torch.inference_mode()` 改为 `torch.no_grad()`（models/transformer.py 的 `generate` 与 scripts/generate.py 的候选/打分函数），DML 推理不再崩溃。
- 验证：在 AMD 780M (DML) 上跑通 generate（IGMCG 与基础双轨）、diagnose、tune_topk。

### 修复：诊断/调参脚本可用性
- `scripts/diagnose.py` 与 `scripts/tuning/*.py` 新增 `--model` / `--vocab` / `--device` 参数（默认仍指向 `checkpoints/`）。
- 修复 `scripts/tuning/*.py` 缺失的 `sys.path` 注入（此前直接运行会 `ModuleNotFoundError: No module named 'models'`）。
- 修正此前编辑引入的模块级缩进错误（`    checkpoint = torch.load(...)` 多出的 4 空格导致 `IndentationError`）。

### 实验脚本
- `experiments/_diag_igmcg.py`、`experiments/_gen_opt_test.py` 同步 `inference_mode` → `no_grad`。

### 已知小问题
- 词表 `vocab.json` 中存在少量 `U+FFFD` 替换字符条目（语料读取 `errors='replace'` 所致），对生成质量影响极小，后续可做语料清洗时一并修复。

## `bd219f7`（已提交，待 push）

- 新增 **bf16 混合精度训练**：`precision: bf16` 在 CPU/CUDA 开启（约 2~2.5× 提速、loss 基本无损）；`fp16` 仅 CUDA（GradScaler）；DML 自动回退 fp32。
- **向量化 SSM 选择性扫描**（并行前缀扫描，log2(L) 步），修复 DML 上逐时间步 for 循环导致的 iGPU 设备重置（TDR 超时）。
- 修复设备选择 `get_device('dml')` 显式支持；`config_loader.build_model(..., device=device)` 在 `.to(device)` 后重新绑定权重共享（tie_weights 在 DML 上不被破坏）。
- 文档更新：configs/README、README、QUICK_START、docs/TRAINING_GUIDE、models/README。
