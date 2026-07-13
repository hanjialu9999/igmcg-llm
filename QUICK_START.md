# 快速开始 (Quick Start)

本指南帮助你在 5 分钟内跑通「训练 → 推理」完整流程。

## 1. 安装依赖

推荐使用项目自带环境 `.amd_venv`（已装 torch 2.4.1 + torch-directml，开箱即用）；或自建虚拟环境：

```bash
python -m venv .amd_venv
.amd_venv\Scripts\activate
pip install -r requirements.txt
# AMD / Intel 核显或独显（Windows）需额外装 torch-directml；
# 注意：PyPI 上的 torch-directml 需 torch<2.2，本项目用兼容构建（如 0.2.5.dev240914，配合 torch 2.4.1）
pip install torch-directml
```

## 2. 准备数据

主训练语料默认是 `data/pretrain_corpus/merged.txt`（本地语料，**不纳入仓库**；小样本调试可用 `merged_sample.txt` 或 `merged_sample_8k.txt`）。
原始 QA 数据在 `data/datasets/`（同样仅本地保留，未上传 git）。如需从 `data/datasets/` 重新构建语料：

```bash
python scripts/merge_data.py           # 合并 datasets/ 下语料并构建词表（统一入口）
python scripts/process_data.py         # 转换为 jsonl（可选）
```

## 3. 训练语言模型（基座模型）

```bash
python scripts/train.py --config configs/pretrain.yaml
```

训练结束后会在 `checkpoints/` 下生成：
- `final_model.pt`：最终模型权重
- `vocab.json`：词表（推理 / 诊断脚本依赖它）

训练过程支持 warmup、混合精度（CUDA / CPU bf16，DML 自动回退 fp32）、早停与自动清理旧检查点。

## 4. 微调（可选）

若已有预训练 / 微调模型，可在此基础上继续微调（数据来自 `data/datasets/`）：

```bash
python scripts/train_finetune.py
```

产出 `best_finetuned_model.pt`，供 `scripts/chat.py` / `tools/dialogue_interactive.py` 使用。

## 5. 对话 / 生成

```bash
# 交互式对话（续写式）
python scripts/chat.py

# 带历史管理的交互式对话
python tools/dialogue_interactive.py
```

CPU 推理提速/降功耗：`--dtype bf16`（默认 auto 会启用，支持的 CPU/CUDA 约 1.5~1.8× 提速且质量基本无损）；`--cpu-threads N` 限制线程数；纯 CPU 还可加 `--quantize` 启用 int8 动态量化，进一步降低内存带宽与功耗（约 4× 更小模型，质量无损）。`generate.py` 同样支持这些参数。

生成行为由命令行参数控制（见 `python scripts/chat.py --help`）：`--temperature` / `--top-k` / `--repetition-penalty` / `--max-length` 等。

## 6. 调参（可选）

```bash
python scripts/tuning/tune_temperature.py   # 扫描不同温度
python scripts/tuning/tune_topk.py          # 扫描不同 top-k
python scripts/tuning/showcase_optimal_params.py  # 展示最优参数并回写 chat_config.json
```

## 常见问题

- **`FileNotFoundError: vocab.json`**：先执行第 3 步训练（或单独构建词表）。
- **想换模型结构**：只改 `configs/pretrain.yaml` 的 `model` 段，所有脚本会自动同步。
- **显存不足**：在 `configs/pretrain.yaml` 调小 `training.batch_size`，或在 `train_finetune.py`
  中减小 `DataLoader` 的 `batch_size`。
