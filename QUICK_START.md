# 快速开始 (Quick Start)

本指南帮助你在 5 分钟内跑通「训练 → 推理」完整流程。

## 1. 安装依赖

```bash
python -m venv .my_venv
.my_venv\Scripts\activate
pip install -r requirements.txt
```

## 2. 准备数据

主训练语料默认是 `data/train_data_final.txt`（已包含在仓库中）。
如需从 `data/datasets/` 下的原始 QA 数据重新构建：

```bash
python scripts/prepare_training.py     # 合并 datasets/ 下所有 txt
python scripts/process_data.py         # 转换为 jsonl（可选）
```

## 3. 训练语言模型（基座模型）

```bash
python scripts/train.py --config config/pretrain.yaml
```

训练结束后会在 `checkpoints/` 下生成：
- `final_model.pt`：最终模型权重
- `vocab.json`：词表（推理 / 诊断脚本依赖它）

训练过程支持 warmup、混合精度（CUDA）、早停与自动清理旧检查点。

## 4. 微调（可选）

若已有预训练 / 微调模型，可在此基础上继续微调（数据来自 `data/datasets/`）：

```bash
python train_finetune.py
```

产出 `best_finetuned_model.pt`，供 `scripts/chat.py` / `dialogue_interactive.py` 使用。

## 5. 对话 / 生成

```bash
# 交互式对话（续写式）
python scripts/chat.py

# 带历史管理的交互式对话
python dialogue_interactive.py
```

生成行为由 `chat_config.json` 控制：`temperature`、`top_k`、`repetition_penalty`、
`min_new_tokens`、`max_new_tokens`、`context_rounds`。

## 6. 调参（可选）

```bash
python scripts/tuning/tune_temperature.py   # 扫描不同温度
python scripts/tuning/tune_topk.py          # 扫描不同 top-k
python scripts/tuning/showcase_optimal_params.py  # 展示最优参数并回写 chat_config.json
```

## 常见问题

- **`FileNotFoundError: vocab.json`**：先执行第 3 步训练（或单独构建词表）。
- **想换模型结构**：只改 `config/config.yaml` 的 `model` 段，所有脚本会自动同步。
- **显存不足**：在 `config.yaml` 调小 `training.batch_size`，或在 `train_finetune.py`
  中减小 `DataLoader` 的 `batch_size`。
