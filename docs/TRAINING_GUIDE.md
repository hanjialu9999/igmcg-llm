# 训练指南 (Training Guide)

## 训练流程总览

1. 准备语料 → `data/pretrain_corpus/merged.txt`（本地语料，**不纳入仓库**）
2. 运行训练 → `python scripts/train.py --config configs/pretrain.yaml`
3. 产物 → `checkpoints/final_model.pt` + `checkpoints/vocab.json`

## 关键配置（configs/pretrain.yaml）

```yaml
model:
  vocab_size: 12000          # 覆盖 ~98.9% token；DML 在更大词表下 backward 会触发设备错误
  embedding_dim: 512
  num_heads: 8
  num_layers: 6
  hidden_dim: 1024
  max_seq_length: 64
  dropout: 0               # 单轮预训练不过拟合，关掉让模型充分拟合语料
  tie_weights: true        # 输出头复用 embedding 权重
  gradient_checkpointing: true

training:
  batch_size: 128
  epochs: 1                # 基座模型单次完整遍历语料
  learning_rate: 1.0e-3    # 单轮只过一遍，需更大 lr
  weight_decay: 0.01
  gradient_clip: 1.0
  warmup_steps: 0.1        # 占整个 epoch 有效步数的比例（前 10% 线性升温）
  early_stop_patience: 5
  label_smoothing: 0.1
  # 默认训练采用 SEL v2 分段选择性增强调度（2026-07-14 起）
  # 8 段：1 全开 + 1 全关极端 + 6 局部（attn_temp 仅全关段关，平时恒开）
  enhancement_schedule:
    - { qk_norm: true,  attn_temp: true,  residual_gate: true,  hybrid_gate: false }
    - { qk_norm: false, attn_temp: false, residual_gate: false, hybrid_gate: false }
    - { qk_norm: true,  attn_temp: true,  residual_gate: false, hybrid_gate: false }
    - { qk_norm: false, attn_temp: true,  residual_gate: true,  hybrid_gate: false }
    - { qk_norm: true,  attn_temp: true,  residual_gate: false, hybrid_gate: false }
    - { qk_norm: false, attn_temp: true,  residual_gate: true,  hybrid_gate: false }
    - { qk_norm: true,  attn_temp: true,  residual_gate: false, hybrid_gate: false }
    - { qk_norm: false, attn_temp: true,  residual_gate: true,  hybrid_gate: false }

data:
  train_file: "data/pretrain_corpus/merged.txt"
  vocab_size: 12000
  max_seq_length: 64
  num_workers: 0           # Windows 必须为 0
  test_split: 0.1
```

## 训练特性

- **混合精度 (AMP)**：`precision: bf16` 在 CUDA 与 CPU 下自动开启（约 2~2.5× 提速，loss 基本无损，无需 loss scaling）；`fp16` 仅 CUDA（用 GradScaler）。AMD DirectML 暂不支持 AMP，自动回退 fp32。
- **Warmup**：训练首 epoch 内线性升温学习率。
- **Label Smoothing**：当前**未启用**——`CrossEntropyLoss` 在同时使用 `label_smoothing` 与 `ignore_index`（padding 屏蔽）时不支持，训练脚本会忽略该配置并打印警告（见 `scripts/train.py`）。
- **早停**：连续 `early_stop_patience` 个 epoch 无提升则停止。
- **检查点清理**：仅保留最近若干 epoch 与最佳模型。
- **SELv2 默认增强调度**：`pretrain.yaml` 默认带 `training.enhancement_schedule`（SELv2 8 段：1 全开 + 1 全关极端 + 6 局部，attn_temp 仅全关段关），即默认训练按分段选择性增强训练；`validate` 强制全开。受控对比可用 `config_cmp_{enh,sel,selv2}_full.yaml` + 根目录 `run_full_cmp.ps1` 一键三模型顺序训练。

## 监控训练

```bash
python tools/monitor/monitor_training.py
python tools/monitor/monitor_gpu_training.py   # GPU 显存 / 利用率
python tools/monitor/monitor_live.py
```

## 检查点与恢复

- 训练脚本每次运行都从配置**重新构建模型并从头训练**（当前未实现自动断点续训）。
- 当 `epochs > 1` 时，每个 epoch 结束会保存 `model_epoch_*.pt` 与最佳模型备份，主要用作容错备份（非自动续训）。
- 单 epoch 配置（`epochs=1`）不保存逐 epoch 检查点，仅输出 `final_model.pt` + `vocab.json`。
- 若需从已有权重继续，可手动加载 `final_model.pt` 的 `model_state_dict` 后再训练（或作为微调底座）。

## 微调

若要在预训练模型基础上针对对话数据微调：

```bash
python scripts/train_finetune.py
```

微调数据来自 `data/datasets/`（每两个非空行构成一对 Q/A；该目录仅本地保留，未上传 git）。
产出 `best_finetuned_model.pt`，供 `scripts/chat.py` / `tools/dialogue_interactive.py` 使用。
