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

data:
  train_file: "data/pretrain_corpus/merged.txt"
  vocab_size: 12000
  max_seq_length: 64
  num_workers: 0           # Windows 必须为 0
  test_split: 0.1
```

## 训练特性

- **混合精度 (AMP)**：在 CUDA 下自动开启，省显存。
- **Warmup**：训练首 epoch 内线性升温学习率。
- **Label Smoothing**：`CrossEntropyLoss(label_smoothing=0.1)` 提升泛化。
- **早停**：连续 `early_stop_patience` 个 epoch 无提升则停止。
- **检查点清理**：仅保留最近若干 epoch 与最佳模型。

## 监控训练

```bash
python tools/monitor/monitor_training.py
python tools/monitor/monitor_gpu_training.py   # GPU 显存 / 利用率
python tools/monitor/monitor_live.py
```

## 断点续训 / 恢复

训练脚本每次 epoch 都会保存 `model_epoch_*.pt` 与 `final_model.pt`，
如需从某检查点恢复，可加载对应 `.pt` 后继续。

## 微调

若要在预训练模型基础上针对对话数据微调：

```bash
python train_finetune.py
```

微调数据来自 `data/datasets/`（每两个非空行构成一对 Q/A；该目录仅本地保留，未上传 git）。
产出 `best_finetuned_model.pt`，供 `scripts/chat.py` / `dialogue_interactive.py` 使用。
