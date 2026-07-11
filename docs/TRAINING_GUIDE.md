# 训练指南 (Training Guide)

## 训练流程总览

1. 准备语料 → `data/train_data_final.txt`
2. 运行训练 → `python scripts/train.py --config configs/pretrain.yaml`
3. 产物 → `checkpoints/final_model.pt` + `checkpoints/vocab.json`

## 关键配置（configs/pretrain.yaml）

```yaml
model:
  vocab_size: 8731
  embedding_dim: 512
  num_heads: 8
  num_layers: 6
  hidden_dim: 1024
  max_seq_length: 64
  dropout: 0.1

training:
  batch_size: 128
  epochs: 200
  learning_rate: 0.0005
  weight_decay: 0.0001
  gradient_clip: 1.0
  warmup_steps: 30
  early_stop_patience: 10

data:
  train_file: "data/train_data_final.txt"
  vocab_size: 8731
  max_seq_length: 64
  num_workers: 0      # Windows 必须为 0
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

微调数据来自 `data/datasets/`（每两个非空行构成一对 Q/A）。
产出 `best_finetuned_model.pt`，供 `scripts/chat.py` / `dialogue_interactive.py` 使用。
