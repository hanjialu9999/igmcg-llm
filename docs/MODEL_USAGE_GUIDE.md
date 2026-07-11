# 模型使用指南 (Model Usage Guide)

## 模型结构

定义在 `models/transformer.py` 的 `TransformerModel`：

- 词嵌入 + 正弦位置编码
- 6 层 `nn.TransformerEncoder`（8 头，d_model=512，ffn=1024）
- 线性输出头映射到词表
- 训练时启用梯度检查点以省显存

生成（自回归）由 `TransformerModel.generate()` 实现，支持 temperature /
top-k / 重复惩罚 / 特殊 token 屏蔽。

## 加载模型（统一方式）

所有脚本通过 `models/config_loader.py` 构建与加载模型，避免到处硬编码结构：

```python
from models.config_loader import load_config, build_model, load_vocab

config = load_config()                   # 读取 configs/pretrain.yaml
model = build_model(config).to(device)  # 结构与 config 完全一致
vocab = load_vocab('checkpoints/vocab.json')
ckpt = torch.load('checkpoints/final_model.pt', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
```

## 查看模型

```bash
python tools/view_model.py
```

## 对话

```bash
python scripts/chat.py
```

`chat.py` 会从 `best_finetuned_model.pt` 加载微调模型与 `checkpoints/vocab.json`，
生成参数来自 `chat_config.json`。

## 诊断 / 对比

```bash
python tools/diagnose_vocab.py     # 词表编码检查
python tools/compare_epochs.py     # 对比不同 epoch 输出
```

## 注意

- 修改模型结构（层数 / 维度 / 词表大小）**只改 `configs/pretrain.yaml`**，
  所有训练 / 推理脚本会自动同步，无需改动代码。
- 推理前必须存在与模型词表大小一致的 `checkpoints/vocab.json`。
