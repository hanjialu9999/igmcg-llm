# 模型使用指南 (Model Usage Guide)

## 模型结构

定义在 `models/transformer.py` 的 `TransformerModel`：

- **主干**：Pre-LN + RMSNorm + RoPE + SwiGLU（现代 decoder-only LM 配方）
- **可配置混合架构**：通过 `layer_plan` 指定每层类型 —— `attn`（滑动窗口注意力，可选相对位置偏置）/`ssm`（MambaSSM 选择性状态空间）/`hybrid`（并行 attn+ssm）
- 默认配置（`pretrain.yaml`）：6 层纯 `attn`，`d_model=512`、`nhead=8`、`ffn=1024`、`max_seq_length=64`、`vocab_size=12000`
- **权重共享**：输出头复用词嵌入权重（`tie_weights=True`），减少参数并常提升 LM 质量
- **梯度检查点**：训练时可选开启（`gradient_checkpointing=True`），以计算换显存

生成（自回归）由 `TransformerModel.generate()` 实现，支持：
- temperature / top-k / repetition_penalty / EOS 惩罚 / min_length
- n-gram 统计先验双轨叠加（`ngram_fn` + `ngram_weight`）
- IGMCG 直觉引导多候选生成（`intuition` 向量 + 综合评分）
- KV-cache 增量解码（纯 attn / 混合架构均支持，SSM 含增量状态缓存）

## 加载模型（统一方式）

所有脚本通过 `models/config_loader.py` 构建与加载模型，避免到处硬编码结构：

```python
from models.config_loader import load_config, build_model, load_vocab

config = load_config()                   # 读取 configs/pretrain.yaml
model = build_model(config, device=device)  # 结构与 config 完全一致
vocab = load_vocab('checkpoints/vocab.json')
ckpt = torch.load('checkpoints/final_model.pt', map_location='cpu', weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
```

> ⚠️ **安全**：所有 `torch.load` 均使用 `weights_only=True`，防止 pickle RCE；配置单独存为 `*_config.yaml`。

## 查看模型

```bash
python tools/view_model.py
```

## 对话

```bash
python scripts/chat.py
```

`chat.py` 默认从 `checkpoints/final_model.pt` 加载模型与 `checkpoints/vocab.json`，生成参数由命令行参数（`--temperature` / `--top-k` / `--repetition-penalty` / `--max-length` 等）控制。`tools/dialogue_interactive.py` 则会读取仓库根的 `chat_config.json`（由 `scripts/tuning/showcase_optimal_params.py` 回写）作为持久化对话参数。

## 诊断 / 对比

```bash
python tools/diagnose_vocab.py     # 词表编码检查
python tools/compare_epochs.py     # 对比不同 epoch 输出
```

## 注意

- 修改模型结构（层数 / 维度 / 词表大小 / 混合架构）**只改 `configs/pretrain.yaml`**（或 `config_hybrid.yaml` 等），所有训练 / 推理脚本会自动同步，无需改动代码。
- 推理前必须存在与模型词表大小一致的 `checkpoints/vocab.json`。
- 混合架构（含 SSM）的增量推理在 `max_seq_length` 内为 O(L)；纯注意力模型同样用 KV-cache。