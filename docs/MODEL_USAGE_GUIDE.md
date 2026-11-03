# 模型使用指南 (Model Usage Guide)

## 模型结构

定义在 `models/transformer.py` 的 `TransformerModel` / `TransformerBlock`：

- **主干**：Pre-LN + RMSNorm + RoPE + SwiGLU（现代 decoder-only LM 配方）
- **可配置混合架构**：通过 `layer_plan` 指定每层类型 —— `attn`（滑动窗口注意力，可选相对位置偏置）/`ssm`（MambaSSM 选择性状态空间）/`hybrid`（并行 attn+ssm）
- 默认配置（`pretrain.yaml`）：6 层纯 `attn`，`d_model=512`、`nhead=8`、`ffn=1024`、`max_seq_length=64`、`vocab_size=8000`
- **权重共享**：输出头复用词嵌入权重（`tie_weights=True`），减少参数并常提升 LM 质量
- **梯度检查点**：训练时可选开启（`gradient_checkpointing=True`），以计算换显存
- **可插拔增强**：QK-Norm、可学习注意力温度、门控残差、char-merge、VRC、FiLM / input highway / progressive residual / layer skip / MoE / early exit / cross-layer routing 等，均由 `config['model']` 开关控制；加载无该键的旧权重时默认关
- **Controller/Generator 双模型编排**（`models/controller.py`）：Controller（GatedDeltaNet）产出压缩记忆 `mem_kv`、FiLM、direction 三类信号；三信号零初始化即中性。`controller_direction_causal` / `controller_causal_memory` / `controller_chunk_scan` 控制因果化与分块实现
- **n-gram 统计先验 + IGMCG**：解码期可选融合（`models/ngram.py` / `scripts/generate.py`），不改变模型、不需重训

生成（自回归）由 `TransformerModel.generate()` 实现，支持：
- temperature / top-k / top-p / repetition_penalty / EOS 惩罚 / min_length
- n-gram 统计先验叠加（`ngram_fn` + `ngram_weight`）
- IGMCG 直觉引导多候选生成（`intuition` 向量 + z-score 综合评分）
- KV-cache 增量解码（纯 attn / 混合架构均支持，SSM 含增量状态缓存）

## 加载模型（统一方式）

推荐用 `models.checkpoint.load_model`（一行完成构建 + 加载 + 设备搬运 + `*_config.yaml` 透传增强开关）：

```python
from models.checkpoint import load_model

model, vocab = load_model('checkpoints/final_model.pt', 'checkpoints/vocab.json',
                          device=device, quantize=False, compile_model=False)
```

- `quantize=True`：仅 CPU 的 int8 动态量化，DML 自动跳过。
- `compile_model=True`：`torch.compile`（CUDA/CPU），DML 自动跳过。

> ⚠️ **安全**：所有 `torch.load` 均使用 `weights_only=True`，防止 pickle RCE；配置优先从同目录 `<权重文件名>_config.yaml` 读取，旧 checkpoint 回退到内嵌 `config` 字段。

如需手动控制（如调试 / 自定义加载），等价低层写法：

```python
from models.config_loader import load_config, build_model, load_vocab
import torch

config = load_config()                   # 读取 configs/pretrain.yaml
model = build_model(config, device=device)  # 结构与 config 完全一致
vocab = load_vocab('checkpoints/vocab.json')
ckpt = torch.load('checkpoints/final_model.pt', map_location='cpu', weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
```

## 查看模型

```bash
python tools/view_model.py
```

## 对话

```bash
python scripts/chat.py
```

`chat.py` 默认加载 `checkpoints_train_8k_r42/final_model.pt` 与同名 `vocab.json`，设备默认 `cpu`；
生成参数由命令行控制：`--max-length`（默认 64）、`--temperature`（0.7）、`--top-k`（40）、
`--repetition-penalty`（2.0）、`--controller-direction on|off`（默认 off，只切运行时开关）。
`tools/dialogue_interactive.py` 则读取仓库根的 `chat_config.json`（由
`scripts/tuning/showcase_optimal_params.py` 回写）作为持久化对话参数。

## 运行时开关（不改 config / checkpoint）

`scripts/generate.py` 与 `scripts/chat.py` 在 `load_model` 之后、首次前向前可覆盖：

| 开关 | CLI | 默认 | 作用 |
|------|-----|------|------|
| CharMerge 增量滚动缓冲 | `--char-merge-buffer on/off` | `on` | `on` = 修好的读取路径；`off` = r42 旧零填充行为 |
| Controller direction 信号③ | `--controller-direction on/off` | `off` | 只设 `model.controller.use_direction`；Controller 未启用时无载体 |

## 诊断 / 对比

```bash
python tools/diagnose_vocab.py     # 词表编码检查
python tools/compare_epochs.py     # 对比不同 epoch 输出
python scripts/baseline_eval.py    # tf / prefix / incremental 三口径 ppl + 固定提示词
python scripts/eval_pipeline.py    # 一键评测流程（含生成对比与盲评）
```

## 注意

- 修改模型结构（层数 / 维度 / 词表大小 / 混合架构）**只改 `configs/pretrain.yaml`**（或 `config_hybrid.yaml` 等），所有训练 / 推理脚本会自动同步，无需改动代码。
- 推理前必须存在与模型词表一致的 `<checkpoint_dir>/vocab.json`（`load_vocab` 自动识别字符级 / BPE 格式）。
- 混合架构（含 SSM）的增量推理在 `max_seq_length` 内为 O(L)；纯注意力模型同样用 KV-cache。
