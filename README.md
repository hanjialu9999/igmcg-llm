# 中文语言模型 (Attention × SSM × IGMCG + n-gram 解码)

基于 Transformer 的中文 LM 训练 / 推理项目，融合自定义架构（注意力 × SSM × IGMCG 直觉引导解码）与统计式 n-gram 双轨解码。目标是在 CPU / AMD iGPU（DirectML）等低资源设备上，也能训练并跑出连贯的中文生成。

## 特性

- **混合主干**：`TransformerModel` 支持纯注意力，或与 `MambaSSM` 组成的 SSM×注意力混合架构；均带 KV-cache，生成高效。
- **现代结构**：Pre-LN + RMSNorm + RoPE + SwiGLU（见 `models/transformer.py`）。
- **向量化 SSM 扫描**：选择性扫描用并行前缀扫描实现（log2(L) 步），既加速 CPU 训练，也避免低功耗 iGPU 上 DML 因单步 kernel 过多触发 TDR 设备重置。
- **双轨解码**：神经模型输出与统计 n-gram 先验在解码期叠加，互补长短。
- **IGMCG 反碎片化**：生成多个温度候选，按综合分选优，抑制“碎片式”输出。
- **跨设备**：自动选择 `cuda` / `dml`(AMD) / `cpu`；DML 推理现已可用，训练侧 bf16 在 CPU/CUDA 上开启混合精度。
- **低功耗选项**：CPU 可用 `--cpu-threads` 限线程、`--quantize` 启 int8 动态量化。

## 目录结构

```
models/          模型与基础设施
                  transformer.py  (TransformerModel / MambaSSM, 支持 KV-cache)
                  data_utils.py   (Vocabulary / 数据加载)
                  config_loader.py (load_config / build_model)
                  device.py       (get_device / apply_cpu_threads)
scripts/         入口与数据处理
                   train.py        训练主程序 (--config)
                   train_finetune.py  微调训练（QA 两阶段：预训练底座 → 微调）
                   generate.py     生成 API: generate_text / generate_igmcg / NGramModel
                   chat.py         对话式 CLI (--ngram / --igmcg / --intuition)
                   chat_zh.bat     中文 Windows 一键对话启动器
                    merge_data.py, process_data.py, convert_dialogue_to_qa.py, convert_statements_to_qa.py, data_manager.py ...
                   data/download_pretrain_data.py, tuning/  (参数扫描)
configs/         所有 YAML 配置 (pretrain.yaml 为规范默认；dml_* / hybrid_* / test_* 为变体)
experiments/     实验 / 诊断 / 一次性脚本 (原根目录 _*.py，可独立运行，自带路径修正)
tools/           检查与监控工具 (view_model / compare_epochs / dialogue / dialogue_interactive / monitor/ ...)
tests/           正式 pytest 单元测试（已纳入 git 跟踪）：test_config_loader.py / test_transformer.py / test_generation_pipeline.py（当前 27 passed）
test/            本地自测沙箱（gitignore，仅本机运行，不入库）
data/            语料 (pretrain_corpus/) 与数据集 (datasets/)
logs/            运行日志
checkpoints/     训练产出 (checkpoints_dml_b32 等子目录维持原样，未迁移)
 archive_unused/  历史归档 (未动)
```

> 各子目录内另有 `README.md` 详述其文件用途（`models/` `scripts/` `configs/` `experiments/` `tools/` `docs/` `data/` 等）。

## 环境依赖

- Python 3.10+，依赖 PyTorch 2.4+。
- 在 AMD GPU 上用 DirectML 推理 / 训练，需额外安装 `torch-directml`（本项目 venv 为 `.amd_venv`）。
- 推荐在虚拟环境中运行；Windows 下终端为 GBK，中文日志请查看 UTF-8 文件（如 `logs/generation_output.txt`）或用 `cmd /c "python ... > out.txt"` 重定向以获得正确编码。

## 快速开始

```bash
# 训练（默认规范配置）
python scripts/train.py --config configs/pretrain.yaml

# 生成（神经 + n-gram 双轨 + IGMCG 多候选联合解码）
python scripts/generate.py --prompt "今天天气怎么样" --ngram --igmcg --ngram-weight 0.3

# 对话
python scripts/chat.py --ngram --igmcg --intuition 0.3,0.8,0.5,0.2,0.6,0.4,0.5
```

更多示例：

```bash
# 指定权重 / 词表 / 设备（例如 AMD DML 推理）
python scripts/generate.py \
    --model archive_unused/checkpoints_backup/_stab_ckpt/final_model.pt \
    --vocab checkpoints_dml_test/vocab.json \
    --device dml --dtype fp32 \
    --prompt "中国的首都是" --ngram --igmcg --max-length 60

# 诊断模型输出（前向 / Top-k 分布）
python scripts/diagnose.py --model checkpoints/final_model.pt --vocab checkpoints/vocab.json --device auto

# 参数扫描（Top-K / Temperature）
python scripts/tuning/tune_topk.py --model checkpoints/final_model.pt --vocab checkpoints/vocab.json --device auto
```

## 模型架构

- **主干**：`TransformerModel` 为 Pre-LN + RMSNorm + RoPE + SwiGLU 的 Transformer；可通过配置切换为含 `MambaSSM` 的 SSM×注意力混合架构。
- **MambaSSM 选择性扫描**：已用**并行前缀扫描**向量化（log2(L) 步、数值稳定），消除逐时间步 `for` 循环：大幅加快 CPU 训练，并避免低功耗 iGPU 上 DML 因单步 kernel 过多触发 TDR 设备重置。
- **KV-cache**：`generate()` 支持 `use_cache`，自回归逐 token 解码只算新增一步，速度随序列长度近线性。
- **双轨解码**：神经对数概率与 n-gram 模型统计先验在解码期按权重叠加，n-gram 只遍历与上下文相关的少量 token，开销极低。
- **IGMCG**：见下节。

## 训练

- 配置集中在 `configs/`（详见 `configs/README.md`）。`pretrain.yaml` 为规范默认：词表 12000、6 层、emb512、单轮遍历。
- 数据：`data/pretrain_corpus/merged.txt` 为默认训练语料（本地，不入库）；小样本调试可用 `merged_sample.txt`。词表在训练时自动构建，存于 `checkpoints/vocab.json`。
- **混合精度**：`precision: bf16` 在 **CPU / CUDA** 开启（约 2~2.5× 提速、loss 基本无损）；`fp16` 仅 CUDA（启用 GradScaler）；AMD DirectML 暂不支持 AMP，自动回退 fp32。
- **DML 训练**：SSM/hybrid 的选择性扫描已向量化，可在 DML 上正常训练（旧版逐时间步 for 循环会因 kernel 风暴触发 iGPU 设备重置）。

## 推理与生成

`scripts/generate.py` 主要参数：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--prompt` / `--prompt-file` | 输入文本或文本文件 | — |
| `--max-length` | 生成最大长度 | 30 |
| `--temperature` | 采样温度（0=贪心） | 0.8 |
| `--top-k` | Top-K 截断 | 50 |
| `--repetition-penalty` | 重复惩罚 | 1.7 |
| `--device` | `cpu` / `cuda` / `dml` / `auto` | `auto` |
| `--dtype` | `fp32` / `bf16` / `auto` | `auto` |
| `--quantize` | 启用 int8 动态量化（纯 CPU） | 关 |
| `--cpu-threads` | 限制 CPU 线程数（降功耗） | 4 |
| `--ngram` / `--ngram-weight` | 开启 n-gram 双轨及权重 | 关 / 0.3 |
| `--igmcg` / `--igmcg-candidates` | 开启 IGMCG 及候选数 | 关 / 5 |
| `--intuition` | 7 维直觉向量（逗号分隔） | 全 0.5 |
| `--model` / `--vocab` | 显式指定权重 / 词表 | `checkpoints/` |

> **DML 推理现已可用**：权重统一先加载到 CPU 再 `.to(device)` 搬运，生成路径改用 `torch.no_grad()`（DML 后端不支持 `inference_mode`，会报 `Cannot set version_counter for inference tensor`）。

## IGMCG 反碎片化设计

IGMCG 生成多个温度候选，按综合分选优：

```
score = 1.5 * 连贯度(coh) + 0.15 * 流畅度 + 0.15 * 风格匹配 - 2.5 * 重复度
```

- **连贯度(coh)**：用 n-gram 模型计算序列相邻 token 的预测概率，越高=越相连，是抑制“碎片化”的核心信号。
- 流畅度（单 token 置信度）只作轻微 tiebreaker——孤立高频词也会拉高它，故不主导。
- 风格匹配为 7 维直觉的温和偏置（在连贯候选间微调，绝不压过连贯度）。
- 候选温度范围收窄 (0.75~1.35×)，生成期重复惩罚 1.4，避免候选本身过度发散或循环。

## 可配置架构增强（实验性，2026-07-14 起默认开）

以下增强均经 `config['model']` 开关控制（**默认 `True`**），新训练默认即开、无需手动开启；**加载旧权重（无该键）时默认关，保持向后兼容**。开启后需重新训练才能学到有效参数（门控默认 init 1.0 ≈ 普通残差）：

- **QK-Norm**：注意力 Q/K 在做 RoPE 前先过一层 `RMSNorm(head_dim)`，稳定注意力尺度。
- **可学习注意力温度**：每层一个可学习标量 `log_temp`，按 `temp = exp(log_temp)` 缩放 Q/K（替代固定 `/sqrt(head_dim)`）。
- **门控残差（residual_gate）**：每个 block 的注意力 / SSM 分支与 FFN 分支各带一个可学习门控 `nn.Parameter`，相当于可学习的残差缩放（思路 ②/⑥）。
- **混合路径门控（hybrid_gate，⭐A）**：hybrid 块内 attn 与 ssm 两路各自带可学习门控，让模型自决每层偏重。

对应配置键：`qk_norm` / `attn_temp` / `residual_gate` / `hybrid_gate`（`models/config_loader.py` 读取；`scripts/generate.py` 的 `load_model` 从各 checkpoint 的 `*_config.yaml` 透传，旧权重缺省关）。

### 性能与实测（2026-07-14）
- 受控 A/B（8000 行×1 epoch，DML fp32）：增强开 Val 7.22 vs 关 7.72；但 DML 训练吞吐降至约 2370 tok/s（关约 3270，慢约 28%）。解码开销可忽略（top-k/IGMCG 约 0.95–0.98x）。详见 `experiments/cmp_enh_base.txt` 与提交 `f9452ed`。
- **为何变慢（根因）**：`gradient_checkpointing=True` 会在反向时重算整个 block 的前向，导致廉价的 QK-Norm/门控等算子被**执行两次**。微基准（DML，小模型 forward+backward）：`qk_norm` 单独约 +9%、`residual_gate` 约 +3%、`attn_temp` 约 +1%，叠加约 +28%；而关掉梯度检查点后基线本身快约 22%（115.7→90.6 ms/step）。因此该开销主要是检查点重算放大，并非算子本身昂贵。
- **优化点**：① 小模型训练把 `gradient_checkpointing` 设为 `false` 可整体提速（基线即 −22%，增强开销只付一次）；② 更通用做法——把廉价增强算子（QK-Norm/RoPE/门控）放到检查点重算区域之外、仅对注意力/FFN 重算力部分做检查点，既保大模型显存又去掉翻倍开销；③ `attn_temp` 已融合为单次标量乘法（免去 `sqrt`）。相应微基准见 `experiments/_bench_enh.py`。

### 实测对比（2026-07-14，同数据同参数受控 A/B）

在 8000 行语料（merged.txt 前 8000 行，确定性，非随机）× 1 epoch、`configs/config_cmp_base.yaml`（四关）vs `configs/config_cmp_enh.yaml`（四开）、DML(fp32) 下对照：

- **训练**：增强开 **Val Loss 7.22** vs 关 **7.72**；但 DML 训练吞吐降至约 **2370 tok/s**（关约 3270，慢约 28%）。开销来自 QK-Norm/RMSNorm、可学习温度缩放与门控残差在 DML 的反向/批前向上。
- **生成**：解码开销可忽略——top-k 约 34 vs 33 tok/s，IGMCG 约 10 vs 10 tok/s（约 0.95–0.98x）。
- **文本**：8000 行/1 epoch 仍偏词块破碎；增强版输出略更连贯（如「中国的首都是我们，但目前…」），与更低的 val loss 一致。
- **注意**：开启增强后须重新训练；且 `scripts/generate.py` 的 `load_model` 需把上述四标志从 `*_config.yaml` 透传给 `TransformerModel`，否则增强权重无法加载（见提交 `f9452ed`）。

## 性能

训练与生成速度均在本机验证（AMD 780M iGPU / DirectML，`privateuseone:0`，fp32，配置 embed512 / 6 层 / batch32 / seq64）：

- **训练**：约 **3550 tok/s**（215 batch / 8000 行 / 1 epoch，逐批速度见 `scripts/train.py` 日志）。
- **生成（DML）**：基线 top-k 约 **40 tok/s**；IGMCG 开约 **12 tok/s**（约 0.31x，因每步多候选 batch 前向 + 打分）。
- 纯注意力模型 **CPU** 生成约 107 tok/s（KV-cache，4 线程）；IGMCG 多候选经批量化前向（单次 batch 共享 KV-cache），有效吞吐约 290 tok/s（含打分）。
- n-gram 先验叠加在解码期按需计算（仅遍历与上下文相关的少量 token）。
- > 注：DML 上 AdamW 的 `aten::lerp` 算子会回退 CPU 运行（torch-directml 限制，性能影响很小）；AMD DirectML 不支持 AMP，训练 / 推理在 DML 上均为 fp32。

## 已知限制 / 注意事项

- **词表**：`vocab.json` 中存在少量 `U+FFFD` 替换字符条目（语料读取 `errors='replace'` 所致），对生成质量影响极小，后续做语料清洗时可一并修复。
- **DML 精度**：AMD DirectML 不支持 AMP，训练 / 推理在 DML 上均为 fp32；bf16 仅 CPU/CUDA。
- **数据量**：当前示例模型多在 4000 / 8000 行 × 1 epoch 量级冒烟训练，生成质量偏弱（语法破碎、偶发 `<unk>`）；提升质量需更大语料与更多 epoch。
- **生成编码**：Windows GBK 终端可能误显中文，建议读取 UTF-8 日志或重定向输出。

## 文档索引

- `CHANGELOG.md`：主要修复与功能变更（对照提交历史）。
- `configs/README.md`：各训练 / 推理配置说明。
- `docs/`：`TRAINING_GUIDE.md`、`TUNING_GUIDE.md`、`MODEL_USAGE_GUIDE.md`、`DATA_USAGE_GUIDE.md` 等。
- `models/README.md`、`scripts/README.md`、`experiments/README.md`、`tools/README.md`、`data/README.md`：分模块说明。
