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
                  generate.py     生成 API: generate_text / generate_igmcg / NGramModel
                  chat.py         对话式 CLI (--ngram / --igmcg / --intuition)
                  prepare_data.py, merge_datasets.py, convert_*_to_qa.py, data_manager.py ...
                  data/download_pretrain_data.py, tuning/  (参数扫描)
configs/         所有 YAML 配置 (pretrain.yaml 为规范默认；dml_* / hybrid_* / test_* 为变体)
experiments/     实验 / 诊断 / 一次性脚本 (原根目录 _*.py，可独立运行，自带路径修正)
tools/           检查与监控工具 (view_model / compare_epochs / dialogue / monitor/ ...)
test/            pytest 风格测试
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

## 性能

- 纯注意力模型 CPU 生成约 107 tok/s（KV-cache，4 线程）。
- IGMCG 多候选经批量化前向（单次 batch 共享 KV-cache），有效吞吐约 290 tok/s（含打分）。
- n-gram 先验叠加在解码期按需计算（仅遍历与上下文相关的少量 token）。

## 已知限制 / 注意事项

- **词表**：`vocab.json` 中存在少量 `U+FFFD` 替换字符条目（语料读取 `errors='replace'` 所致），对生成质量影响极小，后续做语料清洗时可一并修复。
- **DML 精度**：AMD DirectML 不支持 AMP，训练 / 推理在 DML 上均为 fp32；bf16 仅 CPU/CUDA。
- **数据量**：当前示例模型多在 4000 行 × 1 epoch 量级训练，生成质量偏弱（语法破碎、偶发 `<unk>`）；提升质量需更大语料与更多 epoch。
- **生成编码**：Windows GBK 终端可能误显中文，建议读取 UTF-8 日志或重定向输出。

## 文档索引

- `CHANGELOG.md`：主要修复与功能变更（对照提交历史）。
- `configs/README.md`：各训练 / 推理配置说明。
- `docs/`：`TRAINING_GUIDE.md`、`TUNING_GUIDE.md`、`MODEL_USAGE_GUIDE.md`、`DATA_USAGE_GUIDE.md` 等。
- `models/README.md`、`scripts/README.md`、`experiments/README.md`、`tools/README.md`、`data/README.md`：分模块说明。
