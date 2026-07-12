# 中文语言模型 (Attention × SSM × IGMCG + n-gram 解码)

基于 Transformer 的中文 LM 训练/推理项目，融合自定义架构（注意力 × SSM × IGMCG 直觉引导解码）与统计式 n-gram 双轨解码。

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

## 快速开始

```bash
# 训练（默认规范配置）
python scripts/train.py --config configs/pretrain.yaml

# 生成（神经 + n-gram 双轨 + IGMCG 多候选联合解码）
python scripts/generate.py --prompt "今天天气怎么样" --ngram --igmcg --ngram-weight 0.3

# 对话
python scripts/chat.py --ngram --igmcg --intuition 0.3,0.8,0.5,0.2,0.6,0.4,0.5
```

GPU/加速：设备上 `get_device('auto')` 自动选择 cuda / dml(AMD) / cpu；推理默认在支持的 CPU/CUDA 上用 **bf16 精度（约 1.5~1.8× 提速，质量基本无损）**，可用 `--dtype fp32|bf16|auto` 控制。CPU 生成可用 `--cpu-threads N` 限制线程数降功耗；纯 CPU 还可加 `--quantize` 启用 int8 动态量化，进一步降低内存带宽与功耗（约 4× 更小模型，质量无损）。`--compile` 需本机有 C++ 编译器才会生效，否则自动回退 eager。

训练侧可用 `precision: bf16` 在 **CPU / CUDA** 开启混合精度训练（约 2~2.5× 提速、loss 基本无损）；`fp16` 仅 CUDA（启用 GradScaler）；AMD DirectML 暂不支持 AMP，自动回退 fp32。SSM/hybrid 架构的选择性扫描已向量化，可在 DML 上正常训练（旧版逐时间步 for 循环会因 kernel 风暴触发 iGPU 设备重置）。

推理侧 **DML 设备现已可用**：权重统一先加载到 CPU 再 `.to(device)` 搬运，生成路径改用 `torch.no_grad()`（DML 后端不支持 `inference_mode`，会报 `Cannot set version_counter for inference tensor`）。`generate.py` / `diagnose.py` / `scripts/tuning/*.py` 均支持 `--model` / `--vocab` / `--device` 显式指定权重与词表（默认 `checkpoints/`）。

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
