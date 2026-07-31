# 中文语言模型（Attention × SSM + n-gram 双轨解码）

基于 Transformer 的中文 LM 训练 / 推理项目，融合自定义混合架构（注意力 × SSM × IGMCG 直觉引导解码）与统计式 n-gram 双轨解码。目标是在 CPU / AMD iGPU（DirectML）等低资源设备上也能训练并跑出连贯的中文生成。

当前状态（2026-07-31，R39）：全量测试 **922 passed / 2 skipped / 1 xfailed**；DML 训练实测 **7346 tok/s**（12 层，R38 优化后）。

## 特性

### 核心架构
- **混合主干**：`TransformerModel` 支持纯注意力，或与 `MambaSSM`（含 `ssm_type=cast` 变体）组成的 SSM×注意力混合架构；均带 KV-cache，生成高效。
- **现代结构**：Pre-LN + RMSNorm + RoPE + SwiGLU + QK-Norm + 可学习注意力温度（见 `models/transformer.py`）。
- **向量化 SSM 扫描**：选择性扫描用并行前缀扫描实现（log2(L) 步），既加速 CPU 训练，也避免低功耗 iGPU 上 DML 因单步 kernel 过多触发 TDR 设备重置。
- **双轨解码**：神经模型输出与统计 n-gram 先验在解码期叠加，互补长短。
- **IGMCG 反碎片化**：生成多个温度候选，按综合分（连贯度 + 流畅度 + 风格 − 重复度）选优，抑制"碎片式"输出。
- **跨设备**：自动选择 `cuda` / `dml`(AMD) / `cpu`；DML 推理与训练均已验证可用；bf16 混合精度在 CPU/CUDA 上开启。

### 增强架构机制（opt-in，默认关，向后兼容旧权重）
全部由 `config['model']` 开关控制，旧权重（无该键）加载时自动关。开启后需重新训练才能学到有效参数（门控默认 init 1.0 ≈ 普通残差）。详见各 `configs/*.yaml` 注释与 `models/model_config.py`。

- **记忆类**：可学习遗忘 `MemoryBank`（memory_forget）、全上下文检索（memory_retrieval_full / memory_retrieval_topk）、product_key、统一记忆预算（memory_budget）、SSM 作为隐式记忆（ssm_as_memory）。
- **注意力类**：可学 RoPE 频率 + ALiBi（rope_learnable / alibi_learnable / shared_alibi）、可学滑动窗口（learn_window）、Partial RoPE（rope_dim_fraction）、NoPE 层（nope_layers）、维度级 RoPE（dim_wise_rope）、per-head 注意力温度（head_temp）、value-side 相对编码（value_relative_coding）、GPAS 梯度保留激活缩放、头内 RoPE/NoPE 混合（intra_hybrid_rope）、YaRN 长度外推（yarn_scale）、Output Gating（output_gate）、Zero-Centered RMSNorm（zero_centered_norm）。
- **MLA KV 潜空间压缩（use_mla_kv + kv_latent_dim）**：K/V 拼接后下投影到低维潜空间，cache 只存潜向量，长序列 KV cache 内存降 `2*dim/kv_latent_dim` 倍（灵感：DeepSeek-V3 MLA）。
- **线性注意力**：`mixer=linear|hybrid|gated_delta|attn`；GatedDeltaNet（delta rule + α/β 门控 + k L2 归一化）、RWKV-7 广义 Delta Rule（rwkv7）、KDA 逐通道衰减（gated_delta_channel_wise）、**chunk-wise 矩阵前缀扫描（gated_delta_chunk_scan）**把 S 更新从 O(T²) 降到 O(T log T)（DML 实测 T=64 快 7.2×）；线性注意力修正（linear_correction）、可配 head_dim。
- **效率类**：SwiGLU w1/w3 合并（fuse_swiglu，省 1 次 GEMM 调用）、KV cache int8 量化（kv_cache_int8，内存 4×，标准 attn 路径有效）、gradient checkpointing 自动禁用（grad_ckpt_auto）、**MoE FFN（moe）**：top-k 路由多专家替换 SwiGLU，含负载均衡 + router z-loss（DML 兼容 dense 实现）、**提前退出（early_exit）**：训练期出口层辅助 CE 损失 + 推理期置信度阈值提前返回（增量解码因 KV cache 一致性不启用）。
- **跨层协作**：cross_layer_routing（Top-k 稀疏路由）、cross_ssm_transfer、progressive_residual（1/√d 残差衰减）、layer_film（跨层 FiLM）、highway_gate / input_highway（动态残差 / 嵌入门控注入）、layer_contrastive（训练期相邻层 cos_sim 损失）、shared_alibi、DALA 对齐训练（aligned_training）。
- **解码 / 训练类**：n-gram 融合（ngram_fusion）、IGMCG 多候选解码、QAT 量化感知训练（qat_bits，LSQ-STE）、CharMerge（char_merge）、层跳过（layer_skip）、层共享（share_attn_proj / share_ffn / share_norm）。

## 目录结构

```
models/          模型与基础设施
                  transformer.py  (TransformerModel / TransformerBlock / EnhancementsMixin, 支持 KV-cache)
                  mixers.py       (SlidingWindow/Linear/AxialLinear/Differential/GatedDeltaNet/MambaSSM(+CAST)/SwiGLU)
                  moe.py          (MoELayer: dense MoE, top-k 路由 + 负载均衡 + z-loss)
                  norms.py / rope.py / memory.py / sampling.py / layers.py (CharMerge)
                  model_config.py (ModelConfig dataclass, 全特性开关) / state.py (BlockState) / constants.py
                  data_utils.py   (BaseTokenizer: 字符级零 OOV + 字节回退 + CharTokenizer/BPE)
                  config_loader.py (load_config / build_model) / checkpoint.py (load_model, 自动旧权重转换)
                  device.py       (get_device / apply_cpu_threads)
scripts/         入口与数据处理
                   train.py        训练主程序 (--config)
                   train_finetune.py  微调训练（QA 两阶段：预训练底座 → 微调，产出 best_finetuned_model.pt）
                   generate.py     生成 API: generate_text / generate_igmcg / NGramModel
                   chat.py         对话式 CLI (--model / --vocab / --device / --max-length / --temperature / --top-k)
                   chat_zh.bat     中文 Windows 一键对话启动器
                    merge_data.py, process_data.py, convert_dialogue_to_qa.py, data_manager.py ...
                   data/download_pretrain_data.py, tuning/  (参数扫描)
configs/         所有 YAML 配置 (pretrain.yaml 为规范默认；config_full_dml.yaml 为 DML 生产配置)
experiments/     实验 / 诊断 / 一次性脚本（可独立运行，自带路径修正）
tools/           检查与监控工具 (view_model / compare_epochs / dialogue / dialogue_interactive / monitor/ ...)
tests/           正式 pytest 单元测试（已纳入 git 跟踪，当前 922 passed / 2 skipped / 1 xfailed）
test/            本地自测沙箱（gitignore，仅本机运行，不入库）
data/            语料 (pretrain_corpus/) 与数据集 (datasets/)
logs/            运行日志
checkpoints/     训练产出（含 final_model.pt / vocab.json；chat / diagnose / tune 使用）
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
python scripts/chat.py
```

更多示例：

```bash
# 指定权重 / 词表 / 设备（例如 AMD DML 推理）
python scripts/generate.py \
    --model checkpoints/final_model.pt \
    --vocab checkpoints/vocab.json \
    --device dml --dtype fp32 \
    --prompt "中国的首都是" --ngram --igmcg --max-length 60

# 诊断模型输出（前向 / Top-k 分布）
python scripts/diagnose.py --model checkpoints/final_model.pt --vocab checkpoints/vocab.json --device auto

# 参数扫描（Top-K / Temperature）
python scripts/tuning/tune_topk.py --model checkpoints/final_model.pt --vocab checkpoints/vocab.json --device auto
```

### 完整快速开始（数据 / 微调 / 调参 / FAQ）

**准备数据**：小样本调试可用 `merged_sample.txt`（由 `scripts/data_manager.py merge` 自动生成）；原始 QA 数据在 `data/datasets/`（仅本地保留，未入库）。重新构建语料：

```bash
python scripts/data_manager.py merge     # 合并 datasets/ 下语料并构建词表（统一入口）
python scripts/data_manager.py to-jsonl   # 转换为 jsonl（可选）
```

**训练基座**：`python scripts/train.py --config configs/pretrain.yaml`；结束在 `checkpoints/` 产出 `final_model.pt` 与 `vocab.json`。支持 warmup、混合精度、早停、自动清理旧检查点与断点续训（resume 恢复 global_step / best_loss / LR 进度）。

**微调（可选）**：`python scripts/train_finetune.py`（数据来自 `data/datasets/`，产出 `best_finetuned_model.pt`，供 `chat.py` / `dialogue_interactive.py` 使用）。优化器 / 学习率 / 轮数均从 `configs/pretrain.yaml` 的 `training` 段读取；最佳权重与 `_config.yaml` 同步保存，可被 `load_model` 原样加载。

**CPU 推理提速**：`--dtype bf16`（默认 auto 启用，CPU/CUDA 约 1.5~1.8× 提速且质量基本无损）；`--cpu-threads N` 限线程；`--quantize` 启 int8 动态量化（约 4× 更小模型、质量无损）。

**调参**：`scripts/tuning/tune_temperature.py` / `tune_topk.py` / `showcase_optimal_params.py`（展示最优参数并回写 `chat_config.json`）。

**常见问题**：
- `FileNotFoundError: vocab.json`：先执行训练（或单独构建词表）。
- 换模型结构：只改 `configs/pretrain.yaml` 的 `model` 段，所有脚本自动同步。
- 显存不足：调小 `training.batch_size`，或微调脚本中减小 `DataLoader` 的 `batch_size`。

## 模型架构

- **主干**：`TransformerModel` 为 Pre-LN + RMSNorm + RoPE + SwiGLU 的 Transformer；可通过配置切换为含 `MambaSSM` 的 SSM×注意力混合架构。
- **MambaSSM 选择性扫描**：已用**并行前缀扫描**向量化（log2(L) 步、数值稳定），消除逐时间步 `for` 循环：大幅加快 CPU 训练，并避免低功耗 iGPU 上 DML 因单步 kernel 过多触发 TDR 设备重置。`ssm_type=cast` 提供带 CAST 混合精度块的变体。
- **KV-cache**：`generate()` 支持 `use_cache`，自回归逐 token 解码只算新增一步，速度随序列长度近线性；可选 int8 量化（`kv_cache_int8`）。
- **GatedDeltaNet**（`mixer=gated_delta`）：delta rule 线性注意力改进，`S_t = α_t·S_{t-1} + β_t·(v_t − S_{t-1}·k_t)⊗k_t`，k L2 归一化；`gated_delta_chunk_scan` 用矩阵前缀扫描把 S 更新降到 O(T log T)（DML 实测 T=16: 2.3× / T=32: 4.1× / T=64: 7.2×）。
- **MoE**（`moe`）：dense MoE（所有专家对所有 token 求值，广播比较建掩码，无 gather/scatter）——DML 上可训练可推理；top-k 路由 + 负载均衡 + z-loss。
- **双轨解码**：神经对数概率与 n-gram 模型统计先验在解码期按权重叠加，n-gram 只遍历与上下文相关的少量 token，开销极低。
- **IGMCG**：见下节。

## 训练

- 配置集中在 `configs/`（详见 `configs/README.md`）。`pretrain.yaml` 为规范默认：词表 8000、6 层、emb512、单轮遍历。DML 生产配置见 `configs/config_full_dml.yaml`（ngram_fusion + 记忆检索 + 窗口注意力等）。
- 数据：`data/pretrain_corpus/merged.txt` 为默认训练语料（本地，不入库）；小样本调试可用 `merged_sample.txt`。词表在训练时自动构建，存于 `checkpoints/vocab.json`。
- **混合精度**：`precision: bf16` 在 **CPU / CUDA** 开启（约 2~2.5× 提速、loss 基本无损）；`fp16` 仅 CUDA（启用 GradScaler）；AMD DirectML 暂不支持 AMP，自动回退 fp32。
- **DML 训练**：SSM/hybrid 的选择性扫描已向量化，可在 DML 上正常训练；AdamW 已打 `_dml_foreach_lerp` patch（把 `lerp_` 替换为 4 算子展开，避免 DML CPU 回退）。
- **增强调度（SELv2）**：`pretrain.yaml` 与 `config_full_dml.yaml` 默认带 SELv2 分段选择性增强调度（`training.enhancement_schedule`），按 8 段掩码交替开关增强特性训练——实验表明含"全关极端"的 SELv2 在分布外 / 极端采样下泛化最好。

## 推理与生成

`scripts/generate.py` 主要参数：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--prompt` / `--prompt-file` | 输入文本或文本文件 | — |
| `--max-length` | 生成最大长度 | 30 |
| `--temperature` | 采样温度（0=贪心） | 0.8 |
| `--top-k` | Top-K 截断 | 50 |
| `--repetition-penalty` | 重复惩罚（>1 抑制重复，1.0=关闭） | 1.4 |
| `--device` | `cpu` / `cuda` / `dml` / `auto` | `auto` |
| `--dtype` | `fp32` / `bf16` / `auto` | `auto` |
| `--quantize` | 启用 int8 动态量化（纯 CPU） | 关 |
| `--cpu-threads` | 限制 CPU 线程数（降功耗） | 4 |
| `--ngram` / `--ngram-weight` | 开启 n-gram 双轨及权重 | 关 / 0.3 |
| `--igmcg` / `--igmcg-candidates` | 开启 IGMCG 及候选数 | 关 / 5 |
| `--intuition` | 7 维直觉向量（逗号分隔） | 全 0.5 |
| `--model` / `--vocab` | 显式指定权重 / 词表 | `checkpoints/` |

> **DML 推理**：权重统一先加载到 CPU 再 `.to(device)` 搬运，生成路径改用 `torch.no_grad()`（DML 后端不支持 `inference_mode`，会报 `Cannot set version_counter for inference tensor`）。

## IGMCG 反碎片化设计

IGMCG 生成多个温度候选，按综合分选优：

```
score = 1.5 * 连贯度(coh) + 0.15 * 流畅度 + 0.15 * 风格匹配 - 2.5 * 重复度
```

- **连贯度(coh)**：用 n-gram 模型计算序列相邻 token 的预测概率，越高=越相连，是抑制"碎片化"的核心信号。
- 流畅度（单 token 置信度）只作轻微 tiebreaker——孤立高频词也会拉高它，故不主导。
- 风格匹配为 7 维直觉的温和偏置（在连贯候选间微调，绝不压过连贯度）。
- 候选温度范围收窄 (0.75~1.35×)，生成期重复惩罚 2.0，避免候选本身过度发散或循环。

## 性能（AMD 780M iGPU / DirectML 实测）

训练与生成速度均在本机验证（`privateuseone:0`，fp32）：

- **DML 训练（R38，2026-07-31）**：12 层 smoke 模型（12 特性全开 + adamw + attn）修复 `torch.lerp` CPU 回退后，step 243.4ms → 209.1ms，**7346 tok/s（+16.4%）**；修复前 6311 tok/s。
- **DML 训练（R37，2026-07-31）**：同一 smoke 配置 1 epoch 实测 **6068~6078 tok/s**、3.95 it/s、loss 6.68、Val Best 5.8330——12 层 DML 历史最优。
- **GatedDeltaNet chunk_scan**：T=16 快 2.3× / T=32 快 4.1× / T=64 快 7.2×（增量解码 T=1 略慢 0.84×，绝对差 0.6ms 可忽略）。
- **生成（DML，2026-07 数据）**：基线 top-k 约 40 tok/s；IGMCG 开约 12 tok/s。纯注意力模型 **CPU** 生成约 107 tok/s（KV-cache，4 线程）；IGMCG 多候选经批量化前向，有效吞吐约 290 tok/s（含打分）。
- **n-gram 先验**：解码期按需计算（仅遍历与上下文相关的少量 token），开销可忽略。

> 说明：DML 上 `torch.compile` 不可用（`backend='directml'` 无效，OpaqueTensorImpl 存储访问报错），手动算子级优化是唯一提速路径（R38 已验证）。

## DML 已知坑（已修复 / 规避）

- `torch.lerp` / `lerp_`（如 AdamW 更新）在 DML 回退 CPU——已用 4 算子展开替换（`models/layers.py`、`models/ngram.py`、train.py patch）。
- `torch.eye(D, device=X)` 在 DML 后端返回空张量——先 CPU 创建再 `.to(device)`。
- `torch.cat` 拼接空 tensor（如 num_chunks==1 时）DML 报"参数错误"——特判跳过。
- `inference_mode` 不可用——用 `no_grad()`。
- MoE 的 `topk` 反向在 DML 报 `device_ready_queues_.size() INTERNAL ASSERT FAILED`（scatter 限制）——topk 仅取索引放 `no_grad()`，权重经掩码传播梯度（`models/moe.py`）。

## 已知限制 / 注意事项

- **词表**：`vocab.json` 中存在少量 `U+FFFD` 替换字符条目（语料读取 `errors='replace'` 所致），对生成质量影响极小，后续做语料清洗时可一并修复。
- **DML 精度**：AMD DirectML 不支持 AMP，训练 / 推理在 DML 上均为 fp32；bf16 仅 CPU/CUDA。
- **数据量**：当前示例模型多在 4000 / 8000 行 × 1 epoch 量级冒烟训练，生成质量偏弱（语法破碎、偶发 `<unk>`）；提升质量需更大语料与更多 epoch。
- **生成编码**：Windows GBK 终端可能误显中文，建议读取 UTF-8 日志或重定向输出。
- **早期退出（early_exit）**：增量解码（use_cache=True）不启用提前退出——每层每 token KV 必须完整，提前退出会破坏后续 attention 一致性。

## 文档索引

- `CHANGELOG.md`：主要修复与功能变更（本地维护，不入库，对照提交历史）。
- `configs/README.md`：各训练 / 推理配置说明。
- `docs/`：`TRAINING_GUIDE.md`、`TUNING_GUIDE.md`、`MODEL_USAGE_GUIDE.md`、`DATA_USAGE_GUIDE.md` 等。
- `models/README.md`、`scripts/README.md`、`experiments/README.md`、`tools/README.md`、`data/README.md`：分模块说明。
