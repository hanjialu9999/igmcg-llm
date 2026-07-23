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
- **增强架构机制**（默认关闭、向后兼容旧权重）：可学习遗忘 `MemoryBank`（memory_forget）、可学 RoPE 频率 + ALiBi（rope_learnable / alibi）、全上下文检索（memory_retrieval_full / memory_retrieval_topk）、可学滑动窗口（learn_window / window_base）、选择性跳过层（layer_skip）、线性注意力 mixer（mixer=attn|linear|hybrid|gated_delta）、计算复杂度奖励（training.complexity_lambda）、统一记忆预算（memory_budget）、线性注意力修正（linear_correction）、位置编码门控（pe_gate）、跨层稀疏路由（cross_layer_routing）、QAT 量化感知训练（qat_bits, LSQ-STE）、SSM 作为隐式记忆（ssm_as_memory）、跨层协作（cross_ssm_transfer / progressive_residual / layer_film / highway_gate / input_highway / layer_contrastive / shared_alibi）、Partial RoPE（rope_dim_fraction）、Output Gating（output_gate）、Zero-Centered RMSNorm（zero_centered_norm）。详见各 `configs/*.yaml` 注释。

## 目录结构

```
models/          模型与基础设施
                  transformer.py  (TransformerModel / TransformerBlock / EnhancementsMixin, 支持 KV-cache)
                  mixers.py       (SlidingWindow/Linear/AxialLinear/Differential/GatedDeltaNet/MambaSSM/SwiGLU)
                  norms.py / rope.py / memory.py / sampling.py / layers.py (CharMerge)
                  model_config.py (ModelConfig dataclass) / state.py (BlockState) / constants.py
                  data_utils.py   (BaseTokenizer: 字符级零 OOV + 字节回退)
                  config_loader.py (load_config / build_model) / checkpoint.py (load_model)
                  device.py       (get_device / apply_cpu_threads)
scripts/         入口与数据处理
                   train.py        训练主程序 (--config)
                   train_finetune.py  微调训练（QA 两阶段：预训练底座 → 微调）
                   generate.py     生成 API: generate_text / generate_igmcg / NGramModel
                   chat.py         对话式 CLI (--model / --vocab / --device / --max-length / --temperature / --top-k)
                   chat_zh.bat     中文 Windows 一键对话启动器
                    merge_data.py, process_data.py, convert_dialogue_to_qa.py, data_manager.py ...
                   data/download_pretrain_data.py, tuning/  (参数扫描)
configs/         所有 YAML 配置 (pretrain.yaml 为规范默认；dml_* / hybrid_* 为变体)
experiments/     实验 / 诊断 / 一次性脚本 (原根目录 _*.py，可独立运行，自带路径修正)
tools/           检查与监控工具 (view_model / compare_epochs / dialogue / dialogue_interactive / monitor/ ...)
tests/           正式 pytest 单元测试（已纳入 git 跟踪）：test_config_loader.py / test_transformer.py / test_generation_pipeline.py / test_new_mechanisms.py 等（当前 299 passed / 1 skipped）
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
    --model archive_unused/checkpoints_backup/_stab_ckpt/final_model.pt \
    --vocab checkpoints_dml/vocab.json \
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

**训练基座**：`python scripts/train.py --config configs/pretrain.yaml`；结束在 `checkpoints/` 产出 `final_model.pt` 与 `vocab.json`。支持 warmup、混合精度、早停与自动清理旧检查点。

**微调（可选）**：`python scripts/train_finetune.py`（数据来自 `data/datasets/`，产出 `best_finetuned_model.pt`，供 `chat.py` / `dialogue_interactive.py` 使用）。优化器 / 学习率 / 轮数均从 `configs/pretrain.yaml` 的 `training` 段读取。

**CPU 推理提速**：`--dtype bf16`（默认 auto 启用，CPU/CUDA 约 1.5~1.8× 提速且质量基本无损）；`--cpu-threads N` 限线程；`--quantize` 启 int8 动态量化（约 4× 更小模型、质量无损）。

**调参**：`scripts/tuning/tune_temperature.py` / `tune_topk.py` / `showcase_optimal_params.py`（展示最优参数并回写 `chat_config.json`）。

**常见问题**：
- `FileNotFoundError: vocab.json`：先执行训练（或单独构建词表）。
- 换模型结构：只改 `configs/pretrain.yaml` 的 `model` 段，所有脚本自动同步。
- 显存不足：调小 `training.batch_size`，或微调脚本中减小 `DataLoader` 的 `batch_size`。

> 更详尽的步骤即上方「完整快速开始」章节（原 `QUICK_START.md` 已并入本文件）。

## 模型架构

- **主干**：`TransformerModel` 为 Pre-LN + RMSNorm + RoPE + SwiGLU 的 Transformer；可通过配置切换为含 `MambaSSM` 的 SSM×注意力混合架构。
- **MambaSSM 选择性扫描**：已用**并行前缀扫描**向量化（log2(L) 步、数值稳定），消除逐时间步 `for` 循环：大幅加快 CPU 训练，并避免低功耗 iGPU 上 DML 因单步 kernel 过多触发 TDR 设备重置。
- **KV-cache**：`generate()` 支持 `use_cache`，自回归逐 token 解码只算新增一步，速度随序列长度近线性。
- **双轨解码**：神经对数概率与 n-gram 模型统计先验在解码期按权重叠加，n-gram 只遍历与上下文相关的少量 token，开销极低。
- **IGMCG**：见下节。

## 训练

- 配置集中在 `configs/`（详见 `configs/README.md`）。`pretrain.yaml` 为规范默认：词表 8000、6 层、emb512、单轮遍历。**自 2026-07-14 起 `pretrain.yaml` 与 `config_dml_full.yaml` 默认带 SELv2 分段选择性增强调度（`training.enhancement_schedule`），即默认训练即按 SELv2 训练。**
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
| `--repetition-penalty` | 重复惩罚（>1 抑制重复，1.0=关闭） | 1.4 |
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
- 候选温度范围收窄 (0.75~1.35×)，生成期重复惩罚 2.0，避免候选本身过度发散或循环。

## 可配置架构增强（实验性，2026-07-14 起默认开）

以下增强均经 `config['model']` 开关控制（**默认 `True`**），新训练默认即开、无需手动开启；**加载旧权重（无该键）时默认关，保持向后兼容**。开启后需重新训练才能学到有效参数（门控默认 init 1.0 ≈ 普通残差）：

- **QK-Norm**：注意力 Q/K 在做 RoPE 前先过一层 `RMSNorm(head_dim)`，稳定注意力尺度。
- **可学习注意力温度**：每层一个可学习标量 `log_temp`，按 `temp = exp(log_temp)` 缩放 Q/K（替代固定 `/sqrt(head_dim)`）。
- **门控残差（residual_gate）**：每个 block 的注意力 / SSM 分支与 FFN 分支各带一个可学习门控 `nn.Parameter`，相当于可学习的残差缩放（思路 ②/⑥）。
- **混合路径门控（hybrid_gate，⭐A）**：hybrid 块内 attn 与 ssm 两路各自带可学习门控，让模型自决每层偏重。

对应配置键：`qk_norm` / `attn_temp` / `residual_gate` / `hybrid_gate`（`models/config_loader.py` 读取；`scripts/generate.py` 的 `load_model` 从各 checkpoint 的 `*_config.yaml` 透传，旧权重缺省关）。

### 进阶架构特性（第十一~十五轮，默认关 opt-in）

以下特性均为 **默认关、config 显式开启才生效**，保证旧权重向后兼容。灵感来自 Qwen3-Next / Gated DeltaNet (ICLR 2025) / DeepSeek-V3 MLA / RWKV-7 / Kimi KDA 等 2025-2026 主流架构。详见 `CHANGELOG.md` 与 `models/model_config.py`。

- **线性注意力修正（linear_correction）**：对线性 mixer 输出施加可学习修正项。
- **位置编码门控（pe_gate）**：门控 RoPE 注入强度。
- **跨层稀疏路由（cross_layer_routing）**：Top-k 稀疏路由跨层传递信息（init bias=-3 弱注入，STE 传梯度）。
- **QAT 量化感知训练（qat_bits）**：LSQ-STE 可学习步长量化，monkey-patch `Linear.forward` 不破坏 state_dict 兼容。
- **SSM 作为隐式记忆（ssm_as_memory）**：SSM 状态 mean-pool 为单 slot 注入注意力 mem_kv。
- **跨层协作**：`cross_ssm_transfer`（hybrid 块间 SSM 传递）/ `progressive_residual`（残差 1/√d 衰减）/ `layer_film`（跨层 FiLM γ,β 调制）/ `highway_gate`（动态残差门控，与 residual_gate 互斥）/ `input_highway`（embedding x0 门控注入每层）/ `layer_contrastive`（训练期相邻层 cos_sim 损失）/ `shared_alibi`（所有注意力层共用 alibi_slopes buffer 减参）。
- **Gated DeltaNet（mixer='gated_delta'）**：delta rule + α/β 门控的线性注意力改进，`S_t = α_t·S_{t-1} + β_t·(v_t - S_{t-1}·k_t)⊗k_t`，k L2 归一化。
- **Partial RoPE（rope_dim_fraction）**：仅前 k% 维度应用 RoPE，后段 NoPE 纯内容维度。
- **Output Gating（output_gate）**：注意力输出后 sigmoid 门控，消除 Attention Sink / Massive Activation。
- **Zero-Centered RMSNorm（zero_centered_norm）**：先去均值再 RMS 归一化，防止归一化权重异常增大。

### 性能与实测（2026-07-14）
 - 受控 A/B（8000 行×1 epoch，DML fp32）：增强开 Val 7.22 vs 关 7.72；但 DML 训练吞吐一度降至约 2370 tok/s（关约 3270，慢约 28%）。**优化②落地后**（见下）增强训练吞吐回升至约 **3424 tok/s**（与 BASE 差距缩至约 8%）。解码开销可忽略（top-k/IGMCG 约 0.85–0.90x）。详见下方全量主对比（`experiments/cmp_sel_full.txt`）；默认训练现已采用 **SELv2 分段选择性增强**（见 `configs/pretrain.yaml` 的 `training.enhancement_schedule`）。
- **为何变慢（根因）**：`gradient_checkpointing=True` 会在反向时重算整个 block 的前向，导致廉价的 QK-Norm/门控等算子被**执行两次**。微基准（DML，小模型 forward+backward）：`qk_norm` 单独约 +9%、`residual_gate` 约 +3%、`attn_temp` 约 +1%，叠加约 +28%；而关掉梯度检查点后基线本身快约 22%（115.7→90.6 ms/step）。因此该开销主要是检查点重算放大，并非算子本身昂贵。
- **优化点**：① 小模型训练把 `gradient_checkpointing` 设为 `false` 可整体提速（基线即 −22%，增强开销只付一次）。**② 已完成**：`SlidingWindowCausalSelfAttention.forward` 拆为 `project_and_norm`（廉价：QKV 投影+QK-Norm+温度+RoPE）与 `attend`（重算力），`TransformerBlock.forward` 仅对 `attn.attend`/`ssm`/`ffn` 做 `checkpoint`，`ln1`/`ln2`/门控在重算区外只跑一次；微基准 `all` 增强单步 147.8→132.5 ms（约 −10%），训练吞吐 +44%。③ `attn_temp` 已融合为单次标量乘法（免去 `sqrt`）。微基准见 `experiments/_bench_enh.py`。

### 实测对比（2026-07-14，同数据同参数受控 ENH vs SEL）

#### 8000 行（历史，权重/脚本已清理；早期 4 段 SEL，现已被 8 段 SELv2 取代）
8000 行语料（merged.txt 前 8000 行，确定性，非随机）× 1 epoch、DML(fp32) 对照：
- **训练**：ENH(常开) **Val 7.1021 / ~3424 tok/s**；SEL(分段选择性交替：attn_temp 恒开、qk_norm/residual_gate 按 4 段掩码循环) **Val 7.2254 / ~3413 tok/s**；另 ALT(整体 off=0.5) Val 7.3543 / ~3596、BASE(四关) 7.8300 / ~3735（均已弃用清理）。**SEL 优于 ALT**（更接近 ENH），故 ALT 方案已弃用、采用 SEL。
- **生成（推理时同常开）**：top-k ENH 37.2 / SEL 36.5；IGMCG ENH 10.9 / SEL 10.7 tok/s（开销可忽略）。
- **鲁棒性探针（扫温度/top_k 算 self-loss）**：ENH 各设置 self-loss 均略低于 SEL（差距 ~0.1–0.3 nat），ENH 略稳健、SEL 接近且更快。

#### 全量 39700 行（当前主对比，`config_cmp_{enh,sel,selv2}_full.yaml`）

语料扩到 **`merged.txt` 全量 39700 行**、SEL 改用 **SELv2 8 段掩码**（1 全开 + 1 全关极端 + 6 局部；attn_temp 仅全关段关、平时恒开），同参数 1 epoch、DML(fp32)：

- **训练（Val）**：ENH(常开) **5.3762**；SEL旧(8 段无全关) **5.4492**；SELv2(全开+全关) **5.4969**。Val 仍常开 ENH 最低、SELv2 最高（差距与 20k 同量级）。
- **鲁棒性（self-loss，越低越自洽）**：温和设置（T=0.8 / K=30 / K=100）ENH 最优；**SELv2 在极端采样温度最优（T=0.5=2.70、T=1.4=4.14）**，SEL旧在 T=1.1/K=10 最优。即 **SELv2（含全关极端）在分布外/极端采样下泛化最好**——印证"SEL 泛化更好"，20k 时信号太弱未显形。
- **结论**：数据量越大 SEL 越接近常开 ENH（质量/稳健基本持平）且训练更省；SELv2 进一步在分布外鲁棒性上占优。**默认训练方式已改为 SELv2**（见上）。对照权重较大不入库，可由 config + 语料复现。原始生成与 self-loss 数据见 `experiments/cmp_sel_full.txt`（`experiments/_cmp_sel_full.py` 复现；`run_full_cmp.ps1` 一键三模型顺序训练）。
- **注意**：增强开启后须重新训练；`scripts/generate.py` 的 `load_model` 需把四标志从 `*_config.yaml` 透传给 `TransformerModel`，否则增强权重无法加载（见提交 `f9452ed`）。

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
