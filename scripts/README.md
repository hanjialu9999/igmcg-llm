# scripts/

训练 / 推理入口与数据处理脚本（项目主程序所在目录）。

## 入口

| 文件 | 说明 |
|------|------|
| `train.py` | 训练主程序：`python scripts/train.py --config configs/pretrain.yaml`。支持 warmup、早停、分阶段增强调度（`enhancement_schedule` / `enhancement_off_prob`）、Controller 按步交替、打包语料、自动备份旧检查点（label smoothing 因 PyTorch 限制当前未启用，见 `docs/TRAINING_GUIDE.md`）。 |
| `generate.py` | 生成 API：`generate_text` / `generate_igmcg` / `NGramModel`。IGMCG 多候选按 z-score 后的综合分（连贯度 / 流畅度 / 风格 / 多样性）选优，生成期 `repetition_penalty=2.0`。CLI 支持 `--dtype fp32/bf16/auto`（默认 auto：支持的 CPU/CUDA 用 bf16）、`--cpu-threads N`、`--quantize`（纯 CPU int8 动态量化）、`--compile`、`--ngram` 固定 n-gram 先验、`--igmcg`、`--controller-direction`（生成期 direction 因果化）、`--char-merge-buffer`（增量 char-merge 缓冲）。 |
| `chat.py` | 对话式 CLI（参数：`--model` / `--vocab` / `--device` / `--max-length` / `--temperature` / `--top-k` / `--repetition-penalty` / `--controller-direction`），具体见 `--help`。 |
| `chat_zh.bat` | 中文 Windows 一键对话启动器（自动定位项目根，调用 `scripts/chat.py`）。 |
| `train_finetune.py` | 微调训练（QA 两阶段：预训练底座 → 微调），产出 `best_finetuned_model.pt`。 |
| `build_bpe_vocab.py` · `build_char_vocab.py` | 单独构建 BPE / 字符级词表（训练时也会自动构建）。 |
| `bench_linear2d.py` | 2D 轴向线性注意力基准。 |

## 评测 / 对比

| 文件 | 说明 |
|------|------|
| `baseline_eval.py` | 质量基准：tf / prefix / incremental 三口径 ppl + 固定提示词生成，落 JSON。 |
| `eval_pipeline.py` | 一键评测流程：4 个 CPU 评测 → 生成对比 → 盲评出题 → 裁判任务，产物落工作目录。 |
| `gen_cmp.py` | 生成对比：各模型 Controller 开/关 × 提示词 × 采样配置，输出 JSON。 |
| `gen_tune.py` | 生成端单因素调参（top_p / repetition_penalty / eos_penalty 变体）。 |
| `blind_eval.py` | 盲评工具：`make` 出题、`task` 生成裁判任务、`score` / `agree` / `mark` 汇总。 |

## 数据处理

`data_manager.py`（统一入口：`merge` / `stats` / `vocab` / `sample` / `to-jsonl`）· `convert_dialogue_to_qa.py` · `diagnose.py`。

> `merge_data.py` 与 `process_data.py` 现为 `data_manager.py` 的兼容薄包装。

## 诊断 / 剖析（`_` 前缀，只读工具）

`_profile_train_steps.py`（50 步训练分段计时）· `_prefix_loss_by_pos.py`（逐位置 prefix loss）· `_diag_dml_layer_diff.py` · `_diag_sdpa_dml.py` · `_diag_vrc_conv_dml.py`（DML/CPU 算子级偏离定位）。

## 子目录

- `data/`：`download_pretrain_data.py` 下载/准备预训练语料、`fetch_ms_corpus.py` 从魔搭拉取语料。
- `tuning/`：参数扫描（见 `scripts/tuning/README.md`）。
