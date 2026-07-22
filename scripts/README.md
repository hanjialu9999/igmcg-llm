# scripts/

训练 / 推理入口与数据处理脚本（项目主程序所在目录）。

## 入口

| 文件 | 说明 |
|------|------|
| `train.py` | 训练主程序：`python scripts/train.py --config configs/pretrain.yaml`。支持 warmup、早停、自动备份旧检查点（label smoothing 因 PyTorch 限制当前未启用，见 `docs/TRAINING_GUIDE.md`）。 |
| `generate.py` | 生成 API：`generate_text` / `generate_igmcg` / `NGramModel`。IGMCG 多候选按 `1.5*连贯度 + 0.15*流畅度 + 0.15*风格 - 2.5*重复度` 选优，生成期 `repetition_penalty=2.0`。CLI 支持 `--dtype fp32/bf16/auto`（默认 auto：支持的 CPU/CUDA 用 bf16，约 1.5~1.8× 提速且质量基本无损）、`--cpu-threads N`（降功耗）、`--quantize`（纯 CPU int8 动态量化，约 4× 更小模型/降带宽）、`--compile`（需本机 C++ 编译器）。 |
| `chat.py` | 对话式 CLI（参数：`--model` / `--vocab` / `--device` / `--max-length` / `--temperature` / `--top-k`），具体见 `--help`。 |
| `chat_zh.bat` | 中文 Windows 一键对话启动器（自动定位项目根，调用 `scripts/chat.py`）。 |
| `train_finetune.py` | 微调训练（QA 两阶段：预训练底座 → 微调），产出 `best_finetuned_model.pt`。 |
| `build_bpe_vocab.py` · `build_char_vocab.py` | 单独构建 BPE / 字符级词表（训练时也会自动构建）。 |
| `bench_linear2d.py` | 2D 轴向线性注意力基准。 |

## 数据处理

`data_manager.py`（统一入口：`merge` / `stats` / `vocab` / `sample` / `to-jsonl`）· `convert_dialogue_to_qa.py` · `diagnose.py`。

> `merge_data.py` 与 `process_data.py` 现为 `data_manager.py` 的兼容薄包装。

## 子目录

- `data/`：`download_pretrain_data.py` 下载/准备预训练语料。
- `tuning/`：参数扫描（见 `scripts/tuning/README.md`）。
