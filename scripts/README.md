# scripts/

训练 / 推理入口与数据处理脚本（项目主程序所在目录）。

## 入口

| 文件 | 说明 |
|------|------|
| `train.py` | 训练主程序：`python scripts/train.py --config configs/pretrain.yaml`。支持 warmup、label smoothing、早停、自动备份旧检查点。 |
| `generate.py` | 生成 API：`generate_text` / `generate_igmcg` / `NGramModel`。IGMCG 多候选按 `1.5*连贯度 + 0.15*流畅度 + 0.15*风格 - 2.5*重复度` 选优，生成期 `repetition_penalty=1.4`。 |
| `chat.py` | 对话式 CLI：`--ngram` / `--igmcg` / `--intuition`，行为由 `chat_config.json` 控制。 |

## 数据处理

`prepare_data.py` · `prepare_training.py`（合并 `data/datasets/` 为单一语料）· `merge_datasets.py` · `merge_data.py` · `merge_data_clean.py` · `improve_data.py` · `reformat_data.py` · `convert_dialogue_to_qa.py` · `convert_statements_to_qa.py` · `data_manager.py` · `diagnose.py` · `analyze_datasets.py` · `process_data.py`（可选转 jsonl）。

## 子目录

- `data/`：`download_pretrain_data.py` 下载/准备预训练语料。
- `tuning/`：参数扫描（见 `scripts/tuning/README.md`）。
