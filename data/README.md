# data/

训练语料与数据集。**注意：本目录下除 `pretrain_corpus/MANUAL.md` 外的所有内容均不纳入 git（见根目录 `.gitignore`），仅本地保留。**

| 路径 | 说明 | 是否入库 |
|------|------|----------|
| `pretrain_corpus/merged.txt` | 默认主训练语料（中文+英文混合），`configs/pretrain.yaml` 引用。 | 否（本地） |
| `pretrain_corpus/merged_sample.txt` | 小样本版，用于快速调试。 | 否（本地） |
| `pretrain_corpus/merged_8k.txt` · `merged_24k.txt` · `merged_72k.txt` | r55 / r56 / r57 语料扩容轮次使用的子集。 | 否（本地） |
| `pretrain_corpus/raw/` | 原始爬取语料。 | 否（本地） |
| `pretrain_corpus/MANUAL.md` | 语料说明文档。 | **是** |
| `datasets/` | 原始 QA 数据（每文件「问题行 / 答案行」交替），用于微调。 | 否（本地） |
| `processed/` | `data_manager.py to-jsonl` 生成的 jsonl（可选）。 | 否（本地） |

> 重新构建语料：`python scripts/data_manager.py merge`（合并 `datasets/`）。词表在训练时由 `data.train_file` 自动构建，存于 `<checkpoint_dir>/vocab.json`（默认配置即 `checkpoints/vocab.json`）；字符级 tokenizer 零 OOV。另可用 `merged_sample.txt` 做小样本冒烟；数据相关配置（`pack` / `tokenizer_pretok` / `long_line_windows` 等）见 `docs/DATA_USAGE.md`。
