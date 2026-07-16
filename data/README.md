# data/

训练语料与数据集。**注意：本目录下除 `pretrain_corpus/MANUAL.md` 外的所有内容均不纳入 git（见根目录 `.gitignore`），仅本地保留。**

| 路径 | 说明 | 是否入库 |
|------|------|----------|
| `pretrain_corpus/merged.txt` | 默认主训练语料（中文+英文混合），`configs/pretrain.yaml` 引用。 | 否（本地） |
| `pretrain_corpus/merged_sample.txt` | 小样本版，用于快速调试。 | 否（本地） |
| `pretrain_corpus/raw/` | 原始爬取语料。 | 否（本地） |
| `pretrain_corpus/MANUAL.md` | 语料说明文档。 | **是** |
| `datasets/` | 原始 QA 数据（每文件「问题行 / 答案行」交替），用于微调。 | 否（本地） |
| `processed/` | `process_data.py` 生成的 jsonl（可选）。 | 否（本地） |

> 重新构建语料：`python scripts/merge_data.py`（合并 `datasets/`）。词表在训练时由 `data/pretrain_corpus/merged.txt` 自动构建，存于 `checkpoints/vocab.json`。另可用 `merged_sample.txt` 做小样本冒烟。
