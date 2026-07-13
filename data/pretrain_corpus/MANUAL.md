# 中英文预训练数据 — 手动收集指南

## 推荐策略

```
预训练 (pretrain_corpus/)                微调 (data/datasets/)
  ├── 中文通用语料  ~10万条                    ├── ai_qa.txt
  ├── 英文通用语料  ~5万条                     ├── trivia_qa.txt
  ├── 中英问答对    ~1万条                     ├── history_qa.txt
  └── merged.txt                             └── ... (21 个文件)

步骤:
  1. 用下面方法下载数据, 放入 data/pretrain_corpus/raw/
  2. python scripts/data/download_pretrain_data.py --prepare
  3. type data\pretrain_corpus\*.txt > data\pretrain_corpus\merged.txt
  4. 修改 configs/pretrain.yaml 中的 train_file
  5. python scripts/train.py --config configs/pretrain.yaml  (预训练)
  6. python scripts/train_finetune.py                             (微调)
```

---

## 方法一：ModelScope 网页下载 (国内最快，推荐)

用浏览器打开以下链接，可以直接下载预处理好的 `.jsonl` 文件：

| 数据集 | 链接 | 文件大小 | 说明 |
|--------|------|----------|------|
| MiniMind 高质量预训练语料 | https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files | ~1.6GB | 中文为主，来自匠数科技，适合小模型 |
| MiniMind 精简SFT数据 | (同上) 下载 `sft_mini_512.jsonl` | ~1.2GB | 中文问答对 |
| 匠数大模型SFT数据 | https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data/files | 4B tokens | 10M中文+2M英文 |

**推荐**: 下载 `gongjy/minimind_dataset` 中的 `pretrain_hq.jsonl`（取前10万条即可）

下载后放入: `data/pretrain_corpus/raw/`

---

## 方法二：通过 hf-mirror.com (HuggingFace 国内镜像)

设置镜像后, 用 datasets 库流式下载:

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

# 英文教育语料 (适合小模型)
ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2",
                  split="train", streaming=True)
with open("data/pretrain_corpus/raw/smollm_en.txt", "w", encoding="utf-8") as f:
    for i, ex in enumerate(ds):
        if i >= 50000: break
        f.write(ex["text"].strip() + "\n")

# 中英问答对
import json
from huggingface_hub import hf_hub_download
path = hf_hub_download("shareAI/ShareGPT-Chinese-English-90k",
                        "sharegpt_jsonl/common_zh_70k.jsonl",
                        repo_type="dataset")
# 然后处理...
```

---

## 方法三：直接从网页下载公开数据集

以下数据集可通过浏览器直接下载（无需翻墙）：

| 数据集 | 直接下载链接 | 说明 |
|--------|-------------|------|
| LCSTS 中文摘要 | https://huggingface.co/datasets/li2017/LCSTS | 中文短文本摘要 |
| Chinese Poetry | https://github.com/chinese-poetry/chinese-poetry | 中文古诗 |
| 百科问答 | https://github.com/bojone/WebQA | 中文百科问答 |
| 中文维基百科 | 百度搜索 "zhwiki 2025 下载" | 百度网盘很多 |

---

## 关于 20M 参数模型的数据量建议

| 阶段 | 建议数据量 | 训练轮数 | 效果预期 |
|------|-----------|---------|---------|
| 预训练 (base) | 3万-10万条文本 | 10-30 epoch | 理解语言基本结构 |
| 微调 (SFT) | 现有 ~7300 条问答对 | 10-20 epoch | 学会回答问题 |
| 总参数量 | ~25M（6 层 / emb512） | - | 可运行在 CPU/AMD GPU |

> **先用现有数据 (data/pretrain_corpus/merged_sample.txt) 跑通全流程，再逐步增加数据量。**
> 太多数据反而可能让 25M 模型欠拟合，建议从 3万条开始试。
