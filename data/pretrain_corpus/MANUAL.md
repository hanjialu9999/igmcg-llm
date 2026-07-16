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

## 方法零（推荐，自动化）：`fetch_ms_corpus.py`

在**已装 modelscope 的虚拟环境**（如 `F:\Projects\.my_venv`，无需代理，直连魔搭）运行：

```powershell
F:\Projects\.my_venv\Scripts\python.exe scripts/data/fetch_ms_corpus.py
```

脚本会：① 从魔搭下载多领域语料到 `data/pretrain_corpus/raw/`；② 调 `download_pretrain_data.py --prepare` 转成训练 txt；
③ 以「原始 merged.txt 基底 + 本次新增文件」重建 `merged.txt`（**不会重复**已含于基底的语料）。

默认目标 ≈ 600MB 训练数据，多领域混合：
- `gongjy/minimind_dataset`：`agent_rl`(工具调用)、`agent_rl_math`(数学)、`lora_exam`(考试)、
  `lora_medical`(医疗)、`rlaif`(RLHF 对话) —— 注意该仓库**已无 `pretrain_hq.jsonl`**，请勿再引用。
- `AI-ModelScope/wikipedia-cn-20230720-filtered`：中文维基百科（截断到 400MB jsonl ≈ 390MB 文本）。

实测下载速度（2026-07-14，直连）：小型文件 ~10MB/s，维基百科 ~9–11MB/s。
`dpo.jsonl` 为偏好数据（chosen/rejected 格式），`download_pretrain_data.py` 暂无法抽取文本 → 0 行，可忽略。

下载后放入: `data/pretrain_corpus/raw/`（若手动下载）

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
