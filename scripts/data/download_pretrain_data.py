"""
download_pretrain_data.py

中英文预训练数据收集与预处理工具。
由于部分海外网站（HuggingFace、GitHub）在国内访问不稳定，本脚本以
"半自动 + 详细指引" 方式帮助你收集适合 20M 参数模型预训练的数据。

用法:
    python scripts/data/download_pretrain_data.py
        -> 创建 data/pretrain_corpus/ 目录并生成详细的手动收集指南

    python scripts/data/download_pretrain_data.py --prepare
        -> 将你手动放入 data/pretrain_corpus/raw/ 的原始文件自动转换为训练格式

策略:
    1. 用大量通用中英文语料预训练 base 模型 (保存在 data/pretrain_corpus/)
    2. 用现有 data/datasets/ 中的问答对微调 (SFT)
"""

import os, sys, json, glob, logging, argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = "data/pretrain_corpus"
RAW_DIR = os.path.join(OUTPUT_DIR, "raw")


def prepare_raw_data():
    """将 data/pretrain_corpus/raw/ 中的原始文件转换为训练格式."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    total = 0
    raw_files = glob.glob(os.path.join(RAW_DIR, "*"))
    if not raw_files:
        logger.info("raw/ 目录为空，请先手动下载数据放入 data/pretrain_corpus/raw/")
        logger.info("参考下方 MANUAL.md 中的指引")
        return 0

    for fpath in raw_files:
        fname = os.path.basename(fpath)
        base, ext = os.path.splitext(fname)
        out_path = os.path.join(OUTPUT_DIR, f"{base}.txt")
        if os.path.exists(out_path):
            logger.info(f"[跳过] {out_path} 已存在")
            continue

        count = 0
        with open(out_path, "w", encoding="utf-8") as fout:
            # ---- JSONL ----
            if ext in (".jsonl", ".json"):
                import json as _json
                with open(fpath, "r", encoding="utf-8") as fin:
                    for line in fin:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = _json.loads(line)
                        except _json.JSONDecodeError:
                            continue

                        text = _extract_text_from_json(data)
                        if text and len(text) >= 10:
                            fout.write(text + "\n")
                            count += 1
                logger.info(f"  [{fname}] JSONL -> {count} lines")

            # ---- Parquet ----
            elif ext == ".parquet":
                try:
                    import pandas as pd
                    df = pd.read_parquet(fpath)
                    for col in df.columns:
                        if df[col].dtype == "object":
                            texts = df[col].dropna().astype(str)
                            for t in texts:
                                t = t.strip()
                                if len(t) >= 10:
                                    fout.write(t + "\n")
                                    count += 1
                            break
                    logger.info(f"  [{fname}] Parquet -> {count} lines")
                except ImportError:
                    logger.warning(f"  [{fname}] 需要 pandas 库: pip install pandas pyarrow")

            # ---- 纯文本 ----
            elif ext == ".txt":
                with open(fpath, "r", encoding="utf-8") as fin:
                    for line in fin:
                        line = line.strip()
                        if line and len(line) >= 10:
                            fout.write(line + "\n")
                            count += 1
                logger.info(f"  [{fname}] TXT -> {count} lines")

            # ---- CSV ----
            elif ext == ".csv":
                try:
                    import pandas as pd
                    df = pd.read_csv(fpath)
                    for col in df.columns:
                        if df[col].dtype == "object":
                            texts = df[col].dropna().astype(str)
                            for t in texts:
                                t = t.strip()
                                if len(t) >= 10:
                                    fout.write(t + "\n")
                                    count += 1
                            break
                    logger.info(f"  [{fname}] CSV -> {count} lines")
                except ImportError:
                    logger.warning(f"  [{fname}] 需要 pandas 库")

            else:
                logger.warning(f"  [{fname}] 不支持的格式: {ext}")
                continue

            total += count

        if count == 0:
            os.remove(out_path)

    logger.info(f"\n总计预处理 {total} 条样本")
    logger.info(f"提示: 用 type {OUTPUT_DIR}\\*.txt > {OUTPUT_DIR}\\merged.txt 合并所有文件")
    return total


def _extract_text_from_json(data):
    """从 JSON 对象中提取文本，兼容多种数据集格式."""
    if isinstance(data, str):
        return data

    # 常见单文本字段
    for key in ("text", "content", "sentence", "passage", "document", "txt", "instruction"):
        val = data.get(key)
        if isinstance(val, str) and len(val) > 10:
            return val

    # conversation 格式
    if "conversations" in data or "conversation" in data:
        key = "conversations" if "conversations" in data else "conversation"
        turns = data[key]
        if isinstance(turns, list):
            parts = []
            for t in turns:
                if isinstance(t, dict):
                    parts.append(t.get("value", t.get("content", "")))
            if parts:
                return " [SEP] ".join(parts)

    # instruction / output 格式
    if data.get("instruction") and data.get("output"):
        return f"{data['instruction']} [SEP] {data['output']}"

    # 问答格式
    if data.get("question") and data.get("answer"):
        return f"{data['question']} [SEP] {data['answer']}"

    # 取第一个字符串字段
    for k, v in data.items():
        if isinstance(v, str) and len(v) > 20:
            return v

    return None


def write_manual_guide():
    """撰写详细的手动数据收集指南."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    guide = """# 中英文预训练数据 — 手动收集指南

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
  3. type data\\pretrain_corpus\\*.txt > data\\pretrain_corpus\\merged.txt
  4. 修改 config/config.yaml 中的 train_file
  5. python scripts/train.py --config config/config.yaml  (预训练)
  6. python train_finetune.py                             (微调)
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
        f.write(ex["text"].strip() + "\\n")

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
| 总参数量 | ~20M | - | 可运行在 CPU/AMD GPU |

> **先用现有数据 (data/train_data_final.txt) 跑通全流程，再逐步增加数据量。**
> 太多数据反而可能让 20M 模型欠拟合，建议从 3万条开始试。
"""
    with open(os.path.join(OUTPUT_DIR, "MANUAL.md"), "w", encoding="utf-8") as f:
        f.write(guide)
    logger.info(f"手动收集指南已写入: {os.path.join(OUTPUT_DIR, 'MANUAL.md')}")


def main():
    parser = argparse.ArgumentParser(description="中英文预训练数据收集与预处理")
    parser.add_argument("--prepare", action="store_true",
                        help="预处理 raw/ 中的原始文件")
    args = parser.parse_args()

    if args.prepare:
        prepare_raw_data()
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(RAW_DIR, exist_ok=True)
        write_manual_guide()
        logger.info(f"目录结构已创建:")
        logger.info(f"  {OUTPUT_DIR}/")
        logger.info(f"    raw/          <- 把你下载的原始文件放这里")
        logger.info(f"    MANUAL.md     <- 详细的手动收集指南")
        logger.info(f"运行 python scripts/data/download_pretrain_data.py --prepare 来转换")


if __name__ == "__main__":
    main()
