# 数据使用说明 (Data Usage)

## 目录结构

```
data/
├── pretrain_corpus/        # 主训练语料（本地不入库）
│   ├── merged.txt          # 默认训练语料（configs/pretrain.yaml 引用）
│   ├── merged_sample.txt   # 小样本调试用
│   ├── merged_8k.txt       # r55 用的 8000 行子集
│   ├── merged_24k.txt      # r56 用的 24000 行子集
│   ├── merged_72k.txt      # r57 用的 72000 行子集
│   └── raw/                # 原始爬取语料（本地）
├── datasets/              # 原始 QA 数据（仅本地保留，未上传 git），每文件「问题行 / 答案行」交替
│   ├── ai_qa.txt
│   ├── biology_qa.txt
│   ├── ...
│   └── natural_chat.txt
└── processed/             # data_manager.py to-jsonl 生成的 jsonl（可选，本地）
```

## 准备训练语料

将 `data/datasets/` 下的多个 QA 文件合并为单一训练文本（统一入口）：

```bash
python scripts/data_manager.py merge          # 合并 datasets/ 为 data/train_data_combined.txt
python scripts/data_manager.py merge --dedup --build-vocab   # 去重并构建词表
python scripts/data_manager.py stats          # 仅查看统计
```

也可对单个目录做更细的处理：

```bash
python scripts/data_manager.py                # 交互式菜单
```

> `scripts/merge_data.py` 与 `scripts/process_data.py` 现为上述命令的兼容薄包装。

## 转换为 JSONL（可选）

```bash
python scripts/data_manager.py to-jsonl
```

把 `data/datasets/*.txt`（奇数行问题、偶数行答案）转换为
`data/processed/*.jsonl`，便于其它工具消费。

## 对话 / 陈述转 QA

```bash
python scripts/convert_dialogue_to_qa.py
```

## 词表

词表在训练时由 `data.train_file` 自动构建，存于 `<checkpoint_dir>/vocab.json`
（默认 `configs/pretrain.yaml` 即 `checkpoints/vocab.json`）。当前统一使用
`CharTokenizer`（字符级、零 OOV，相邻字合并交给模型侧 `CharMergeLayer`），
`load_vocab` 会按词表内容自动识别字符级 / BPE 格式。

特殊 token（`models/constants.py`，顺序即索引）：
`<pad>=0 <???>=1 <bos>=2 <eos>=3 [SEP]=4`。

## 数据相关配置（`data` 段）

| 键 | 默认 | 说明 |
|----|------|------|
| `train_file` / `test_split` | — | 语料路径与验证集比例（`test_split: 0.1`）。 |
| `pack` | `false` | 打包模式：各行首尾相接成长流，按 `max_seq_length` 切块；关闭时每行截断/补齐。 |
| `pack_max_train_chunks` / `pack_max_val_chunks` | 不限 | `pack=true` 时随机抽取的块数上限（r56/r57 用它控步数）。 |
| `tokenizer_pretok` | `1` | 预分词版本：1 旧 BPE 预分词（转小写、丢空格）；2 保留大小写/空格并转半角；3 在 2 基础上仅保留 ASCII 词间空格。写入 `vocab.json`。 |
| `vocab_train_only` | `false` | 仅用训练切分后的行建词表（val 独有字符走 byte fallback）。 |
| `long_line_windows` | `false` | 非 pack 模式下长行按步长切多个窗口（不再只用开头），并按来源行切分 train/val。 |

如需单独诊断词表编码：

```bash
python tools/diagnose_vocab.py
```
