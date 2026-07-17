# 数据使用说明 (Data Usage)

## 目录结构

```
data/
├── pretrain_corpus/        # 主训练语料（merged.txt / merged_sample.txt，本地不入库）
│   ├── merged.txt          # 默认训练语料（configs/pretrain.yaml 引用）
│   ├── merged_sample.txt   # 小样本调试用
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
python scripts/improve_data.py
python scripts/reformat_data.py
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
python scripts/convert_statements_to_qa.py
```

## 词表

词表在训练时由 `data/train_file` 自动构建，存于 `checkpoints/vocab.json`。
特殊 token：`<pad>=0 <unk>=1 <bos>=2 <eos>=3 [SEP]=4`。

如需单独诊断词表编码：

```bash
python tools/diagnose_vocab.py
```
