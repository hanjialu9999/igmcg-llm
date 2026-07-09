# 数据使用说明 (Data Usage)

## 目录结构

```
data/
├── train_data_final.txt   # 主训练语料（config.yaml 默认引用）
├── datasets/              # 原始 QA 数据，每个文件为「问题行 / 答案行」交替
│   ├── ai_qa.txt
│   ├── biology_qa.txt
│   ├── ...
│   └── natural_chat.txt
└── processed/             # process_data.py 生成的 jsonl（可选）
```

## 准备训练语料

将 `data/datasets/` 下的多个 QA 文件合并为单一训练文本：

```bash
python scripts/data/prepare_training.py
```

也可对单个目录做更细的处理：

```bash
python scripts/data/merge_data.py
python scripts/data/merge_data_clean.py
python scripts/data/improve_data.py
python scripts/data/reformat_data.py
```

## 转换为 JSONL（可选）

```bash
python scripts/data/process_data.py
```

把 `data/datasets/*.txt`（奇数行问题、偶数行答案）转换为
`data/processed/*.jsonl`，便于其它工具消费。

## 对话 / 陈述转 QA

```bash
python scripts/data/convert_dialogue_to_qa.py
python scripts/data/convert_statements_to_qa.py
```

## 词表

词表在训练时由 `data/train_file` 自动构建，存于 `checkpoints/vocab.json`。
特殊 token：`<pad>=0 <unk>=1 <bos>=2 <eos>=3 [SEP]=4`。

如需单独诊断词表编码：

```bash
python tools/diagnose_vocab.py
```
