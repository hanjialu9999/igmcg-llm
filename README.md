# 中文对话 Transformer 模型

一个从零实现的中文对话 / 文本生成 Transformer 项目：包含数据预处理、语言模型训练、微调、对话推理，以及参数调优工具链。

## 项目结构

```
新项目/
├── config/
│   └── config.yaml            # 唯一配置源：模型 / 训练 / 数据 / 生成参数
├── chat_config.json           # 对话生成参数（temperature / top_k / 惩罚等）
├── models/
│   ├── transformer.py         # TransformerModel 定义与自回归生成
│   ├── data_utils.py          # Vocabulary / 数据集 / 数据加载
│   ├── bpe_tokenizer.py       # 可选 BPE 分词器
│   └── config_loader.py       # 统一加载配置 / 构建模型 / 加载词表
├── scripts/
│   ├── train.py               # 主训练入口（LM 预训练）
│   ├── generate.py            # 单条 / 交互式生成
│   ├── finetune.py*           # 微调入口（见下）
│   ├── data/                  # 数据预处理脚本
│   │   ├── prepare_training.py
│   │   ├── process_data.py
│   │   ├── convert_dialogue_to_qa.py
│   │   ├── convert_statements_to_qa.py
│   │   ├── merge_data.py / merge_data_clean.py
│   │   ├── improve_data.py / reformat_data.py
│   │   └── analyze_datasets.py
│   └── tuning/                # 参数调优与展示
│       ├── tune_temperature.py
│       ├── tune_topk.py
│       └── showcase_optimal_params.py
├── tools/
│   ├── monitor/               # 训练过程监控脚本
│   │   ├── monitor_training.py / monitor_gpu_training.py / monitor_live.py / simple_monitor.py
│   ├── diagnose_vocab.py      # 词表编码诊断
│   ├── view_model.py          # 查看模型结构
│   ├── compare_epochs.py      # 对比不同 epoch
│   ├── quick_test.py / quick_demo.py
│   ├── training_plan.py / training_report.py
│   ├── dialogue.py
│   └── check_*.py             # 检查点 / 文件 / 训练 / 词表 自检
├── chat.py                    # 对话交互入口（使用微调模型）
├── run.py                     # 统一启动菜单（训练 / 生成 / 配置）
├── data/
│   ├── train_data_final.txt   # 主训练语料（由 config 引用）
│   ├── datasets/              # 原始 QA 数据（*.txt）
│   └── (processed/ 由 process_data.py 生成)
├── test/                      # 各类测试脚本
├── archive_unused/            # 已归档 / 冗余脚本（不再维护）
├── checkpoints/               # 训练产物（final_model.pt / vocab.json / *.pt，已 gitignore）
└── requirements.txt
```

> *`train_finetune.py` 当前位于项目根目录，作为微调入口保留；后续可移入 `scripts/` 并重命名为 `finetune.py`。

## 环境准备

```bash
python -m venv .my_venv
.my_venv\Scripts\activate
pip install -r requirements.txt
```

依赖：`torch`、`pyyaml`、`numpy`、`tqdm`。本项目在 CPU / CUDA 下均可运行（自动检测）。

## 快速开始

```bash
# 1) 训练语言模型（产出 checkpoints/final_model.pt + vocab.json）
python scripts/train.py --config config/config.yaml

# 2) 微调（基于已有模型，产出 best_finetuned_model.pt）
python train_finetune.py

# 3) 启动对话
python chat.py
```

更详细的步骤见 `QUICK_START.md`；训练 / 调参 / 数据 / 模型用法分别见
`TRAINING_GUIDE.md`、`TUNING_GUIDE.md`、`DATA_USAGE.md`、`MODEL_USAGE_GUIDE.md`。

## 配置说明

所有模型结构、训练超参、数据路径都集中在 `config/config.yaml`；对话生成参数在
`chat_config.json`。脚本通过 `models/config_loader.py` 统一读取，避免在多个文件中
硬编码（如 `vocab_size / embedding_dim` 等），改结构只需改一处。

## 注意事项

- 训练前请确保 `config.yaml` 中的 `data.train_file` 指向存在的语料文件（默认
  `data/train_data_final.txt`）。
- 词表 `checkpoints/vocab.json` 由训练脚本生成，需先训练（或构建词表）再运行
  `chat.py` / `tools/view_model.py` 等推理 / 诊断脚本。
- 大模型文件（`*.pt`）与日志已加入 `.gitignore`，不会进入版本库。
