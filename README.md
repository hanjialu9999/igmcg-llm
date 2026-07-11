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
├── train_finetune.py          # 微调入口（在预训练底座上做 QA 微调）
├── _dl_c4.py / _run_train.py  # 辅助脚本：下载中文语料 / 后台拉起训练
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

依赖：`torch`、`pyyaml`、`numpy`、`tqdm`。本项目**自动适配不同硬件**，无需改代码：

- **NVIDIA 显卡** → 自动使用 CUDA
- **AMD / Intel 核显或独显（Windows）** → 安装 `torch-directml` 后自动通过 DirectML 后端使用
  （`pip install torch-directml`，需配合 Python 3.10/3.11 + torch 2.0/2.1）
- **无 GPU** → 自动退回 CPU

设备选择由 `models/device.py` 的 `get_device()` 统一处理，配置项 `config.yaml` 的
`device: "auto"` 即为自动探测（也可显式写 `"cuda"` / `"cpu"` / `"dml"`）。

## 专用环境（AMD / Intel 核显，Windows）

本机若只有核显（如 AMD Radeon 780M），推荐单独建一个 Python 3.11 虚拟环境来启用 DirectML
（torch-directml 不支持过新的 Python）：

```bash
py -3.11 -m venv .amd_venv
.amd_venv\Scripts\activate
pip install -r requirements-amd.txt
```

之后用该环境运行训练/生成，`device: "auto"` 会自动探测到 DirectML 并在核显上计算。

> 说明：torch-directml 在**训练（train 模式）**下主体算子（注意力 / 线性层）都在核显上跑，
> 仅优化器的个别小算子会回退 CPU，因此训练可获加速；**生成（eval 模式）**时 Transformer
> 编码层的融合算子 DirectML 暂不支持、会回退 CPU，但功能正常、不影响结果。这是后端限制，
> 非代码问题。如追求生成也在 GPU 上跑，建议使用 ROCm（Linux）环境。

## 怎么运行（不用记虚拟环境）

项目根目录提供了 **`run.bat`** 启动器，会自动选好对应的虚拟环境，直接双击或用命令行：

```bat
run.bat            REM 交互式对话（chat）
run.bat train      REM 训练基座模型（config/pretrain.yaml，产出 checkpoints/final_model.pt + vocab.json）
run.bat finetune   REM 微调（产出 best_finetuned_model.pt）
run.bat chat       REM 对话
run.bat gen "你的问题"   REM 单条生成
```

> 手动跑也行：先 `.amd_venv\Scripts\activate`（AMD 核显）或 `.my_venv\Scripts\activate`（CUDA/CPU），
> 再 `python scripts/train.py --config config/pretrain.yaml`。

## 文档导航（只看本文件即可，其它是分主题细节）

- 上手最快：**[QUICK_START.md](QUICK_START.md)**
- 分主题细节在 **`docs/`**：训练 `docs/TRAINING_GUIDE.md`、调参 `docs/TUNING_GUIDE.md`、
  数据 `docs/DATA_USAGE.md`、模型用法 `docs/MODEL_USAGE_GUIDE.md`

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
