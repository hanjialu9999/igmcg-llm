# tools/

检查、诊断与监控工具（非主流程）。

| 文件 | 说明 |
|------|------|
| `view_model.py` | 查看模型结构与参数量。 |
| `compare_epochs.py` | 对比同一模型不同 epoch 的输出差异。 |
| `diagnose_vocab.py` | 词表编码 / 解码诊断。 |
| `dialogue_interactive.py` | 交互式对话（从 `chat_config.json` 读取生成参数）。 |
| `quick_demo.py` | 快速生成 demo。 |
| `check_training.py` | 训练一致性 / 完整性校验。 |

## 子目录

- `monitor/`：训练过程与显存监控（见 `tools/monitor/README.md`）。
