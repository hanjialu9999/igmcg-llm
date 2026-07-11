# tools/

检查、诊断与监控工具（非主流程）。

| 文件 | 说明 |
|------|------|
| `view_model.py` | 查看模型结构与参数量。 |
| `compare_epochs.py` | 对比同一模型不同 epoch 的输出差异。 |
| `diagnose_vocab.py` | 词表编码 / 解码诊断。 |
| `dialogue.py` | 对话样例生成。 |
| `quick_demo.py` | 快速生成 demo。 |
| `quick_test.py` | 快速冒烟测试。 |
| `check_ckpt.py` · `check_files.py` · `check_training.py` · `check_vocab.py` | 各类一致性 / 完整性校验。 |
| `training_plan.py` · `training_report.py` | 训练计划与报告生成。 |

## 子目录

- `monitor/`：训练过程与显存监控（见 `tools/monitor/README.md`）。
