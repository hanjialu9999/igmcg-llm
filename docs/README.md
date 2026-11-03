# docs/

详细使用文档。

| 文件 | 说明 |
|------|------|
| `ARCHITECTURE.md` | 架构说明书：数据流、模块表、训练/生成流程、Controller 与 IGMCG、配置开关索引、已知问题编号。正文为 R42 快照，§0 汇总 R43–R57 增量；行号与实测数字对应 2026-09-29 版本。 |
| `MODEL_USAGE_GUIDE.md` | 模型结构、统一加载方式、对话 / 诊断用法。 |
| `TRAINING_GUIDE.md` | 训练流程、关键配置（`configs/pretrain.yaml`）、监控与续训。 |
| `DATA_USAGE.md` | 数据目录结构、语料准备、QA 转换、词表说明。 |
| `TUNING_GUIDE.md` | 采样参数（temperature / top_k / repetition_penalty）调参建议。 |
| `H2_CAUSALIZATION_PLAN.md` | H2 泄漏因果化改造方案（B1 逐位置前缀状态 / B2 读侧因果遮蔽 / B3 训推一致化），r43 已实现对应开关。 |

> 快速上手请看项目根目录的 `README.md`「完整快速开始」章节（原 `QUICK_START.md` 已并入）。
