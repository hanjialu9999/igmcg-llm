# configs/

所有 YAML 训练 / 推理配置。`pretrain.yaml` 为规范默认；其余为变体。

| 文件 | 说明 |
|------|------|
| `pretrain.yaml` | 规范默认：中文+英文混合语料，词表 8000，6 层，单轮遍历。CPU / CUDA 均可跑。 |
| `config_full_dml.yaml` | AMD DirectML（780M iGPU）训练专用：DML 扫描最优质量档参数，`gradient_checkpointing` 已关闭，6 层模型显存充裕。 |
| `config_hybrid.yaml` | SSM×注意力混合架构配置。CUDA / CPU / AMD DirectML 均可训练（DML 上选择性扫描已向量化）。 |
| `config_train_8k.yaml` | 8000 行样本训练配置（4 层 256 维，12 项高价值特性全开，R41 训练 SLA 217s）。当前主力训练配置。 |
| `config_train_full.yaml` | 全量语料训练（`merged.txt` 3.65M 行，1 epoch，含 step 级 checkpoint 与断点续训）。 |
| `config_cmp_enh_full.yaml` · `config_cmp_sel_full.yaml` · `config_cmp_selv2_full.yaml` | 增强 vs 基线 受控对比（全量 `merged.txt`）：常开 ENH / 旧 8 段 SEL / SELv2（全开+全关极端）。手动顺序训练三模型即可复现对比。 |

> 修改模型结构只改这里，所有脚本自动同步。
>
> 历史冒烟/实验配置（`config_smoke_*`、`_exp_*`）已归档至 `archive_unused/`（本地保留，不入库）。
