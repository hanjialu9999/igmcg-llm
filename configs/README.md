# configs/

所有 YAML 训练 / 推理配置。`pretrain.yaml` 为规范默认；其余为变体。

| 文件 | 说明 |
|------|------|
| `pretrain.yaml` | 规范默认：中文+英文混合语料，词表 8000（注释提及 12000 更稳定但 DML 下显存更紧张），6 层，单轮遍历。CPU / CUDA 均可跑。 |
| `config_dml_full.yaml` | AMD DirectML（780M iGPU）训练专用：`gradient_checkpointing` 已关闭，6 层模型显存充裕，SSM/hybrid 也可在 DML 上训练（选择性扫描已向量化，避免单步 kernel 风暴触发 iGPU 设备重置）。 |
| `config_hybrid.yaml` | SSM×注意力混合架构配置。CUDA / CPU / AMD DirectML 均可训练（DML 上选择性扫描已向量化，不再触发设备重置）。 |
| `config_smoke_4k.yaml` | 小数据冒烟训练配置（4000 行 × 1 epoch，关防过拟合），用于快速验证训练流程与对比生成速度（IGMCG 开 / 关）。 |
| `config_cmp_enh_full.yaml` · `config_cmp_sel_full.yaml` · `config_cmp_selv2_full.yaml` | 增强 vs 基线 受控对比（全量 `merged.txt` 39700 行）：常开 ENH / 旧 8 段 SEL / SELv2（全开+全关极端）。配合 `experiments/_cmp_sel_full.py` 与根目录 `run_full_cmp.ps1` 可一键三模型顺序训练、复现 `experiments/cmp_sel_full.txt`。 |

> 修改模型结构只改这里，所有脚本自动同步。
>
