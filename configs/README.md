# configs/

所有 YAML 训练 / 推理配置。`pretrain.yaml` 为规范默认；其余为变体。

| 文件 | 说明 |
|------|------|
| `pretrain.yaml` | 规范默认：中文+英文混合语料，词表 12000，6 层，单轮遍历。CPU / CUDA 均可跑。 |
| `config_dml_full.yaml` · `config_dml_test.yaml` | AMD DirectML（780M iGPU）训练配置。`gradient_checkpointing` 已关闭（6 层模型显存够用）。SSM/hybrid 现也可在 DML 上训练（选择性扫描已向量化，消除了曾触发 iGPU 设备重置的 kernel 风暴）。 |
| `config_hybrid.yaml` · `config_hybrid_full_cpu.yaml` · `config_hybrid_full_dml.yaml` · `config_hybrid_small.yaml` | SSM×注意力混合架构配置。CUDA / CPU / AMD DirectML 均可训练（DML 上选择性扫描已向量化，不再触发设备重置）。 |
| `config_pretrain_cpu.yaml` | CPU 预训练配置。 |
| `config_test.yaml` | 小批量冒烟测试配置（2 epoch，便于快速验证流程）。 |
| `config_smoke_4k.yaml` · `config_smoke_8k.yaml` | 小数据冒烟训练配置（4000 / 8000 行 × 1 epoch，关防过拟合），用于快速验证训练流程与对比生成速度（IGMCG 开 / 关）。 |

> 修改模型结构只改这里，所有脚本自动同步。
>
> **可配置架构增强（实验性，默认关）**：`model` 段支持 `qk_norm` / `attn_temp`（注意力 QK-Norm 与可学习每层温度）、`residual_gate`（门控残差）、`hybrid_gate`（hybrid 块内 attn/ssm 两路门控）。开启后需重新训练才生效，旧权重仍可加载（门控默认 1.0 ≈ 普通残差）。
