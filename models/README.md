# models/

模型定义与基础设施。

| 文件 | 说明 |
|------|------|
| `transformer.py` | `TransformerModel`（纯注意力）与 `MambaSSM`（SSM 混合）主干。架构为 Pre-LN + RMSNorm + RoPE + SwiGLU；`generate()` 支持 temperature / top-k / 重复惩罚 / 特殊 token 屏蔽，并带 KV-cache。RoPE 的 cos/sin 在模块级缓存（`_ROPE_CACHE`），按 `(device, head_dim)` 缓存整张位置表并切片，训练时跨层、生成时逐 token 均命中同一张表（避免原实现每步重算）。`MambaSSM` 的选择性扫描已用**并行前缀扫描**向量化（log2(L) 步、数值稳定），消除逐时间步 for 循环：大幅加快 CPU 训练，并避免低功耗 iGPU 上 DML 因单步 kernel 过多触发 TDR 设备重置。 |
| `data_utils.py` | `Vocabulary`（BPE/字级词表构建与编解码）、`TextDataset`（预编码并缓存 padded tensor，避免每 batch 重新 tokenize）。 |
| `config_loader.py` | `load_config` / `build_model` / `load_vocab`：统一从 YAML 构建与加载模型，避免各处硬编码结构。 |
| `device.py` | `get_device('auto')` 自动选择 cuda / dml(AMD DirectML) / cpu；也支持显式 `device: 'dml'`。`apply_cpu_threads` 限制 CPU 线程数。已移除无用的 `supports_amp`。 |

> 修改模型结构（层数 / 维度 / 词表大小）只需改 `configs/*.yaml`，所有脚本自动同步。
