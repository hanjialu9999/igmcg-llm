# models/

模型定义与基础设施。

| 文件 | 说明 |
|------|------|
| `transformer.py` | `TransformerModel`（纯注意力）与 SSM 混合主干、`TransformerBlock`、`CrossLayerRouter`。Pre-LN + RMSNorm + RoPE + SwiGLU；`generate()` 支持 temperature / top-k / top-p / 重复惩罚 / 特殊 token 屏蔽，带 KV-cache；Controller、n-gram 融合与 IGMCG 门控均在此接线。RoPE 的 cos/sin 实例级缓存（线程安全），按 `(device, head_dim)` 缓存整张位置表并切片。注意力可选 **QK-Norm** 与**可学习每层温度**；block 可选**门控残差** / **混合路径门控**（hybrid 块内 attn/ssm 两路各自门控）、LayerFilm / input highway / progressive residual / layer skip / MoE / early exit / cross-layer routing，均经 `config['model']` 开关控制（加载无该键的旧权重时默认关，向后兼容）。 |
| `controller.py` | R42 Controller/Generator 双模型编排的 Controller：GatedDeltaNet 产出压缩记忆 `mem_kv`、FiLM、direction 三类控制信号，经条件化总线注入 Generator；三信号零初始化即中性。支持逐位置因果 direction、因果记忆与 chunk 扫描。 |
| `mixers.py` | 各类 token mixer：`SlidingWindowCausalSelfAttention`、`LinearAttention`、`GatedDeltaNet`、`AxialLinearAttention`、`DifferentialAttention`、`MambaSSM`（并行前缀扫描）及 `SwiGLU`；`EnhancementsMixin` 统一实现 QK-Norm、注意力温度、门控残差、VRC、char-merge 等增强。 |
| `layers.py` | `CharMergeLayer` 等独立层实现（字符级卷积合并）。 |
| `norms.py` | `RMSNorm`、`GPASNorm`。 |
| `rope.py` | `RotaryEmbedding`，含 YaRN 长上下文外推参数。 |
| `state.py` | `BlockState`：逐层 KV / memory / Controller 运行时状态容器。 |
| `sampling.py` | `apply_repetition_penalty` / `sample_next_token` / `_decode_one_step`：top-k、top-p、temperature 采样与增量解码。 |
| `ngram.py` | `NGramModel`：1..N 阶统计 n-gram 先验（固定插值 CLI 路径 + 可学 `order_logits` 融合路径）。 |
| `memory.py` | `MemoryBank`：可学习压缩记忆（`memory_size>0` 时启用）。 |
| `moe.py` | MoE 前馈层（`moe` 开关）。 |
| `gates.py` | 统一门控抽象（direct / sigmoid 标量 / 向量等 5 种模式），`gate=None` 恒等，不改变 state_dict。 |
| `qat.py` | 量化感知训练（LSQ 风格可学习步长伪量化，训练时 monkey-patch Linear，eval 恒等）。 |
| `model_config.py` | `SSMConfig` / `AttnConfig` / `MemoryConfig` / `ModelConfig`：模型 schema、默认值与 `model` 段解析、互斥/缺省告警。 |
| `config_loader.py` | `load_config` / `build_model` / `load_vocab` / `load_generation_config`：统一从 YAML 构建与加载模型，避免各处硬编码结构。`load_vocab` 按词表内容自动识别 BPE / 字符级 tokenizer。 |
| `data_utils.py` | `BaseTokenizer` / `BPETokenizer` / `CharTokenizer`（词表构建与编解码）、`TextDataset`、`load_data` / `load_packed_data`（`data.pack` 预分词打包、长行滑窗）、`create_dataloader`、`split_dataset`。 |
| `checkpoint.py` | `safe_torch_load`（安全反序列化）、`load_model`（权重+词表+设备+Controller/n-gram 重建）、`build_ngram_model`。 |
| `utils.py` | 日志、`IGMCGError` / `cli_guard`、checkpoint 保存/清理/备份、优化器参数分组等通用工具。 |
| `constants.py` | 特殊 token 索引、mask 填充值、RoPE 基频、生成默认超参（单一事实来源）。 |
| `device.py` | `get_device('auto')` 自动选择 cuda / dml(AMD DirectML) / cpu；也支持显式 `device: 'dml'`。`apply_cpu_threads` 限制 CPU 线程数。 |

> 修改模型结构（层数 / 维度 / 词表大小）只需改 `configs/*.yaml`，所有脚本自动同步。
