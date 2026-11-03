# configs/

所有 YAML 训练 / 推理配置。`pretrain.yaml` 为规范默认；其余为变体。

| 文件 | 说明 |
|------|------|
| `pretrain.yaml` | 规范默认：中文+英文混合语料，词表 8000，6 层，单轮遍历。CPU / CUDA 均可跑。 |
| `config_full_dml.yaml` | AMD DirectML（780M iGPU）训练专用：DML 扫描最优质量档参数，`gradient_checkpointing` 已关闭，6 层模型显存充裕。 |
| `config_hybrid.yaml` | SSM×注意力混合架构配置。CUDA / CPU / AMD DirectML 均可训练（DML 上选择性扫描已向量化）。 |
| `config_train_8k.yaml` | 8000 行样本训练配置（4 层 256 维，12 项高价值特性全开，R41 训练 SLA 217s）。无 Controller 的旧主力；带 Controller 的主线见下面 r42 起。 |
| `config_train_8k_pack.yaml` | `config_train_8k.yaml` + `data.pack`（打包语料，900 步 × bs24）。 |
| `config_train_8k_r42.yaml` · `_r43` · `_r44` | Controller/Generator 双模型逐轮实验（r42 双模型、r43 打包 + H1/H2 因果化 + chunk GDN、r44 H3 + VRC 安全造核）。Controller 每步都开，保留原样以便复现。 |
| `config_train_8k_r45.yaml` | r44 + Controller 按步 1:1 交替（`training.enhancement_schedule: [{controller: true}, {controller: false}]`）。r45 是后续所有对照的基线（`scripts/eval_pipeline.py` 里的 `BASE`）。 |
| `config_train_8k_r45s43` · `_r45s44` | r45 的 `init_seed=43/44` 重复实验。 |
| `config_train_8k_r46.yaml` | r45 的比例对照：Controller 只开 1/3（`[开, 关, 关]`）。生成盲评输给 r45（r46 胜 19、r45 胜 36、平 77）。 |
| `config_train_8k_r47.yaml` | Controller × QK-Norm 2×2 交替消融。 |
| `config_train_8k_r47s43` · `_r47s44` | r47 的种子重复实验。 |
| `config_train_8k_r48.yaml` | Controller × 可学习注意力温度 2×2。 |
| `config_train_8k_r49.yaml` | Controller × 门控残差（residual_gate）2×2。 |
| `config_train_8k_r50.yaml` | Controller × value-relative coding 2×2。 |
| `config_train_8k_r51.yaml` | Controller × QK-Norm × char_merge 2×2×2（8 段）。 |
| `config_train_8k_r52.yaml` | controller_film × QK-Norm 2×2。 |
| `config_train_8k_r53.yaml` | Controller × QK-Norm + `data.tokenizer_pretok: 2`（预分词缓存版本 2）。 |
| `config_train_8k_r54.yaml` | 同 r53，`tokenizer_pretok: 3`。 |
| `config_train_8k_r54s43` · `_r54s44` | r54 的种子重复实验。 |
| `config_train_8k_r55.yaml` | 8k 语料、3 epoch，词表 12000 / 4 层，沿用 r47 的 4 段 Controller×QK-Norm 调度，不含 `tokenizer_pretok`。 |
| `config_train_8k_r56.yaml` | 语料扩到 `merged_24k.txt`（64800 train chunks），1 epoch，其余同 r55。 |
| `config_train_8k_r57.yaml` | **最新一轮**：语料扩到 `merged_72k.txt`（194400 train chunks），1 epoch，其余同 r56。 |
| `config_train_full.yaml` | 全量语料训练（`merged.txt` 3.65M 行，1 epoch，含 step 级 checkpoint 与断点续训）。 |
| `config_cmp_enh_full.yaml` · `config_cmp_sel_full.yaml` · `config_cmp_selv2_full.yaml` | 增强 vs 基线 受控对比（全量 `merged.txt`）：常开 ENH / 旧 8 段 SEL / SELv2（全开+全关极端）。手动顺序训练三模型即可复现对比。 |

> **Controller 调度（2026-10-06 起）**：开 `controller` 的新配置从 r45 抄起，用 `training.enhancement_schedule` 按步切 `controller` 键（其余键缺省补 True）；`validate` 仍强制全开。依据：r44（一直开）vs r45（交替）生成盲评 132 对，交替胜 43、一直开胜 23、平 66；r44 关掉 Controller 后 packval 崩到 8.64，r45 关掉仍有 5.04；交替训练还快 16%。r47 起 schedule 同时承载 `qk_norm` / `attn_temp` / `residual_gate` / `value_relative_coding` / `char_merge` / `controller_film` 等消融键。代码默认仍是不调度（`enhancement_schedule=None`），r42–r44 照旧可复现。
>
> 修改模型结构只改这里，所有脚本自动同步。
>
> 历史冒烟/实验配置（`config_smoke_*`、`_exp_*`）已归档至 `archive_unused/`（本地保留，不入库）。
