# 聊天端审查附件（2026-10-03，第八批审后）

只读实验脚本和一个待应用的补丁，不改任何模型代码。基于 `ceec9d9`。在 CPU（torch 2.4.1）上跑过，DML 上的数字待主代理补。

| 文件 | 回答什么 | CPU 结果 |
|---|---|---|
| `gdn_orientation_check.py` | 现有 GatedDeltaNet 默认循环（`mixers.py:1318-1335`）是不是 delta rule | 不是。写 (k1,v1) 再用 q=k1 读，标准形式取回 v1，仓库取回 (k1·v1)·k1；同一 key 写两次也覆盖不掉旧值 |
| `gdn_chunk_proto.py` | 标准 gated delta rule 能不能分块并行、误差多大、快多少 | 块长 16：三种门值下对 fp64 最大误差 ≤2.7e-5（循环 fp32 为 ≤5.5e-6）；前向+反向算子 7686→876（−89%），CPU 163→37 ms。块长 32/64 在长记忆下数值发散，不能用 |
| `gdn_loop_v3.patch` | 现有循环再外提一轮（α/β 与 kf 行向量 unbind、除法挪出循环、删掉没用的 z_l） | 不改数值 |
| `gdn_loop_v3_check.py` | 打补丁前后逐位对拍 + 算子数 + 耗时；`--base` 指定改前提交 | 对 v2（HEAD）：前向、输入梯度、全部参数梯度 torch.equal，GDN 模块前向+反向算子 7305→5277（−27.8%），CPU 129→116 ms。对 v1（`--base 82c44f8`）：同样逐位相等，算子 12061→5277（−56.2%），CPU 234→114 ms。全量测试和改前一致 |

用法：

```
git apply experiments/claude_review_20261003/gdn_loop_v3.patch
python experiments/claude_review_20261003/gdn_loop_v3_check.py --device dml
python experiments/claude_review_20261003/gdn_loop_v3_check.py --base 82c44f8 --device dml
python experiments/claude_review_20261003/gdn_chunk_proto.py --device dml
python experiments/claude_review_20261003/gdn_orientation_check.py
```

在这个环境里跑全量测试（没有 DML、没有本地数据和权重）：1063 passed / 32 skipped / 2 xfailed / 2 failed。两个失败都是测试读了被 git 忽略的本地文件：

- `tests/test_new_mechanisms.py::test_ngram_fusion_save_load_preserves_gate` 读 `data/pretrain_corpus/_ngram_smoke.txt`
- `tests/test_review_batch2.py::test_chat_has_repetition_penalty_and_valid_default_model` 读 `checkpoints_train_8k_r42/final_model.pt`
