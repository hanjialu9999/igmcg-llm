# 第九批【三】v3 patch 报告（2026-10-04，单独 commit）

patch：`experiments/claude_review_20261003/gdn_loop_v3.patch`
（只改默认循环 `models/mixers.py` S1 路线；loop32 别处只读未动）。
内容：α/β 与 kf 行向量 unbind 成视图列表（循环内不再逐步 select/unsqueeze）；
除以 den 挪到循环外整体一次（逐元素除法）；去 `z_l` 未用 unbind。
`git apply --check` 通过；apply 后 diff 仅 `models/mixers.py`（11+/16−）。

## check ×3（`gdn_loop_v3_check.py`，base=HEAD）

三次一致：
前向 torch.equal=True，max|Δ|=0.0；
输入梯度 torch.equal=True；
参数梯度全部 torch.equal=True，最大差 0.0；
前向+反向算子数 改前 7305 → 改后 5277（−27.8%）。
（check 自带 CPU ms：改前 145.2/148.4/116.5，改后 130.7/131.5/133.3——CPU 噪声大，
仅记录，不作为结论。）

## 每步毫秒（v2 同一命令同一 config，DML，3 次）

命令：`.amd_venv python scripts/_profile_train_steps.py --steps 50 --controller on
--config configs/config_train_8k_r42.yaml --out experiments/batch9_20261004/r42_profile_50steps_v3_run{1,2,3}.json`
（`--out` 写新文件；默认会覆盖 `baselines/r42_profile_50steps.json`，本次未碰。）

| run | pass1（有仪器）ms/步 | pass2（无仪器）ms/步 |
|---|---|---|
| 1 | 560.498 | 415.994 |
| 2 | 482.562 | 469.538 |
| 3 | 460.432 | 414.091 |
| 中位（范围） | 482.562（460.4~560.5） | 415.994（414.1~469.5） |

v2（CHANGELOG ceec9d9）：pass1 口径 479.473。
v3 pass1 中位 482.562 vs v2 479.473 → +0.6%，在波动内，持平。
（run1 560.5 为首轮 DML 预热偏高，中位数已吸收。）

## r42 CPU 24 序列三口径（改后 vs 改前逐位）

改后（v3 已 apply，`experiments/batch9_20261004/r42_24_v3after.json`，
b8_run.py 同口径：ctrl 开/dir 关/mem 开/缓冲开/pow 开/hook 关，CPU，seed42 val 前 24 条）：
tf 5.245063 / prefix 5.248623 / incremental 5.248557。
改前（`baselines/r42_seq24_ctrlon_diroff_cmb1.json`，2026-09-30，v2 前测）：
5.245063/5.248623/5.248557。逐位相等 ✅。

## 全量 pytest（v3 已 apply，`.amd_venv`）

1095 passed / 2 skipped / 2 xfailed（41.94s），与 v2 基线逐项一致 ✅。

结论：数值逐位不变（check torch.equal + r42 三口径逐位），算子 −27.8%，
DML 每步与 v2 持平（+0.6%），全量绿。只改了 mixers.py 一处，未碰 r42 config。
