# e8b：6 组，CPU，direction 关 / 缓冲开 / safe_pow 开，评估集一律 seed42 config 的 val（r42 yaml）
# ⚠ 污染：x2_seed42_on / y2_seed42_on 用 seed42 val 前 24 条，其中 20/24 落在 x2/y2 的 seed43 训练集里（VAL42=800/VAL43=800/INTER=74，SEED42_FRONT24_IN_TRAIN43=20/24），只贴不比
# x1_800_on / y1_800_on 为干净配对（seed42 val，800 序列，51130 token）

| 组 | 权重 | 设备 | 评估集 seed | 序列数 | 开关 | tf | prefix | 逐字 | tf−prefix | prefix−逐字 |
|---|---|---|---|---|---|---|---|---|---|---|
| x2_seed42_on | x2 | cpu | 42 | 24 | controller on / direction off / 缓冲 on | 4.979179 | 5.012547 | 5.012895 | -0.033368 | -0.000348 |
| y2_seed42_on | y2 | cpu | 42 | 24 | controller on / direction off / 缓冲 on | 4.839593 | 4.853706 | 4.852924 | -0.014113 | +0.000782 |
| x1_800_on | x1 | cpu | 42 | 800 | controller on / direction off / 缓冲 on | 5.244297 | 5.270609 | 5.269927 | -0.026312 | +0.000682 |
| x1_800_off | x1 | cpu | 42 | 800 | controller off / direction off / 缓冲 on | 6.431205 | 6.431205 | 6.431205 | +0.000000 | +0.000000 |
| y1_800_on | y1 | cpu | 42 | 800 | controller on / direction off / 缓冲 on | 5.142530 | 5.155025 | 5.154058 | -0.012495 | +0.000967 |
| y1_800_off | y1 | cpu | 42 | 800 | controller off / direction off / 缓冲 on | 6.830029 | 6.830029 | 6.830029 | +0.000000 | +0.000000 |
