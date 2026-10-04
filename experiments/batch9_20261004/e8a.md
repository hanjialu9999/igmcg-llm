# e8a：11 组（x2_off_nohook 未跑，故 11 不是 12），CPU，direction 关 / 缓冲开 / safe_pow 开，24 序列×64=1536 token
# 评估集：x1/y1 用 seed42 config 的 val；x2/y2 用 seed43 config 的 val（是它们自己的训练 val，best val 锚点在此集上）
# hook 列：x1 用 hook on（模拟 X 训练态 v 恒 0）+ nohook 对照；y1/y2 一律 hook off

| 组 | 权重 | 设备 | 评估集 seed | 序列数 | 开关 | tf | prefix | 逐字 | tf−prefix | prefix−逐字 |
|---|---|---|---|---|---|---|---|---|---|---|
| x1_on | x1 | cpu | 42 | 24 | controller on / direction off / 缓冲 on | 5.136984 | 5.167380 | 5.166105 | -0.030396 | +0.001275 |
| x1_off | x1 | cpu | 42 | 24 | controller off / direction off / 缓冲 on | 6.318251 | 6.318251 | 6.318251 | +0.000000 | +0.000000 |
| x2_on | x2 | cpu | 43 | 24 | controller on / direction off / 缓冲 on | 5.379028 | 5.402990 | 5.403447 | -0.023962 | -0.000457 |
| x2_off | x2 | cpu | 43 | 24 | controller off / direction off / 缓冲 on | 8.006469 | 8.006469 | 8.006469 | +0.000000 | +0.000000 |
| y1_on | y1 | cpu | 42 | 24 | controller on / direction off / 缓冲 on | 5.084853 | 5.100501 | 5.099901 | -0.015648 | +0.000600 |
| y1_off | y1 | cpu | 42 | 24 | controller off / direction off / 缓冲 on | 6.800645 | 6.800644 | 6.800644 | +0.000001 | +0.000000 |
| y2_on | y2 | cpu | 43 | 24 | controller on / direction off / 缓冲 on | 5.259323 | 5.271935 | 5.271651 | -0.012612 | +0.000284 |
| y2_off | y2 | cpu | 43 | 24 | controller off / direction off / 缓冲 on | 6.998473 | 6.998472 | 6.998472 | +0.000001 | +0.000000 |
| x1_on_nohook | x1 | cpu | 42 | 24 | controller on / direction off / 缓冲 on | 5.136354 | 5.166763 | 5.165598 | -0.030409 | +0.001165 |
| x1_off_nohook | x1 | cpu | 42 | 24 | controller off / direction off / 缓冲 on | 6.298400 | 6.298401 | 6.298401 | -0.000001 | +0.000000 |
| x2_on_nohook | x2 | cpu | 43 | 24 | controller on / direction off / 缓冲 on | 5.379117 | 5.402869 | 5.403309 | -0.023752 | -0.000440 |
