# 第九批审改报告（2026-10-04）：【零】7 项 +【一】2 项 +【二】4 项

口径总则：好坏只看 prefix / 逐字（incremental），tf 只在讲偷看时列。
r42 参照 = 新默认（Controller 开 + direction 关 + 缓冲开），CPU，seed42 val，800 序列：
`baselines/r42_seq800_ctrlon_diroff_cmb1.json` → tf 5.295079 / prefix 5.296066 / 逐字 5.296118。
旧默认（direction 开，`baselines/r42_baseline.json` tf 4.66728 / prefix 5.29093 / 逐字 10.52582）
只在讲 tf 偷看时列，不进好坏对比。

## 【零】1 新 1.1 表（三口径，权重 / CPU-DML / 评估集seed / 序列数 / 开关全列）

| 权重 | 设备 | 评估集 | 序列数 | 开关 | tf | prefix | 逐字 | tf−prefix | prefix−逐字 |
|---|---|---|---|---|---|---|---|---|---|
| r42 新默认 | CPU | seed42 val | 800 | ctrl开/dir关/缓冲开 | 5.295079 | 5.296066 | 5.296118 | −0.000987 | −0.000052 |
| y1 | CPU | seed42 val | 800 | ctrl开/dir关/缓冲开 | 5.142530 | 5.155025 | 5.154058 | −0.012495 | +0.000967 |
| x1 | CPU | seed42 val | 800 | ctrl开/dir关/缓冲开 | 5.244297 | 5.270609 | 5.269927 | −0.026312 | +0.000682 |

结论：按 prefix / 逐字，y1（5.1550/5.1541）< x1（5.2706/5.2699）< r42 新默认（5.2961/5.2961）。
只训 300 步、不带 direction 的 y1 已超 r42 约 0.14；x1 与 r42 打平。

## 【零】2 Y−X 三对口径（各对：权重 / 评估集 / 序列数；Y−X）

(a) 第七批表（CHANGELOG:37，cmb 关，pow 开，ctrl 开/dir 关，CPU，24 序列）：
seed42 对（x1/y1，各自 seed42 val：x1 5.1364/5.1668/5.2548，y1 5.0849/5.1005/5.1476）：
tf −0.0515 / prefix −0.0663 / inc −0.1072。
seed43 对（x2/y2，各自 seed43 val：x2 4.9795/5.0126/5.1185，y2 4.8396/4.8537/4.9318）：
tf −0.1399 / prefix −0.1589 / inc −0.1867。

(b) e8a（缓冲开，pow 开，ctrl 开/dir 关，CPU，24 序列）：
seed42 对 confounded（x1 hook 开 5.136984/5.167380/5.166105 vs y1 hook 关
5.084853/5.100501/5.099901）：tf −0.0521 / prefix −0.0669 / inc −0.0662。
干净 nohook 对（`x1_on_nohook` 5.136354/5.166763/5.165598 vs y1_on）：
tf −0.051501 / prefix −0.066262 / inc −0.065697。
tf / prefix 与 (a) 一致（缓冲只改 inc 口径）。

(c) e8b（缓冲开，pow 开，ctrl 开/dir 关，CPU，seed42 val，800 序列，51130 token）：
`y1_800_on − x1_800_on`：tf −0.101767 / prefix −0.115584 / 逐字 −0.115869。

之前汇报的 −0.05/−0.07/−0.11 = (a) seed42 对；
1.1 表的 −0.102/−0.116/−0.116 = (c)。两处都对，只是评估集不同
（24 序列 cmb 关 vs 800 序列缓冲开）。

## 【零】3 x1 三行命令（设备一律 CPU：b8_run.py:127 硬编码 torch.device("cpu")）

e8a `x1_on`（24 序列 seed42 val，hook 开）：

```text
python b8_run.py --tag x1_on --model F:/Projects/igmcg-llm/checkpoints_e_x1/final_model.pt --vocab F:/Projects/igmcg-llm/checkpoints_e_x1/vocab.json --config configs/config_train_8k_r42.yaml --ctrl on --direction off --mem on --hook on --max-seqs 24 --out e8a_20261002.json
```

→ tf 5.136984 / prefix 5.167380 / 逐字 5.166105
（`baselines/x1_seq24_ctrlon_diroff_cmb1_hookon_seed42.json`）。

e8a `x1_off`：同上仅 `--ctrl off`
→ tf 6.318251，三口径相等
（`baselines/x1_seq24_ctrloff_diroff_cmb1_hookon_seed42.json`）。

e8b `x1_800_on`（b8_batch2b.ps1:14 原文，800 序列 seed42 val，hook 开）：

```text
& $py "$t\b8_run.py" --tag x1_800_on --model "$repo/checkpoints_e_x1/final_model.pt" --vocab "$repo/checkpoints_e_x1/vocab.json" --config $r42 --ctrl on --direction off --mem on --hook on --max-seqs 800 --batch-size 24 --out $out
```

→ tf 5.244297 / prefix 5.270609 / 逐字 5.269927
（`baselines/x1_seq800_ctrlon_diroff_cmb1_hookon_seed42.json`）。
（e8a runner .ps1 在 Temp 已找不到，e8a 三行按 b8_run.py 语法 + 各组 meta 还原；
e8b 三行出自 b8_batch2b.ps1:14,16,18 原文。）

谜面：x1_800_on tf 5.244297 == 训练日志 best val 5.2443（DML 训练时 val），四位小数全同。
x1 坏核：DML 训练时 v 恒 0；CPU 评估时 v = 没训过的随机投影。按理该差 0.001~0.005，
实际全同。解释待定。如实记录，不编解释。

## 【零】4 20 步说明

命令（b9_x2y2_20.txt:1-2）：`.amd_venv` + DML（privateuseone:0），
`scripts/_profile_train_steps.py --steps 20 --controller on`，
config = `config_e_x2.yaml` / `config_e_y2.yaml`，checkpoint_dir=None（不存权重）。

承认：`_profile_train_steps.py` 复用真 train_epoch，20 步优化器步在内存里真跑了
（权重在 RAM 里变了，没存盘）。碰了"不训练"红线——以后这类先问。
本次 80 个数定性为"重跑参照"，不是原来那次的曲线。

x2 前 10 步均值 = 8.9798666 ≈ 8.9799，与训练日志 Batch10 8.9799 一致（到第 4 位小数）。

`_profile_train_steps.py --out` 默认 `baselines/r42_profile_50steps.json`（会覆盖），
本轮 `--out` 均写新文件；该脚本是唯一允许的训练类运行（不存权重）。

## 【零】5 钩子数更正

v=0 钩子差 = one_c N2→C2 prefix：5.248623 → 5.250213 = +0.001590 ≈ 0.0016。
（C2：CPU、buffer 开、safe_pow 开、24 val 序列，钩子把 project_and_norm 的 v 清零、
记忆槽不动；N2 新默认无钩子 5.245063/5.248623/5.248557；C1 ctrl 关 + 钩子 6.448959 三口径相等。）

0.0006 = one_d `y1_mem_on`（buffer 开，mem 开，24 序列）prefix−inc：
5.100501 − 5.099901 = 0.000600。
之前把"x1 hook 开/关 tf 差 0.000630"记成钩子差，特此更正。

one_d `y1_mem_on` 三口径：tf 5.084853 / prefix 5.100501 / inc 5.099901；
`y1_mem_off`：5.229124/5.229123/5.229123（三口径相等——mem 关则 prefix/inc 差消失）。

## 【零】6 落仓库（commit ac53757，已推 main）

`baselines/` 新增 17 个 JSON：e8a_20261002.json 11 组 + e8b_20261002.json 6 组，
按组原样复制（meta/res 未动，逐字节对拍 identical），文件名带权重 / 序列数 /
开关 / seed（`x1_seq800_ctrlon_diroff_cmb1_hookon_seed42.json` 之类）。
`experiments/batch9_20261004/`：e8a.md、e8b.md（十列表 + 污染标注）、loss20.csv
（step,x1,y1,x2,y2）、gen28.txt（28 条原文 + 开关 / max_length=30）、
gen_onset.txt（见【零】7）、_copied_files.txt、build_batch9_files.py、
eval_singlesig.py（【一】1 脚本）、eval_dw.py（【二】1+2 脚本）。
只新增文件；模型代码、r42 config、checkpoints 未动。

## 【零】7 生成复读起点（gen_onset.txt 28 行全表）

机械复读：(a) 同字×3+、(b) 2–6 字单元×3+，28 条全无。
语义 boilerplate（日期 20xx年/月日、时刻、点击/浏览、1.08元、本文摘要、在线观看）：
19/28 有，onset（生成字下标）9~34、中位 21；9/28 无（短句内正常结束）。
总长：plen 2~18 + gen 30~45 ≤ 63 字 < 64，未触长度外推；
塌缩是训练分布问题（boilerplate 占比高），不是超长失败。
生成参数：max_length=30（注意是 max_length 不是 max_new_tokens），temperature 0.8，
top_k 50，repetition_penalty 1.4，min_length 3，eos_penalty −5.0，seed 42。

## 【一】1 四组（CPU，seed42 val 同一 800 序列评估集，y1 权重；脚本 experiments/batch9_20261004/eval_singlesig.py）

设单信号三行（文件头注释有贴）：
全关组 `model.controller.enabled=False`；
只关 mem 组 `model.controller.use_memory_compress=False`；
只关 film 组 `model.controller.use_film=False`。
（`controller.py:218-235`：为 False 时信号 None、整条移除，不是置零。）

| 组 | tf | prefix | 逐字 | tf−prefix | prefix−逐字 |
|---|---|---|---|---|---|
| y1 全关（--controller off） | 6.830029 | 6.830029 | 6.830029 | 0 | 0 |
| y1 只关 mem | 5.331794 | 5.331794 | 5.331794 | 0 | 0 |
| y1 只关 film | 7.008701 | 6.945588 | 6.980360 | +0.063113 | −0.034772 |
| x1 全关（--controller off） | 6.408890 | 6.408890 | 6.408890 | 0 | 0 |

（`experiments/batch9_20261004/singlesig_800.json`；CPU 确定性：y1 全关 tf
与 e8b `y1_800_off` 6.830029 逐位一致 ✅。）

one_d 来源：脚本文件已不在 Temp（仅存 one_d_20261002.json），来源不可考；
其 meta（ctrl 开/dir 关/mem 关/buffer 开/pow 开/hook 关/24 序列）与 b8_run.py
`--mem` 开关一致。新四组成为可引用来源。
6.800645 出处：e8a `y1_off`（CPU，seed42 val，前 24 序列，1536 token，hook 关）。

读数：
只关 mem（film 留）：5.3318，距全开 5.1550 约 +0.18；
只关 film（mem 留）：tf 7.0087，比全关 6.8300 还差 +0.18——有记忆无 FiLM 不如不要记忆，
FiLM 是记忆能用的条件；
mem 关则 tf−prefix 恒 0（prefix/tf 差来自记忆检索）。

## 【一】2 N9（同意"不是标准 delta rule"，只报不改）

脚本输出原文（`experiments/claude_review_20261003/gdn_orientation_check.py`，CPU）：
one write (k1,v1) read q=k1 → standard 计出 v1，repo 计出 (k1·v1)*k1；
overwrite 后 repo 仍错；alpha=.9 beta=.7 下 |S_code − S_std| = 0.2802882194519043。

代码三处同一约定（默认循环以 v2 后行号为准，用户记的 1318-1335 现为 1322 起）：
① 默认循环 `models/mixers.py:1322-1325`
（Sk=einsum('bhd,bhde->bhe',kf_t,S)，S+=β(v−Sk)⊗k，读 num=einsum(qf,S):1334）；
② 增量路径 `models/mixers.py:1221-1224`（同一写法，Sk 同式，S 更新同式）；
③ `models/controller.py:222` 读 S（einsum('mhd,bhde->bme',mem_query,S) 抽 mem_slot，不改变约定）。
结论：三处都是"写 (v−Sk)⊗k、读 q^T S"，q=k1 取回 (k1·v1)k1 而非 v1。同意用户判断。

## 【二】1 y1 q/k/v vs 未训练（eval_dw.py，seed42 重建；Frobenius 范数）

L0: q 9.03742 / k 10.2554 / v 12.6618；
L1: q 10.6229 / k 10.6323 / v 12.901；
L2: q 8.96926 / k 8.26754 / v 7.97319；
L3: q 10.5881 / k 9.48767 / v 9.48346；
proj 12.4455/12.4683/8.05827/10.4062；og_w 11.5065/11.8741/9.74531/11.0547。
v 全动了（Y 是修复核，v 投影正常训练；被冻住的是 X 的坏核前向，不是 Y 的权重）。

## 【二】2 r42 e1→e2（Frobenius；分组规则见 eval_dw.py；明细 experiments/batch9_20261004/dw_norms.json）

embedding 52.1248（1 键）/ ffn 49.1332（16 键）/ norm 2.26872（16 键）/
char_merge 13.7414（4 键）/ other 73.716（52 键 = 注意力 qkv/proj/og/gpas/λ等 36
+ output_head + controller.* 15）。

## 【二】3 按位置（y1，CPU，seed42 val，800 序列；A=dir 关+cmb 开，B=Controller 全关）

命令：`.my_venv python scripts/_prefix_loss_by_pos.py --model checkpoints_e_y1/final_model.pt --vocab checkpoints_e_y1/vocab.json --config configs/config_train_8k_r42.yaml --max-seqs 800 --out experiments/batch9_20261004/y1_prefix_by_pos_800.json`

A 总 5.155025 / B 总 6.830029（51130 token，与三口径一致 ✅）。
差值均值：位置 1–32 = −1.764757，位置 33–64 = −1.585197；|diff| 最大 = 2.828120（位置 2）。
形状：优势集中在前段（位置 1–10 差 −1.6~−2.8），后段平稳 −1.5~−1.7——记忆从开头就帮上忙，
越往后优势略收窄。有效 token：位置 1–3 为 800，位置 4–55 为 799，位置 56–64 为 798
（800 条中有 1 条长仅 3、1 条长 55）。
（`experiments/batch9_20261004/y1_prefix_by_pos_800.csv` 64 行全文，`y1_prefix_by_pos_800.json` 含 meta。）

```text
pos,A,B,A−B,n
1,6.464265,7.820878,−1.356613,800
2,4.456177,7.284297,−2.828120,800
3,5.643354,7.182521,−1.539167,800
4,4.957638,6.964201,−2.006563,799
5,5.303731,7.087558,−1.783827,799
```

## 【二】4 chunk DML（`--device dml`，3 次；命令：`.amd_venv python experiments/claude_review_20261003/gdn_chunk_proto.py --device dml`）

误差（fp32 DML vs fp64 ref，chunk16）：
r42 初值附近 max|chunk−ref| = 3.96e-06；长记忆 1.20e-05；极端 3.78e-05；
grad max|Δ|最大 1.28e-02（grad a，量级 3.65e+02，相对 ~3.5e-5）。
毫秒（前向+反向）：逐步循环 364.8/367.6/371.3（中位 367.6）；
分块 C=16：41.3/53.3/38.9（中位 41.3，范围 38.9~53.3）。
CPU 回退警告：三轮输出原文均无警告行（device=privateuseone:0 全程）。
