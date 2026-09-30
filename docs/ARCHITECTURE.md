# ARCHITECTURE — igmcg-llm 架构说明书（准版本）

> **本文件是架构的准版本，进 git。以后任何架构改动必须在同一个 commit 里更新本文件，并把改动贴进汇报。**
> 生成于 2026-09-29（R42 + 第三批审查后）。标注：**确定** = 有 `文件:行` 或脚本输出；**推测** = 由多条证据推出、未直接验证；**查不到** = 仓库内不存在。
> 参数量由 `Temp\opencode\e2_params.py` 在 CPU 实际构建模型数得（未加载 checkpoint、未碰显卡），总数 **6,756,101** 与 `baselines/r42_baseline.json:19` 一致。

---

## 1. 一句话定位

**基于 Transformer 的中文 LM 训练/推理项目，融合自定义混合架构（注意力 × SSM × IGMCG 直觉引导解码）与统计式 n-gram 双轨解码，目标是在 CPU / AMD iGPU（DirectML）等低资源设备上也能训练并跑出连贯的中文生成**（`README.md:3`，**确定**）。

| 项 | 值 | 出处 |
|---|---|---|
| 架构总纲 | Generator（4 层 attn 主干）+ 独立 Controller（1 层 GatedDeltaNet）双模型编排，Controller 产出 3 类信号条件化 Generator | `README.md:27`、`controller.py:56-63` |
| 架构形态 | **opt-in 特性矩阵**：全部由 `config['model']` 开关控制，默认关、旧权重向后兼容 | `README.md:18`、`model_config.py:247-254` |
| 当前基线 | R42，总参数 **6,756,101**（≈6.76M）、DML 训练 7346 tok/s | `baselines/r42_baseline.json:19`、`README.md:5` |

### 1.1 IGMCG 的全称和含义

**全称：查不到。**（**确定**）—— 全仓 26 个含 `IGMCG` 的文件（`*.py`/`*.md`/`*.yaml`，含 `archive_unused/`、`.opencode/memory/`）**无一处字母级展开**，正则 `IGMCG.{0,40}(Guided|Intuition|Generation|Multi|Candidate)` 与 `IGMCG\s*=` 均 0 命中。仓库只给出**中文释义**，两条：

| 释义 | 出处 |
|---|---|
| **直觉引导（多候选）生成** | `scripts/generate.py:167`、`models/transformer.py:739`、`docs/MODEL_USAGE_GUIDE.md:16` |
| **反碎片化**：生成多个温度候选，按综合分（连贯度 + 流畅度 + 风格 − 重复度）选优，抑制"碎片式"输出 | `README.md:14`、`README.md:160-171` |

含义拆解（**确定**）：

| 件 | 内容 | 文件:行 |
|---|---|---|
| 7 维直觉向量 | `IGMCG_DIMS = ['语气','氛围','意图','情感','风格','受众','创新']`，CLI `--intuition` 默认全 0.5 | `generate.py:174`、`:481-482` |
| 评分公式 | `score = 1.5*连贯度(coh) + 0.15*流畅度 + 0.15*风格匹配 − 2.5*重复度` | `README.md:164-166`、权重键 `generate.py:483-490` |
| 两层作用 | ①**模型内门控**：`igmcg_use_gate` 逐位置 sigmoid 决定"用不用"、`intuition_proj(7→1)` 注入偏置，`fused = logp + gate·ngram_vec`；②**脚本层多候选选优**：`generate_igmcg` 批量生成 N 个候选 + z-score 打分 | `transformer.py:744/750-753/1802-1813`、`generate.py:347-430` |
| 依赖 | `igmcg_enabled = bool(igmcg) and ngram_fusion_enabled` —— **依赖 n-gram 融合，融合关则 IGMCG 连带失效** | `transformer.py:744` |
| 与 direction 的关系 | **不是一回事**：direction 是 R42 学的 Controller 注入信号（`controller.py:240-241`）；IGMCG 是阶段 8.7 的解码期机制。仅共享"候选/采样"表象 | `model_config.py:250` vs `:259` |
| r42 取值 | `igmcg` **未配 → false**；`ngram_fusion` **未配 → false**（两者 r42 都关） | `model_config.py:259/257` |

### 1.2 r42 关键形状（**确定**，CPU 构建实测）

| 项 | 值 | 项 | 值 |
|---|---|---|---|
| vocab_size | 12000（**实际词表 9186**） | gen_dim = embedding_dim | 256 |
| hidden_dim（FFN） | 512 | num_heads | 8（head_dim 32） |
| num_layers | **4**（`layer_plan = attn,attn,attn,attn`） | max_seq_length | 64 |
| rope_max_len | 4096 | dropout | 0 |
| Controller | ctrl_dim 256 / heads 8 / **layers 1** / mem_slots **4** | 三信号 | direction / film / memory_compress **全 true** |
| 总参数 | **6,756,101** | Generator / Controller | 6,025,940（89.19%）/ 730,161（10.81%） |

### 1.3 设计意图（**用户口述拍板，2026-09-30**；只记录，不改码、不训练）

> 本节是**设计意图**，与 §4.2（状态机事实）、§7（direction 取证）、§8（作者看法）分属两套东西：
> §4/§7/§8 记的是**"代码现在是什么样、测出来是什么数"**，本节记的是**"当初想让它成为什么样"**。
> 与本节冲突的既有条目**一律原样保留、不删不改**；冲突点清单单独列在本轮汇报（见 `AGENT_MEMORY.md` 第六批「设计意图冲突点」）。

| # | 意图（用户原话口径） | 现状 / 出处 |
|---|---|---|
| 1 | **Controller 是替主干分担的帮手**：它和主干功能重叠是**故意的**；**主干在没有 Controller 时也必须能正常工作** | `transformer.py:1553-1563`：`_rt_controller=False` 时整段跳过 Controller 前向，主干独立走完；三信号投影零初始化中性起步（`controller.py:136-171`）——opt-in + 零初始化正是"主干不带 Controller 也能跑"的机制 |
| 2 | 分担**主要体现在长上下文**；`max_seq_length=64` 的冒烟里看不太出来，**Controller 在短序列收益小是正常的** | r42 `max_seq_length: 64`（`configs/config_train_8k_r42.yaml:85`）。**本文件与 `AGENT_MEMORY.md` 此前均未记载此口径**（属**缺位**，不是冲突） |
| 3 | 训练**本来就该有时联合训、有时分开训**<br>**细化（2026-09-30，只记录不实施）**：倾向**逐步交替**——一步带 Controller、一步不带，**不按 epoch 切**；比例**可配置**（先 1:1，再 2:1）；**不带的步直接跳过 Controller 的计算** | 现状是 epoch 级 `controller_warmup_frac`（`train.py:457-466`）+ 单边赋值（`train.py:278-279`，见 §4.2）。逐步交替的**只读改动评估**（改动点 / 能否省算力 / film 中性逐位）见本轮汇报 |
| 4 | **memory 的设计意图之一是跨段带着走** | 现状相反：训练 `use_cache=False` → `is_fresh` 恒真 → 每 batch `memory.reset()`（`transformer.py:1513-1519`），**跨段零保留**；且 r42 `memory_size` 未配 → MemoryBank 根本不建 |
| 5 | 现在是**冒烟阶段**，**数据少是故意的**，**不改喂法** | §4.4 记的 92.22% 截断是**事实记录**，不据此改数据管线 |

> **边界**：以上 5 条**只写进文档，本轮不改任何代码、不重训**。第 3 条的逐步交替是**倾向性方案**，未实施；第 5 条明确**否决**了"先修数据管线"这类排期建议（与 §8.2-5、§8.3 的作者推测**意见相反**，冲突点见本轮汇报）。

---

## 2. 数据流

### 2.1 训练整段路径（T=64，一次前向）

```
语料 8000 行 (merged_8k.txt)                         data_utils.py:95-96
  → CharTokenizer.encode 每行 [BOS] + 正文 + [EOS]     data_utils.py:401-413
  → 截断/补齐到 L = max_seq+1 = 65                     data_utils.py:37, 44-47   ← 92.22% 在这里丢
  → arr (N,65) int32 → torch                          data_utils.py:40,49
  → input = row[:64] / target = row[1:65]（右移 1）    data_utils.py:79-80
  → random_split 90/10 (seed=42) → 7200 / 800         train.py:511-518
  → DataLoader batch=24, shuffle=True → (B=24, T=64)   train.py:524,532
        │
        ▼
  embedding (B,T,256) ×√256 → dropout                 transformer.py:1478-1480
  → CharMergeLayer (B,T,256)                          transformer.py:1548 / layers.py:36-56
        │
        ├─► Controller 分支（r42 on, transformer.py:1545-1555）
        │     共享同一 embedding (B,T,256)             transformer.py:1027 / controller.py:104
        │     → ln_pre → GDN ×1 层 逐 t 递推           controller.py:116-119 / mixers.py:1264-1293
        │     → ①mem_kv (mk,mv) 各 (B,4,32)            controller.py:219-225
        │       ②film_per_layer (B,T,256)×3(layer0=None) controller.py:228-236
        │       ③direction = direction_proj(x.mean(T)) → (B,256)   controller.py:237-241
        │
        ▼
  x = x + direction.unsqueeze(1)   （广播到全序列）      transformer.py:1554-1555
        │
        ▼
  4 × Block 循环                                        transformer.py:1622-1728
    ├ attn 分支：ln1 → gpas1 → qkv (B,T,768)            transformer.py:375-379 / mixers.py:540-542
    │            → QK-Norm + 逐头温度                    mixers.py:544-547
    │            → RoPE（层内拆半：4 头 RoPE + 4 头 NoPE） mixers.py:557-568
    │            → attend + 全量因果掩码 + ALiBi          mixers.py:774-810
    │            → mem_cols=4 拼在真实 token 前（非因果）  transformer.py:342-349 / mixers.py:831-832
    │            → proj → output_gate → 残差门           mixers.py:810, 418-421
    ├ FiLM 调制（i>0）：x = x*(1+tanh γ) + β             transformer.py:1645-1648
    └ ff 分支：ln2 → gpas2 → SwiGLU w13(512→1024)/w2 → 残差   mixers.py:1848-1860
        │
        ▼
  ln_f（zero_centered RMSNorm）                         transformer.py:1729 / norma 出处 norms.py:9,22
  → output_head (256→12000, 与 embedding tied)          transformer.py:911-914, 1742
  → logits (B,64,12000)
  → CE(logits.view(-1,12000), target.view(-1), ignore_index=pad)   train.py:290, 593-596
```

### 2.2 逐字生成路径（增量，T=1）与训练路径的差异

```
prompt ids → prefill forward(past=None, use_cache=True)        transformer.py:1873-1876
          → sample_step(logits[0,-1])                          transformer.py:1881
          → 循环：_decode_one_step([[tok]], past)  T=1          sampling.py:90-110
                   → forward → 取 logits[:, -1, :] → 采样       transformer.py:1741-1742
```

| ⚠DIFF | 与训练路径的差异 | 文件:行 |
|---|---|---|
| ⚠DIFF-1 长度 | 训练整段 T=64；推理每步 **T=1**（首步 T=prompt 长） | `train.py:248` vs `sampling.py:106` |
| ⚠DIFF-2 **char_merge** | **第二处训推不一致（第四轮已修，开关默认关）**：`F.pad(x_t,(self.pad,0))` 在 T=1 时只左补 2 个零，窗口退化为 `[0,0,x_t]`，模块内无跨步滚动状态。修法：`char_merge.incremental_buffer=true` 时每步结果 `cat` 进 `_cm_buffer`，再按 kernel 长度切回窗口（等价于把窗口在时间轴上滚动），并把调用点**移到 `is_fresh` 复位块之后**，保证新序列先 `reset_buffer()` | `layers.py:41-42`、根因 `layers.py:22-34`；修 `layers.py` + `transformer.py:1548` |
| ⚠DIFF-3 **direction** | **H1**：训练 `x.mean(dim=1)` 覆盖整段；推理第 2 步起 T=1，均值只剩当前 token → 语义断裂 | `controller.py:240-241` |
| ⚠DIFF-4 Controller 缓存 | `_controller_past`（k/v/S/z）跨步保留，内部强制 `use_cache=True`；`is_fresh` 时 `reset_ngram_state()` 清 | `transformer.py:1546-1549`、`controller.py:207-208`、`:1745-1752` |
| ⚠DIFF-5 记忆列 | r42 MemoryBank 关（`memory_size` 未配→0），但 `mem_cols=4` 来自 Controller `mem_slots`，**恒不施加因果遮蔽** | `transformer.py:338,342-349`、`mixers.py:831-832` |
| ⚠DIFF-6 KV 累积 | 每步 `torch.cat` 累积 → O(L²)；`present=(k,v)` 已剥离 mem_cols | `mixers.py:692-693,723` |
| ⚠DIFF-7 掩码/ALiBi | 增量每步重建 base_mask+alibi，训练侧走 `_bias_cache` 复用；**两条路径同公式 → parity 测不出 H3** | `mixers.py:729,744` vs `:774-785` |
| ⚠DIFF-8 VRC | 增量 `v += λ·cached_last`；全量用 conv1d 递推，cache 存编码后 V 保 parity | `mixers.py:617-661` |
| ⚠DIFF-9 首步复位 | `is_fresh` 清 `_bias_key/_cached_T/_alibi_dist_cache/_cached_x0_proj` + `reset_ngram_state()` | `transformer.py:1513,1524-1537` |
| ⚠DIFF-10 early-exit | early-exit 仅 `not use_cache` 生效；剪枝 `continue` 仅 eval | `transformer.py:1623,1719-1728` |

> **⚠ 测量口径（第四轮实测）**：三口径 ppl **只在 CPU 上可比**。同一份 r42 checkpoint 在 DirectML 上 `tf`/`prefix` 与 CPU 系统性不同（direction off / 24 序列：CPU `tf=prefix=6.7058`，DML `tf=6.4490`、`prefix=6.4954`），而 **`incremental` 两端逐位一致**（cmb off `6.7103`、cmb on `6.7058`）。已 `git worktree` checkout `3f9c5ef`（第四轮改动之前的 `models/`+`scripts/`）在 DML 复跑 24 序列 direction off，得**逐位相同**的 `6.4490/6.4954/6.7103` → 确证 DML 的 `tf≠prefix`（T>1 整段前向泄漏）**先于第四轮改动存在**，与本修复无关（**OUT_OF_SCOPE**，另记）。故凡报 ppl 一律取 CPU；DML 只用于计时与生成。

---

## 3. 模块表

> 「参数量/占比」为 r42 实测；「三口径」= tf / prefix / incremental（越接近越好，`tf−prefix`≈泄漏量、`prefix−incremental`≈增量路径边界差异）。
> r42 三口径基线：**ctrl=on tf 4.6673 / prefix 5.2909 / incremental 10.5258**；**ctrl=off tf = prefix = 6.8774 / incremental 6.8798**（`AGENT_MEMORY §11.5`、`r42_baseline.json:27/:34/:41`）。

| 模块 | 文件:行 | 设计意图 | 输入→输出 | 参数量（占比） | 状态/缓存（存什么·何时清） | config 开关 = r42 | 三口径贡献 | 已知问题 |
|---|---|---|---|---|---|---|---|---|
| **Embedding** | `transformer.py:769`（×√d `:1478`） | 字符→向量 | (B,T)→(B,T,256) | 3,072,000（45.47%） | 无 | `vocab_size=12000`、`embedding_dim=256` | 未单独测 | 死参数见 §8 |
| **CharMergeLayer** | `layers.py:9-64`（根因 `:41-42`） | 深度可分离**因果**卷积取邻域 + sigmoid 门控插值，替代静态 BPE；开销≈注意力 1~2%（文档声明，无实测） | (B,T,D)→(B,T,D) | 66,816（0.99%） | 默认无跨步滚动状态（`:22-34` 只建 conv/gate/norm/drop）；开 `incremental_buffer` 时新增 `_cm_buffer` + `reset_buffer()`（`:15,60-64`） | `char_merge=true`、`kernel=3`、`dropout=0.0`（yaml:30-32）；`char_merge_incremental_buffer` 代码默认 **false** | `prefix−incremental` 原样 **−0.5343** → 关 char_merge 后 **−0.0001**；**开缓冲后 CPU 24 序列 `6.7058`、800 序列 `6.877422` 与 tf/prefix 完全相等（gap = 0.000000）** | **§11.8 任务 A → 已修**（开关默认关，需显式开启） |
| **注意力 QKV/投影** | `mixers.py:268-269,540-542,810` | 因果自注意力 | (B,T,256)→(B,T,256) | 3 层级 qkv 196,608 + proj 65,536 / 层 | KV cache（增量） | `mixer='attn'`、`share_attn_proj=false` | — | — |
| **QK-Norm + 温度** | `mixers.py:338-349` | 训练稳定 | 标量/head 维 | qk_norm 32 + log_temp 8 / 层 | 无 | `qk_norm=true`、`attn_temp=true`、`head_temp=true`（head_temp 依赖 attn_temp，**无校验**） | — | E3-9 |
| **RoPE + 层内拆半** | `rope.py:32,54`、`mixers.py:557-568` | 旋转位置；前半头 RoPE、后半 NoPE | rot_dim 16/32 | 0 | 缓存 `_cached_T`，`is_fresh` 清 | `rope_dim_fraction=0.5`、`yarn_scale=2.0`、`rope_max_len=4096`、`intra_hybrid_rope=true/ratio=0.5` | — | `yarn_orig_max_seq_length=0` 实际 `or 2048`（E3-3） |
| **ALiBi + 记忆列** | `mixers.py:279-291,424,459-473` | 线性位置衰减 | 标量偏置加到 logits | alibi_slopes 8 / 层 | `_bias_key/_alibi_dist_cache`，`is_fresh` 清 | `alibi=true`、`alibi_learnable=true` | tf 不敏感 | **H3**（距离未做 mem_cols 还原；max\|Δbias\|=1.78 分，**需重训**） |
| **Output gate** | `mixers.py:305-309,418-421` | query 门控消 attention sink | (B,T,256) | 65,792 / 层 | 无 | `output_gate=true` | — | — |
| **VRC（value 相对编码）** | `mixers.py:350-353,617-661` | `v += λ·v_{t-1}` | 同上 | 1 / 层 | 增量存编码后 V | `value_relative_coding=true` | — | — |
| **残差门 / GPAS / Norm** | `gates.py:56`、`transformer.py:370-382,438`、`norms.py:9,36` | 静态标量残差门 + LN 后可学 α | 标量 | 8 + 8 + 2,048（合计 0.03%） | 无 | `residual_gate=true`、`highway_gate=false`、`gpas=true`、`zero_centered_norm=true` | — | `residual_gate×highway_gate` 互斥 |
| **SwiGLU FFN** | `mixers.py:1848-1860` | 前馈 | 256→1024→256 | 393,216 / 层（共 1,572,864，23.28%） | 无 | `hidden_dim=512`、`fuse_swiglu=true` | — | 与 `share_ffn`/`moe` 互斥 |
| **cross_layer_routing** | `transformer.py:925-927` | DenseNet 风格 top-k 跨层注入 | — | 0（r42 未建） | 无 | `cross_layer_routing=false` | 未启用 | — |
| **layer_film / input_highway / progressive_residual / layer_skip / MoE / early_exit** | `transformer.py:944/954/938/—`、`moe.py:48`、`transformer.py:984` | 备用机制 | — | 0（r42 全未建） | 无 | **全 false（未配）** | 未启用 | — |
| **Controller 共享 embedding** | `transformer.py:1027`→`controller.py:104,198` | 复用 Generator 词嵌入，零额外词表参数 | (B,T)→(B,T,256) | 0（共享） | 无 | `tie_weights=true` | — | Controller 路径**无 char_merge/dropout**，两路输入表征不同（**推测**：设计差异，无编号） |
| **Controller GDN（逐 t 循环）** | `mixers.py:1264-1293`（`for t in range(T)` `:1265`） | delta rule 线性复杂度看全上下文，S=压缩状态 | (B,8,T,32)→(B,T,256) | 266,289（3.94%） | 每层 `past_kv=(k,v,S,z)`；`_controller_past` 存于 `transformer.py:1546`，`is_fresh`/`generate()` 开头清；**无 batch 尺寸守卫** | `controller=true`、`controller_layers=1`、`mixer` 内固定 GDN（`chunk_scan=False`，DML OOM `controller.py:114-115`）。⚠ **`chunk_scan=True` 与现行 for-loop 数值不等**（`mixers.py:1220-1222` 自述"改用标准形式，与原 for-loop 数值不等，但与增量解码 cache parity 一致"）→ 开启即**改数值、需重训** | — | **N5**（11,264 dispatch/前向、92.3% host 簿记、DML 慢 CPU 1.9×；优化方案 `review\S1.md` 路线①v2 预计 −40% dispatch 且**逐位等价**）；**M9** |
| **Controller norm** | `controller.py:107-111` | Pre-norm + per-layer RMSNorm | 256→256 | 512（0.01%） | 无 | — | — | — |
| **mem_kv（信号①）** | 生产 `controller.py:219-225` / 消费 `transformer.py:1669-1671,342-349`→`mixers.py:667-697` | 末层 GDN 状态 S 压成 M 槽记忆注入 attention | (mk,mv) 各 (B,4,32) | mem_query 1,024 + mem_proj 2,048（0.05%） | 每步现算，无跨步缓存 | `controller_memory_compress=true`、`controller_mem_slots=4` | 关 mem_kv：tf 4.6714 vs 全开 4.6575（差 **0.0139 nats**，几乎不背锅） | **H2 同类**（记忆列恒不遮蔽 `mixers.py:831-832`；H2 编号指向 MemoryBank，Controller 侧**无独立编号=查不到**） |
| **FiLM（信号②）** | 生产 `controller.py:228-236` / 消费 `transformer.py:1645-1648` | 逐层仿射调制 `x=x*(1+tanh γ)+β`，γ=β=0 中性 | (B,T,256)×3（layer0=None） | 394,752（5.84%） | 每步现算 | `controller_film=true` | 单独关：`prefix−incremental` −0.5343 → **+0.0007** | **传导通道**：char_merge 的错经 FiLM 放大（**推测**）；自身**因果、无 H1**（`AGENT_MEMORY §11.1`） |
| **direction（信号③）** | 生产 `controller.py:237-241` / 消费 `transformer.py:1554-1555` | 「动态 prefix bias」全局方向先验 | (B,256) | direction_proj 65,536（0.97%） | 每步现算 | `controller_direction=true` | tf 4.6673 → 关 dir 5.2951；prefix 下 开 5.2572 / 关 5.2486（**只值 −0.0086**）；incremental 关后 10.4623 → **5.7829** | **H1**（整段均值偷看，泄漏 0.6236 中 99.3%）；**需重训** |
| **零初始化中性起步** | `controller.py:136-171`（mem_proj/film/direction 全 0）、调用点 `transformer.py:1444-1447` | Controller 开但信号=0 时输出 ≡ `controller=False` | — | — | — | — | — | **T4**：parity 测试零初始化恒通过（3.6e-07），随机非零必挂 0.714 |
| **LM head（tied）** | `transformer.py:911-914,1449-1458` | 输出投影 | 256→12000 | 0 额外（与 embedding 共享 3,072,000） | 无 | `tie_weights=true` | — | — |
| **n-gram 第一套**（固定 l1/l2/l3 顺序嵌套插值） | `ngram.py:379-445`、入口 `:451-477` | 解码期统计先验 | 上下文→(V,) logprob | 0（不进模型参数） | `_logprob_cache`（512MB 预算 `:56-57`） | CLI `--ngram` 默认**关**、`--ngram-weight=0.3` | 未纳入三口径 | **H5**（两套并存，**记录勿合并**）；**M4**已修；**M5**未改 |
| **n-gram 第二套**（逐阶独立可学插值 + 门控融合） | `ngram.py:184-233,313-374`、消费 `transformer.py:1772-1788` | 训练+推理，模型自选用几个阶 | 同上 | `ngram_gate`、`ngram_order_logits`（r42 未建=0） | 同上 | `ngram_fusion=false`（未配） | **未纳入三口径** | **H5**、**M5**、**M10** |
| **IGMCG** | 门控 `transformer.py:739-753,1802-1813`；选优 `generate.py:347-430` | 多候选 + 综合分选优，反碎片化 | (B,7)→门控标量 | `igmcg_use_gate` 257 + `intuition_proj` 8（r42 未建=0） | 无 | `igmcg=false`（未配）、**依赖 ngram_fusion** | 未纳入三口径 | **M8**（评分把 prompt 计入均值）未改 |
| **采样 / temperature_applied** | `sampling.py:34,55-89`；property `transformer.py:1229-1237` | logits 域操作全做完再 softmax | (V,)→标量 id | 0 | 无 | 非 config（`GEN_PARAMS` 写死） | — | **L1/L2/L9 已修**、**H4 已接线**；**top-p 未实现**（`constants.py:22-23`） |
| **MemoryBank（备用）** | `memory.py:8,66,81,192`、`transformer.py:762,916-921` | 可学习压缩记忆 | — | 0（r42 未建） | `is_fresh`/batch 变才 `memory.reset` | `memory_size=0`（未配）→ 关 | 未启用 | **H2**、**M3**已修 |
| **QAT** | `qat.py`、`train.py:565-571` | 量化感知训练（LSQ-STE） | — | 0 | 无 | `qat_bits` 未配 → 关 | 未启用 | — |

**读表口径**：`tf−prefix ≈ 泄漏让指标虚低多少`；`prefix−incremental ≈ 增量路径自身的边界差异`；两者都应 ≈ 0（`baseline_eval.py:14-15`）。

### 3.1 参数量汇总（**确定**，CPU 实测）

| 模块组 | 参数量 | 占比 |
|---|---|---|
| embedding（=LM head 共享） | 3,072,000 | 45.47% |
| 4× FFN SwiGLU | 1,572,864 | 23.28% |
| 4× 注意力 | 1,311,940 | 19.42% |
| Controller（**不含共享 embedding**） | 730,161 | 10.81% |
| └ FiLM 投影 | 394,752 | 5.84% |
| └ GDN mixer | 266,289 | 3.94% |
| └ direction_proj | 65,536 | 0.97% |
| └ mem_query + mem_proj + norm | 3,584 | 0.06% |
| CharMerge | 66,816 | 0.99% |
| norms + 残差门 + GPAS + ln_f | 2,320 | 0.03% |
| **总计** | **6,756,101** | 100% |
| ⚠ **死参数**：embedding 第 9186~11999 行（词表实际 9186 < vocab_size 12000） | 720,384 | **10.66%** |

---

## 4. 训练流程

### 4.1 loss 组成（**确定**）

**r42 实际 loss = 主 CE 单项**（5 项辅助 loss 的开关在 r42 yaml 里全缺 → 全为 None/0）。

| 项 | 加法位置 | 算式 | 权重 / config key | r42 |
|---|---|---|---|---|
| **主 CE** | `train.py:290,593-596` | `CE(logits, target.view(-1), ignore_index=pad)` | 1.0；`label_smoothing` **被强制 0**（`train.py:584-591`）⚠ 代码注释称"与 ignore_index 冲突"，**该前提是错的**——torch 2.4.1 实测可开（`review\S5.md` #14）→ 归**明确 bug** | **开（唯一项）** |
| 层间对比 | `train.py:296-298` | `+= 0.01·Σ(1−cos_sim)` | 硬编码 0.01，**无 config key** | 关（`layer_contrastive=false`） |
| 早期退出 | `train.py:301-303` | `+= λ·Σ_k w_k·CE_k`，`w_k=1/(k+1)` | `early_exit_loss_weight=0.5` | 关（`early_exit=false`） |
| MoE 负载均衡 | `train.py:306-308` | `+= w·L_bal`（Switch 公式） | `moe_load_balance_weight=0.01` | 关 |
| MoE router z-loss | `train.py:309-311` | `+= w·L_z` | `moe_router_z_loss_weight=0.001` | 关 |
| 复杂度约束 | `train.py:317-324` | `λ·relu(comp − budget·max)` | `complexity_lambda=0.0` | 关（λ=0） |

- **蒸馏 loss：查不到**（全仓 `distill|teacher|kl_div|KLDiv` 0 命中）。
- **n-gram 不产生 loss 项**（只做前向门控融合，r42 关）。
- QAT 不产生 loss 项（`train.py:565-571` 仅伪量化算子）。

### 4.2 Controller 分阶段开关（**确定**，r42 = 3 epoch）

| 阶段 | 位置 | r42 实际 |
|---|---|---|
| 判定 `controller_warmup_active` | `train.py:457-466`（`(epoch-1)/total >= frac`），读配置 `:797`，逐 epoch `:815-816`，传参 `:851/:865` | epoch1 (0) <0.3 → **OFF**；epoch2 1/3=0.333 ≥0.3 → **应 ON**；epoch3 → ON |
| **强制关（bug 所在）** | `train.py:278-279`：`if not controller_active and model.controller_enabled: model._rt_controller = False`，**只有 False 分支、无 else** | 被判 OFF 的 epoch 关；被判 ON 的 epoch **不会开回** |
| 验证期开关 | `train.py:415`（`set_enhancements_active(True)` → `_rt_controller=True`，`transformer.py:1195,1202`）+ `:417-418`（同样单边关） | **全流程唯一能把开关打开的路径** |
| 状态打印 | `train.py:817-821`（**先打印、后进 train_epoch**） | epoch2 打印 "Controller ON"，但训练期实际仍 False |

**后果（代码状态机推导，无 r42 运行日志佐证 = 推测）**：epoch1 OFF（符合设计）→ **epoch2 训练期仍 OFF**（丢 300/900 = **33% 联合训练步**），而 epoch2 的 validate 却 ON → 训/验行为不一致 → epoch3 靠 `validate:415` 兜底 ON。

### 4.3 优化器与学习率调度（**确定**）

| 项 | 位置 | r42 取值 / 行为 |
|---|---|---|
| 优化器 | `train.py:603`→`AdamW :651-658` | `adamw`；`lr=3.0e-3`、`weight_decay=0.0`；`betas=(0.9,0.999)`、`eps=1e-8` **代码写死**（无 config key）；单参数组 |
| DML lerp 补丁 | `train.py:616-629` | privateuseone + adamw/adam 时 monkey-patch `lerp_`/`_foreach_lerp_` |
| 调度函数 | `compute_lr` `train.py:96-120` | `lr_schedule=wsd` |
| **warmup 步数** | `train.py:205` | `yaml warmup_steps=10` → **绝对 10 步**（非比例），clamp ≤ total |
| **WSD 三段** | `train.py:114-119` + yaml:71-73 | ① 10 步 warmup → 3.0e-3；② stable（progress<0.9，eff_step≤270）；③ eff_step 271→300 共 **30 步**余弦降到 `eta_min=0.0`（**末步 LR 精确 = 0**） |
| **LR 每 epoch 重放（bug）** | `total_eff` 按 epoch 重算 `:201-202`；`eff_step = initial_eff_step` `:210`；`initial_eff_step = resume_skip//accum if epoch==start_epoch else 0` `:850` | 非 resume epoch 恒 0 → **每 epoch 重跑 warmup(10)+stable+decay(30)**；epoch 末 LR=0（`:870` 打印）。日志 `logs_train_8k_v3_stdout.txt:698/1358/2018` 三次 `Learning rate: 0.000000`（⚠ 那是 SGD 另一配置，同一代码路径） |
| grad clip | 实现 A 循环 `train.py:146-153`；实现 B foreach `:143-144` | `gradient_clip=1.0`；**`use_foreach_norm_clip=false`**（实测 14.83→8.16 ms/步 −45.0%，默认关保数值路径） |
| AMP | `train.py:740-746` | `precision=fp32` → scaler=None |
| 早停 | `train.py:809` | `early_stop_patience=9999`（实际不早停） |

> **`use_foreach_optimizer` 已于 2026-09-29 撤销**：torch `AdamW(foreach=)` 默认值是 `None`（自动探测），DML 已选中快路径；显式 `false` 反而慢 40%（`None=62.83 / True=62.01 / False=86.16 ms`）。同梯度 10 步权重逐位相等（max\|diff\|=0.0）。现只剩注释：`config_train_8k_r42.yaml:76`、`train.py:599`、`_profile_train_steps.py:209`。

### 4.4 数据怎么切（**确定**，独立复算 `Temp\opencode\review\d_stats.py`）

| 步 | 位置 | 输入 → 输出 | 丢了什么 |
|---|---|---|---|
| 1 读行 | `data_utils.py:95-96` | 19,347,363 B / 6,864,161 字 → 8000 行 / 6,856,161 字 | 0 行 |
| 2 去重去空 | `data_utils.py:31-32→52-69` | 8000 → 8000 | 实测 dropped=0 |
| 3 建词表 | `data_utils.py:101-103` | 上限 12000 → 实际 **9186** | 0 |
| 4 分词 | `data_utils.py:350-360`（吃空白 `:353`、无效整词丢 `:358`） | 6,856,161 字 → 6,664,560 token | 191,601 |
| 5 +BOS/EOS | `data_utils.py:401-413` | → 6,680,560 id | +16,000 |
| 6 **截断** | `L=max+1` `:37`、**`tokens[:L]` `:44-45`** | 6,680,560 → **519,655** | **6,160,905 = 92.2214%**；7,975/8000 行（99.69%）被截；每行只剩 BOS + 前 63 正文，**EOS 仅 25 行幸存** |
| 7 切窗（**无 stride**） | `data_utils.py:74-83` | 每行 1 个窗口：input 64 + target 64 | 行内第 65 token 之后永不进训练 |
| 8 划分 | `train.py:511-518` + `data_utils.py:142-153` | train 7200（**460,525** 非 pad target）/ val 800（**51,130**） | val 10% |
| 9 Loader | `data_utils.py:122-139` | 7200/24 = **300 步/epoch**、1,536 tok/步、×3 = **900 步** | 0 尾批 |

**两个百分比不是互补量**（**确定**）：

| 口径 | 分子 / 分母 | 结果 |
|---|---|---|
| 截断丢弃率 | 6,160,905 / 6,680,560（未截断编码 id） | **92.2214%** |
| 同分母留存率 | 519,655 / 6,680,560 | **7.7786%**（= 100 − 92.22） |
| 「进训练比例」≈6.7% | 460,525（train 非 pad target）/ 6,864,161（语料字数） | **6.7091%**（**推测**：原始算式查不到） |

---

## 5. 生成流程

### 5.1 `scripts/generate.py` 分支表（**确定**）

| 分支 | 文件:行 | 触发条件 | 与别的分支差异 |
|---|---|---|---|
| interactive | `generate.py:560-567`→`interactive_mode :97-148` | `--interactive` | 一条 prompt 跑温度序列；不传 `--temperature` → 双温度 [0.7,0.9]（`:63,90-94`） |
| prompt + IGMCG | `generate.py:569-586`→`generate_igmcg :347-425` | `--prompt` 且 `--igmcg` | 多候选 batch 并行 + z-score 打分（`:401-424`）；缺省 rep=2.0 |
| prompt 普通 | `generate.py:587-606`→`generate_text :28-44` | `--prompt` 无 `--igmcg` | 单序列自回归；缺省 rep=**1.4**；结果写 `logs/generation_output.txt` |
| 默认示例 | `generate.py:607-628` | 无 `--prompt` 且无 `--interactive` | 4 条硬编码 prompt 循环 |
| n-gram 构建 | `generate.py:535-546` | `--ngram` | `NGramModel(max_order=3, max_lines=2000, min_count=2)` |
| 量化/编译/精度 | `generate.py:516,518,524-531` | 显式 CLI | bf16/fp32 AMP 上下文 |

### 5.2 `model.generate` 与采样层（**确定**）

| 分支 | 文件:行 | 触发条件 | 行为 |
|---|---|---|---|
| greedy | `sampling.py:55,84-85` | `temperature <= 0` | 跳过除法与 softmax，掩码/惩罚后 `argmax` |
| 采样 | `sampling.py:86-89` | `temperature > 0` | `softmax` → `multinomial` |
| 全 -inf 回退 | `sampling.py:72-83` | 所有合法 token 被屏蔽 | `raw_logits.clone()`（L9）仅屏蔽 pad/bos |
| 低置信提前终止 | `sampling.py:87-88` | `probs.max() < 0.01` | 批量填 pad 保对齐；单序列 break |
| EOS / min_length | `sampling.py:64-67` | `len(generated)−len(prompt_ids) ≥ min_length` 才放行 EOS | `min_length=3`、`eos_penalty=-5.0` |
| 温度 | `sampling.py:55-56` | — | `lt = logits_t / τ`；`temperature_applied` 或 greedy 时不除 |
| 重复惩罚（**加性**） | `sampling.py:7-23,57` | — | `Counter(generated_ids)`；`logits[ids] -= penalty·count` |
| n-gram 先验叠加 | `sampling.py:58-59` | 非融合路径 | `lt += ngram_weight·ngram_fn(...)`，`ngram_weight=0.3` |
| top-k | `sampling.py:68-71` | — | `lt[lt<thr] = -inf`，`top_k=50` |
| **top-p** | **查不到实现** | — | `constants.py:22-23` 明写"未在 sample_next_token 实现" |
| 融合 vs 非融合 forward | `transformer.py:1734-1743` / `:1785-1818` | `_ngram_fusion_active` | 融合：`log_softmax(z/τ)+gate·prior`；非融合：raw logits，除温在采样端 |

**softmax 与 logprob 先后**：一切 logits 域操作（温度/惩罚/先验/屏蔽/top-k）**全部先做完，最后才 softmax**（`sampling.py:56→57→59→60-71→86`）；采样端**不算 logprob**。评分侧 `_fluency_batch` 返回值**不是**归一化 log-prob，必须再 `F.log_softmax`（`generate.py:320-326`，关联 **M6**）。

**`temperature_applied`**：单一事实来源是 property `transformer.py:1229-1237` = `ngram_fusion_enabled and _ngram_fusion_active`（与 forward `:1734` 分支同式）；调用点 `transformer.py:1864,1890`、`generate.py:247,272`。作用是告诉采样端"forward 已除过 τ"，避免错误缩放 n-gram 先验。**r42 → False** → 采样端自己除。

### 5.3 状态初始化与重置（**确定**）

| 事项 | 文件:行 | 内容 |
|---|---|---|
| `generate()` 开头 | `transformer.py:1842-1844` | `self.eval()` + `reset_ngram_state()`（清 `_ngram_last_ids` + `_controller_past`，`:1745-1752`） |
| 首步 forward | `transformer.py:1483-1513` | `past=None` → 包成 `BlockState`；`is_fresh = (not use_cache) or all(pk is None)` |
| `is_fresh` 额外清 | `transformer.py:1524-1537` | attn `_bias_key`/`_cached_T`/`_alibi_dist_cache`、`_cached_x0_proj`、再调 `reset_ngram_state()`（**M9 修复点**） |
| **batch 变化** | `transformer.py:1767-1771` | `_ngram_last_ids.shape[0] != src.shape[0]` → 用 pad 重建滚动缓冲；**batch 相同而序列不同靠 `is_fresh` 清**（M9 成因） |
| 批量候选开头 | `generate.py:241-252` | `no_grad` + `model.reset_ngram_state()` + 首次 `forward(past=None, temperature=(N,))` |
| 三口径之间 | `baseline_eval.py:211-229` | `set_enhancements_active(True)` → 再设 `_rt_controller`（**顺序不能反**）；每 mode 前 `zero_grad()` |
| 三口径输入构造 | tf `baseline_eval.py:108-119`（整批一次前向）／prefix `:122-136`（逐 t 前缀，O(T²)）／incremental `:139-155`（`use_cache=True` 逐 token） | `DEFAULT_PROMPTS` 7 条见 `baseline_eval.py:42-53` |

### 5.4 `GEN_PARAMS`（生成基准，**确定**）

`baseline_eval.py:55-58` = `{max_length:30, temperature:0.8, top_k:50, repetition_penalty:1.4, min_length:3, eos_penalty:-5.0}`；greedy 分支覆写 `temperature=0.0`（`:171`）；`--seed` 默认 **42**，每 (prompt, mode) 前重设 `random/np/torch` seed（`:166-168`），`mode ∈ {sample, greedy}`。

---

## 6. 演进史

| # | 时期/轮次 | 动机 | 效果 / 为什么留下或放弃 | 出处 |
|---|---|---|---|---|
| 1 | 2026-07 早期 | 词级词表 UNK 泛滥（vocab5000 覆盖仅 21.85%，生成 68% 是 `<???>`） | CharTokenizer 零 OOV 落地，编码 UNK 0.000% → **留下** | `CHANGELOG`「可学习字符词表落地」段 |
| 2 | R1~R8 | 工程化 | ABCDE 工作流整合、`transformer.py` 拆分、ModelConfig 特性开关集中 → **留下** | `AGENT_MEMORY §6 轮次演进表`（表起 `:194`） |
| 3 | R9~R10 | 性能 | DML fused SDPA **~22×** 训练提速（1500→68 ms/step）；ngram logprob 向量化 **63×**（83→1.5 s/it）→ **留下** | `CHANGELOG:466/498` |
| 4 | R11~R14 | 跨层协作特性堆叠 | cross_layer_routing / layer_film / input_highway / progressive_residual / layer_contrastive… + RoPE 缓存线程安全 + loss_sum 泄漏修 → 大多**留作 opt-in、默认关** | `AGENT_MEMORY §6 :200`、`CHANGELOG:754-755` |
| 5 | R15~R17 | 线性注意力/记忆 | GatedDeltaNet / Partial RoPE / OutputGate / ZeroCentered + MLA KV；**R17 cache 协议修复** → **留下** | `AGENT_MEMORY §6 R15~17` |
| 6 | R18~R27 | 继续堆 | iRoPE / KDA / YaRN / RWKV7 / intra_hybrid / GPAS / alibi_learnable；MLA+VRC 使增量解码 O(T²)→O(1) → **留下** | `AGENT_MEMORY §6 R25~27`、`CHANGELOG` 第二十六轮 |
| 7 | R28~R34 | 转向算子合并精简，无新架构 | R29 mixer 对比：**attn_linear Val 5.86 最好**，linear2d/hybrid_linear2d/linear 均 5.94 | `AGENT_MEMORY §6 R29`、`§2.1 F` |
| 8 | R35~R36 | early_exit、MoE、chunk_scan、QAT | **R35 的 lerp/addcmul 融合 kernel 实测 DML backward 崩 + CPU 回退 → R40 全面回退**（`layers.py:47-51` 注释仍在） | `AGENT_MEMORY §3`、`§8.2` |
| 9 | R37 | 审查修复 | `sgd`→`adamw`；**attn_linear→attn**（R29 实测速度/质量相等，省线性分支 + mixer_gate） | `CHANGELOG:95`、`configs/config_train_8k.yaml:32` |
| 10 | R38/R40/R41 | DML 收尾 | DML 提速 16%、融合算子回退收尾、ngram 内存爆炸补字节预算 | `CHANGELOG` |
| 11 | **R42** | 双模型编排 | **Controller(GDN)+Generator**，三信号零初始化中性起步、`controller_warmup_frac` 三阶段、与 `linear2d/hybrid_linear2d` 互斥 → **留下（当前基线）** | `CHANGELOG:12`、`README.md:27`、`config_loader.py:76-80` |
| 12 | 2026-09-28 架构审查 | 正确性 | H1~H5 / M1~M11 / L1~L9 分级 → 零风险批① + 第二批 H4/M11/三口径基准脚本 | `AGENT_MEMORY §11.1-§11.7` |
| 13 | 2026-09-28 泄漏归因 | 找根因 | **99.3% 泄漏 = H1**；产出 `docs/H2_CAUSALIZATION_PLAN.md`（B1 逐位置前缀状态 / B2 读侧因果遮蔽 / B3 训推一致化） | `§11.7`、`H2 §3` |
| 14 | 2026-09-29 第三批 | 继续 | 第二处不一致定位到 `layers.py:41-42`（**推翻上轮「char_merge 实现正确」**）、GDN 11,264 dispatch 定量、foreach_norm 裁剪 −45%/步（开关默认关）、六方向 111 条审查（26 条需重训） | `§11.8` |

**试过但被否 / 回滚的**（均**确定**）：

| 试过什么 | 结局 | 出处 |
|---|---|---|
| ALT 整体 50% 随机关 | 弃用改 SEL；SELv2 全量下最差、ENH 最好 | `CHANGELOG 290b790`、`42323fb` |
| attn_linear 混合 | R37 切回纯 attn | `config_train_8k.yaml:32` |
| hybrid_linear2d 门控初始化 | linear2d 占主导、attn 被压制 | `§2.1` 特性表 |
| 逐 token 因果写记忆 | 否决，开销 **14×**（0.21→3.20 ms/write） | `CHANGELOG:474` |
| chunk_scan=True | DML OOM（A_mats/B_mats 50MB/tensor）→ 只能 for-loop | `AGENT_MEMORY:314`、`CHANGELOG:73` |
| QAT 权重缓存 | Val 9.22→12.70 **回退** | `CHANGELOG:443` |
| 20M memory 扩容 | 4000 行小数据过拟合（mem0 vl8.5 ≫ mem32 vl13.6），**4.3M 档仍是默认甜点** | `CHANGELOG:510/535` |
| layer_skip | 训练期负收益 → 保持关 | `CHANGELOG:550` |
| CLSA | 不实施 | `CHANGELOG:281` |
| 早期扫描脚本把路径字符串误传 `TextDataset` | 此前所有 val_loss 绝对值作废、速度数字不受影响 | `CHANGELOG:515` |
| 「Controller 开销 15%」 | 更正为 **3.0×**（616 vs 203 ms/步） | `§10.8`、`CHANGELOG:38` |
| 「DML 慢路径/CPU 回退」怀疑 | 被「92.3% host 簿记」推翻 | `§11.6 N5` |
| 「char_merge 实现正确」 | **推翻**（`layers.py:41-42`） | `§11.8 A` |

**主线（推测）**：堆特征 opt-in（R1-25）→ 算子级压榨与多次试错回滚（R26-41）→ 双模型编排收口决策（R42）→ 审查驱动的正确性/训推一致性量化（9-28 起）；目标始终是低资源设备上跑连贯中文（`README.md:3`）。

---

## 7. direction：原本的作用与替代方案

### 7.1 设计意图（**确定**）

`controller.py:237-241`：Controller 末层输出 `x (B,T,ctrl_dim)` 沿时间取**整段均值** → `direction_proj (ctrl_dim→gen_dim)` → `(B,256)`；`transformer.py:1554-1555` 以 `x = x + direction.unsqueeze(1)` **广播到全序列**。投影零初始化 → 中性起步、与 `controller=False` 逐位 parity（`controller.py:157-158`、`transformer.py:1331-1334`）。意图是**动态 prefix bias**：让 Controller 概括"这一段在讲什么"，给 Generator 一个全局方向先验。

### 7.2 为何偷看（H1，**确定**）

- `x.mean(dim=1)` 是**整段**均值，位置 t 的 direction 含 t+1..T-1 信息 → 非因果；增量第 2 步起 T=1，均值只剩当前 token → 训练(整段)/推理(单 token) **语义断裂**。
- 定量：泄漏 0.6236 nats 中 **99.3% 来自 H1**；关 direction 后 `tf−prefix` 由 0.5997 → **0.0035**，关 mem_kv 几乎不降（差 0.0139）。
- 生成侧同向证据：`ctrl_on` 全部塌缩、`dir_off` 明显改善、`ctrl_off` 略优于 `dir_off`（繁体/疑似乱码更多，见 §8）。
- 现有 parity 测试**空转**：零初始化下 diff 3.6e-07，随机化投影后 0.714（**T4**）。

### 7.3 砍掉后谁补（**确定**）

| 事实 | 数字 |
|---|---|
| direction 在**因果口径**只值 | prefix 下 开 5.2572 / 关 5.2486 → **−0.0086 nats（略负）** |
| tf 口径会掉 | 4.6673 → 5.2951（**0.62 nats 是作弊分**） |
| incremental 口径关掉反而**少** 4.68 nats 退化 | 10.4623 → 5.7829 |
| 条件化由 mem_kv + FiLM 继续承担 | 两者因果合计 **1.46 nats** |
| 剩下的 0.53 nats 增量差归 char_merge | **已修**（`char_merge_incremental_buffer`）：G1 `gap +0.529219` → G2 `+0.000052`；ctrl=off 口径 24 序列 `−0.0045 → 0`、800 序列 `−0.002340 → 0.000000`（§11.8 A → §11.10） |

**推测**：泄漏与真实增益耦合，现有证据无法区分 direction 那 0.62 里有多少是真本事。

### 7.4 三个"不偷看"的替代方案

| 方案 | 机制 | 为何因果 | 代价 | 收益不确定性 | 出处 |
|---|---|---|---|---|---|
| **0 号（对照，不改码）** | `controller_direction: false`（运行时开关 `controller.py:239` 已实测 `SWITCH_EFFECTIVE`） | — | **0 代码、0 参数、不重训也能立刻止血**；代价 tf −0.63（不可兑现） | — | `model_config.py:252`、`§11.8 a` |
| **① 用 GDN 第 t 步隐状态 h_t**（**最推荐**） | 去掉 `x.mean(dim=1)`，直接 `x→direction_proj→(B,T,D)`，注入处去 `unsqueeze(1)`；改 2~3 行 + 新开关 | GDN delta rule 递推 `h_t` 只依赖 ≤t；增量 T=1 同构 → **顺带修掉 H1 语义断裂** | **需重训**；0 额外参数；算力增量≈0 | 由"一个全局先验"变"逐位置局部信号"，单点信息量可能下降 | `controller.py:240-241`、`transformer.py:1555`、`§11.6` |
| **② 只用 prompt 算方向、生成期冻结** | 首步算一次存实例状态，后续复用常量；先例 `input_highway._cached_x0` | prompt 全部 ≤ 当前 t；生成期是常量 | ~15 行；缓存复位须并入集中管理（否则重蹈 M9/N3）；**需重训** | **训练侧语义必须先定义**——预训练 seq64 定长切块没有 prompt/续写边界，需指定 anchor，否则又造一处训推不一致（三方案里唯一要动数据管线假设的） | `transformer.py:1523/1531`、`§11.3 M9`、`§11.6 N3` |
| **③ 因果累积均值 / 滑窗均值** | `direction_t = proj(mean(x_0..x_t))`；训练用 `_parallel_prefix_scan`，增量用 running sum O(1) | 每个 t 只用 ≤t；训练 cumsum 与增量 running mean 数学等价 → **训推天然一致** | ~10-20 行；0 参数；**需重训**；滑窗多一个超参 w | 与原语义最接近（同为均值，只是范围收到前缀），迁移成本最低、最可能"零惊喜"；但仍是全局量，长序列不如滑窗稳定 | `H2 §3-B1`、`mixers.py` GatedDeltaNet 在用 |

**决策提示（推测）**：只先止血 → 0 号；一次重训定长期方向 → **①**，②③ 作消融对照。三者都必须与 H2/H3 同批重训，并按"新默认开、r42 保持旧行为"走开关。

---

## 8. 我自己的看法（**全部标推测**）

### 8.1 设计得好的地方

| # | 判断 | 依据（**确定**的事实 → 推测的结论） |
|---|---|---|
| 1 | **opt-in 特性矩阵 + 零初始化中性起步** 是这套架构最值钱的设计 | 全部开关默认关、旧权重加载自动关；Controller 三信号投影全 0 → 开着也不改变输出（`controller.py:136-171`）。**推测**：这让 42 轮大改动能安全落地，也是唯一让"边跑边改"不翻车的机制 |
| 2 | **三口径基准（tf/prefix/incremental）** 是这轮最有价值的基础设施 | 它把"训推不一致"从感觉变成了可测的 nats（`baseline_eval.py:108-155`）。**推测**：H1 和 char_merge 两处根因都是靠它定位的，没有它这套架构根本查不下去 |
| 3 | **mem_kv + FiLM 这两条信号选得对** | 因果口径合计 1.46 nats、且自身无 H1（`§11.1`）。**推测**：把"前文摘要"职责放在递归状态 S 上是自然且正确的，direction 反而是多余的 |
| 4 | **梯度裁剪分两套实现 + 开关默认关** 的处理干净 | foreach 实测 −45%/步但默认关保数值路径，取证充分（EQUIVALENT 三项 + 逐位相等 0.0）。**推测**：这是本仓少数"性能与正确性解耦"做得干净的案例 |

### 8.2 累赘或互相冲突的地方

| # | 判断 | 依据 |
|---|---|---|
| 1 | **同一份输入喂给两条路径时表征不一致**（Generator 过 CharMerge+dropout，Controller 不过） | `controller.py:195-201` 全文无 char_merge，`transformer.py:1479-1482` 有。**推测**：Controller 看到的字符流和 Generator 不是同一个，这会削弱三信号的条件化效果，也可能就是"两路输入表征不同"的隐形 bug；**查不到**是否有意为之、无编号无测试 |
| 2 | **`_rt_controller` 的状态机靠副作用维持** | `train.py:278-279` 单边关、无 else；唯一打开路径在 `validate:415`。**推测**：这是"用 validate 的副作用给 train 打补丁"，r42 因此丢了 33% 联合训练步——属于**控制流与配置语义冲突**，比任何单个算子 bug 都严重 |
| 3 | **LR 调度按 epoch 重放** | `train.py:201-202,210,850`。**推测**：WSD 的"稳定→衰减"被切成了 3 段重复，每个 epoch 末 LR 精确归 0，等于每轮都在做一次独立的完整退火。与"3 epoch 连续训练"的意图直接冲突 |
| 4 | **两套 n-gram 插值并存且语义不同**（固定 l1/l2/l3 嵌套 vs 逐阶独立可学） | `ngram.py:21-26` docstring 自己写明"保留两条数值路径…语义不同"，**H5 定为"记录勿合并"**。**推测**：这是历史上两套方案各留一半，现在变成需要读者记住的隐性语义分叉；不算 bug，但属于**认知负担型累赘** |
| 5 | **数据管线是整台机器的天花板** | 每行截断到 65 token、**无 stride**、92.22% 的 token 永不进训练。**推测**：在 H1/char_merge/GDN 全部修完之前，模型质量的上限就被卡在这里——**修数据的收益大概率大于修任何单个算子** |
| 6 | **Embedding 尾部 10.66% 是死参数** | 词表实际 9186 < `vocab_size` 12000，`train.py:503-508` 只校验 `>=` 不校验 `=`。**推测**：tied LM head 还要为 2814 个不存在的类算 logits，属于**白算 + 白训**；改法只需把 `vocab_size` 降到实测值或加等式校验 |
| 7 | **死配置没有统一告警** | memory 有 M3 warning（`model_config.py:132-143`），但 r42 配的 8 个 `ssm_*` 因 `layer_plan` 全 attn 而永不构建，**查不到对应 warning**。**推测**：同一类问题两种待遇，读者会误以为 ssm 配置生效了 |
| 8 | **`controller_warmup_frac` 与 `enhancement_schedule` 两套调度机制并存但只有一套在用** | r42 无 `enhancement_schedule` 键；`controller_warmup_active` 是独立实现。**推测**：属于**调度机制冗余**，且正是这个冗余让 M1 的"单边关"逃过了测试 |

### 8.3 结论（**推测**）

这台机器**骨架是对的**（因果主干 + 零初始化 opt-in + 三口径可测），**主要问题不在架构选型，而在三处"接线"**：①状态复位/开关的控制流（M1/LR 重放）、②增量路径的滚动状态（char_merge/H1）、③数据吞吐（92.22% 截断）。**先修不影响权重的三项，再把必须重训的打包成一次重训**，是当前风险最低的顺序。

---

## 附录 A 配置开关索引（默认值 ≠ r42 取值，或本轮/上轮新增）

> 全部 116 个 schema 字段 + 44 个 training/data 字段的完整清单见 `models/model_config.py`（字段声明行 12-264）与 `configs/config_train_8k_r42.yaml`；本表只列**需要关注**的行。

### A.1 `model` 段（**确定**）

| 开关 | 默认值 | r42 | 作用 | 互斥/校验 |
|---|---|---|---|---|
| `vocab_size` | 0（必填） | **12000** ⚠ | 词表行数（**实际建成 9186**） | `model_config.py:267`；与 tokenizer 单向 `>=` 校验 `train.py:503-508` |
| `embedding_dim` | 0（必填） | **256** | 隐藏维（=gen_dim=ctrl_dim 回退） | `:268` |
| `num_heads` | 0（必填） | **8** | 头数 | `:269/:273` |
| `num_layers` | 0（必填） | **4** | Generator 块数 | `:270` |
| `hidden_dim` | 0（必填） | **512** | FFN 中间维 | `:271` |
| `max_seq_length` | 0（必填） | **64** | 上下文窗口 | `:272` |
| `gradient_checkpointing` | true | **false** ⚠ | 反向重算 | `transformer.py:1042`，与 `grad_ckpt_auto` 联动 |
| `layer_plan` | None（全 attn） | `"attn,attn,attn,attn"` ⚠等价 | 每层 block 类型 | 与 mixer 组合警告 `config_loader.py:62-72` |
| `rope_max_len` | None→64 | **4096** ⚠ | 位置编码容量 | 回退 `model_config.py:275-276` |
| `char_merge` | false | **true** ⚠ | CharMergeLayer | 已修：需配 `char_merge_incremental_buffer=true`（r42 **未配**） |
| `tie_weights` | true | true | head/embedding 共享 | — |
| `fuse_swiglu` | false | **true** ⚠ | SwiGLU w13 合并 | — |
| `yarn_scale` | 1.0 | **2.0** ⚠ | YaRN 外推倍数 | `rope.py:62` |
| `alibi` | false | **true** ⚠ | ALiBi 位置衰减 | `+mem_cols>0` 触发 **H3** 警告 `model_config.py:307-318` |
| `alibi_learnable` | false | **true** ⚠ | 斜率可学 | 需 `alibi=true` `:108-111` |
| `output_gate` | false | **true** ⚠ | 注意力输出门 | — |
| `zero_centered_norm` | false | **true** ⚠ | Zero-Centered RMSNorm | — |
| `rope_dim_fraction` | 1.0 | **0.5** ⚠ | Partial RoPE 占比 | — |
| `head_temp` | false | **true** ⚠ | per-head 可学温度 | 需 `attn_temp=true`，**否则静默失效、无校验** `mixers.py:347` |
| `value_relative_coding` | false | **true** ⚠ | `v += tanh(λ)·v_{t-1}` | — |
| `intra_hybrid_rope` | false | **true** ⚠ | 层内 head 拆半 RoPE/NoPE | 需 `alibi=true`、禁 `use_mla_kv`、仅 attn/attn_linear `model_config.py:88-106` |
| `intra_hybrid_ratio` | 0.5 | 0.5 | NoPE head 占比 | (0,1) 开区间 `:104-106` |
| `gpas` | false | **true** ⚠ | LN 后可学 α | — |
| `controller` | false | **true** ⚠ | 启用 Controller | **与 `mixer=linear2d/hybrid_linear2d` 互斥** `config_loader.py:76-80` |
| `controller_layers` | 2 | **1** ⚠ | GDN 层数 | `:287` |
| `controller_mem_slots` | 4 | 4 | mem_kv 槽数 M | `:288` + H3 警告 `:311-313` |
| `controller_direction` | true | true | 信号③ | 三信号至少其一 `:294-296` |
| `controller_film` | true | true | 信号② | 同上 |
| `controller_memory_compress` | true | true | 信号① | 同上 |
| `ngram_fusion` | false | 未配→false | 统计先验融合 | 开但无 model → warning `transformer.py:733-736` |
| `igmcg` | false | 未配→false | 直觉引导多候选 | **依赖 ngram_fusion** `transformer.py:744` |
| `memory_size` | 0 | 未配→0 | 记忆槽总数 | =0 时其余 memory 键=死配置 warning `:132-143` |
| `char_merge_incremental_buffer` | **false** | 未配→**false** | 第四轮新增：CharMerge 增量滚动缓冲（T=1 时把结果 `cat` 进 `_cm_buffer` 再按 kernel 切窗，替代左补零） | `layers.py:15,44-57,60-64`、`model_config.py`、调用点 `transformer.py:1548`、`reset_ngram_state()` 清缓冲；**非 config 键，也可运行时 `model.char_merge.incremental_buffer = True` 直开** |
| `gated_delta_channel_wise` | false | 未配 | KDA 逐通道衰减 | ⚠ **`from_dict` 漏读 → 配置键静默失效** `model_config.py:333-365` |

### A.2 `training` 段（**确定**）

| 开关 | 代码默认 | r42 | 说明 |
|---|---|---|---|
| `batch_size` | 必填 | 24 | `train.py:524,532` |
| `epochs` | 必填 | 3 | `train.py:750,813` |
| `learning_rate` | 必填 | **3.0e-3** | |
| `optimizer` | adamw | adamw | DML 下 lerp patch `train.py:616` |
| `weight_decay` | 必填 | 0.0 | |
| `gradient_clip` | 必填 | 1.0 | |
| `warmup_steps` | 0 | **10** ⚠ | **绝对步数** `train.py:205` |
| `early_stop_patience` | 5 | **9999** ⚠ | ≈关闭 |
| `label_smoothing` | 0.0 | 0.0 | >0 会被**强制忽略**+warning `train.py:586-591` |
| `cpu_threads` | None | **4** ⚠ | |
| `checkpoint_percents` | [] | **[0.25,0.5,0.8]** ⚠ | |
| `precision` | fp32 | fp32 | |
| `grad_accum_steps` | 1 | 1 | |
| `eta_min` | 0.0 | 0.0 | WSD 末步 LR **精确 = 0** |
| `lr_schedule` | cosine | **wsd** ⚠ | |
| `wsd_decay_frac` | 0.1 | 0.1 | |
| `compile` | false | false | DML 下不可用 |
| **【新】** `use_foreach_norm_clip` | false | false | `_foreach_norm` 实测 14.83→8.16 ms/步（−45.0%），默认关保数值路径 |
| **【新】** `controller_warmup_frac` | 0.0 | **0.3** ⚠ | `train.py:457-466`；关联 **M1** |
| `enhancement_schedule` | None | 未设 | 键名须在 `ENHANCEMENT_KEYS` `transformer.py:615-616` |
| ~~`use_foreach_optimizer`~~ | **已撤销** | 仅剩注释 | torch `foreach=None`=自动探测，DML 已选快路径 |

### A.3 `data` / `paths` / `device` / `seed`（**确定**）

| 开关 | 代码默认 | r42 | 说明 |
|---|---|---|---|
| `train_file` | 必填 | `data/pretrain_corpus/merged_8k.txt` | `train.py:493` |
| `vocab_size`（data 段） | 必填 | 12000 | **建词表上限**，实际 9186 |
| `max_seq_length`（data 段） | 必填 | 64 | 与 model 段**无交叉校验** |
| `num_workers` | 4 | **0** ⚠ | Windows 强制 0 `data_utils.py:127-128` |
| `test_split` | 0.0 | **0.1** ⚠ | `train.py:511` |
| `checkpoint_dir` | 必填 | `checkpoints_train_8k_r42` | |
| `log_dir` | 无默认 | `logs_train_8k_r42` | ⚠ **全仓无代码读取 → 死键** |
| `device` | auto | auto | |
| `seed` | 必填 | 42 | `train.py:475` |

---

## 附录 B 已知问题编号索引

> 编号唯一依据：`AGENT_MEMORY.md` §11.1~§11.8 + `CHANGELOG.md`。状态：✅ 已修/已接线　❌ 未改　⚠ 需重训

### B.1 H（训推一致性 / 正确性）

| 编号 | 一句话 | 位置 | 状态 |
|---|---|---|---|
| **H1** | Controller direction `x.mean(dim=1)` 整段均值 → 增量 T=1 语义断裂；泄漏 0.6236 nats 中 **99.3%** | `controller.py:240-241`、`transformer.py:1554-1555` | ⚠ 需重训（三方案见 §7.4） |
| **H2** | 记忆读侧未来泄漏：记忆列不施加 causal（实测 **0.0035 nats = 0.7%**） | `mixers.py:831-832` | ⚠ 需重训（方案 `docs/H2_CAUSALIZATION_PLAN.md`） |
| **H3** | ALiBi 距离未做 mem_cols 还原；max\|Δbias\| = **1.78 分**（同 row 相对扭曲 ≈ **8×0.445859 ≈ 3.57**，记忆列 bias 恒 0 → 整体平移相对记忆列不抵消） | `mixers.py:459-473` | ⚠ 需重训；RuntimeWarning 已加 `model_config.py:307-318` |
| **H4** | `generate.py` 4 采样参数 3/4 分支被无视 | `generate.py:387-393` 等 | ✅ 已接线（`_GEN_DEFAULTS`） |
| **H5** | 两套 n-gram 插值并存 | `ngram.py:374-399`、`transformer.py:1776` | ❌ **记录，勿合并** |

### B.2 M（状态/缓存/接线）

| 编号 | 一句话 | 位置 | 状态 |
|---|---|---|---|
| **M1** | Controller warmup OFF→ON 依赖 validate 副作用；`train_epoch` 单边关无 else | `train.py:278-279,415-418` | ❌ 未改；r42 **丢 33% 联合训练步** |
| **M2** | 剪枝 docstring 与实现不符 + 记忆路径不对称 | `transformer.py:1300-1329,1623` | ✅ 已修 docstring |
| **M3** | 配置校验缺口（死配置静默） | `model_config.py:126-149,297-317` | ✅ 已修（warn） |
| **M4** | ngram `_logprob_cache` 无字节预算 | `ngram.py:51-58,434-435` | ✅ 已修（512MB） |
| **M5** | 训练期 ngram 每步 `.cpu()` 同步 | `ngram.py:255-267,324-329` | ❌ 未改（r42 ngram 关） |
| **M6** | `_fluency_batch` 走 `use_cache=False` 可能早退 | `generate.py:259`、`transformer.py:1719-1728` | ❌ 未改 |
| **M7** | KV 逐 token `cat` 累积 O(L²) | `mixers.py:692-693,729` | ❌ 未改 |
| **M8** | IGMCG 评分把 prompt 计入均值 | `generate.py:175,268-288,337` | ❌ 未改（r42 igmcg 关） |
| **M9** | `_controller_past` 缺 batch 守卫 | `transformer.py:1745-1752`（对照 `:1767-1768` 有守卫） | ✅ 部分修（`is_fresh` 统一） |
| **M10** | `ngram_fusion` 语料缺失静默降级 | `checkpoint.py:78-84`、`transformer.py:733-736` | ✅ 已改 warning |
| **M11** | validate 按 batch 等权平均有偏 | `train.py:403-415` | ✅ 已接线（token 加权） |

### B.3 L（低危）/ N（新增）/ T（测试）

| 编号 | 一句话 | 位置 | 状态 |
|---|---|---|---|
| **L1** | `temperature<=0` 崩溃 | `sampling.py:51-54,82-83` | ✅ 已修 |
| **L2** | `temperature_applied` 判定两处不一致 | property `transformer.py:1229-1237` | ✅ 已修 |
| **L3** | 死代码 3 项 | `generate.py:21`、`transformer.py:1525`、`ngram.py:403` | ✅ 已清 |
| **L4** | 过期注释 4 处 | yaml:3、`train.py:261`、`transformer.py:1709`、`ngram.py:22` | ✅ 已修 |
| **L5** | `--igmcg-candidates 1` 温度静默 0.75× | `generate.py:373-376` | ✅ 已修 |
| **L6/L7** | chat.py 默认权重路径 + `--repetition-penalty` 缺失 | `chat.py:35-44`、README:45 | ✅ 已修 |
| **L8** | `clip_grad_norm_dml` 隐式 `.item()` | `train.py:222-228` | ✅ 已修（`use_foreach_norm_clip` 开关默认关） |
| **L9** | fallback 分支就地改 logits | `sampling.py:74` | ✅ 已修（`.clone()`） |
| **N1** | `temperature_applied` docstring 行号错（写 `:1700`，实为 `:1734`） | `transformer.py` property | ❌ 未改 |
| **N2** | `logprob_matrix` 死分支仍在 | `ngram.py:460` + 2 测试 | ❌ 未改 |
| **N3** | `reset_ngram_state` 漏 4 类状态 | `transformer.py:1524-1531` | ❌ 未改 |
| **N4** | 数据头截断吞 **92.22%** | `data_utils.py:44-45`（审查原标 `:41` 是循环头，**行号已漂**） | ❌ 未改（见 §4.4） |
| **N5** | Controller 慢 3.0×（11,264 dispatch、92.3% host 簿记） | `mixers.py:1265` 逐 t 循环 | ❌ 未改 |
| **T4** | parity 测试零初始化恒通过（空转） | `controller.py:149-158` | ❌ 测试缺陷未改 |
| **char_merge** | 第二处训推不一致：`F.pad` 左补零、T=1 缺滚动状态 | `layers.py:41-42`（修 `layers.py` + `transformer.py:1548`） | ✅ **已修**（开关 `char_merge_incremental_buffer`，代码默认 false、r42 未配；CPU 实测 `prefix−incremental` 24 序列 −0.0045→0、800 序列 −0.002340→**0.000000**） |

---

## 附录 C 本文件的来源与核对

| 节 | 来源 | 核对方式 |
|---|---|---|
| §1 定位 / §1.1 IGMCG | `README.md`、`docs/`、`scripts/generate.py`、`models/transformer.py` 全仓 grep | IGMCG 全称正则 0 命中 → 记为**查不到** |
| §2 数据流 | 初稿 A（`Temp\opencode\review\ARCH_A.md`） | 逐条回读 `transformer.py`/`controller.py`/`layers.py`/`train.py` |
| §3 模块表 | 初稿 A/B/C（`ARCH_A/B/C.md`） | 参数量由 `e2_params.py` CPU 实测；行号抽验 `controller.py:240-241`、`layers.py:41-42`、`mixers.py:1264-1265`、`train.py:278-279`、`data_utils.py:44-45` 全部吻合 |
| §4 训练流程 | 初稿 D（`ARCH_D.md`） | loss 6 项、Controller 状态机、WSD 三段、数据 9 步均复算 |
| §5 生成流程 | 初稿 C（`ARCH_C.md`） | 采样分支、`GEN_PARAMS`、三口径构造 |
| §6 演进史 | 旧版 S4（`review\S4.md`）§5 扩写 | 每条带 `CHANGELOG`/`AGENT_MEMORY` 出处 |
| §7 direction | 旧版 S4 §3/§4 | 数字复核自 `docs/H2_CAUSALIZATION_PLAN.md` |
| §8 我的看法 | 本人 | 全部标**推测** |
| 附录 A | 初稿 E（`ARCH_E.md`，116 schema 字段全覆盖） | CPU 构建实测 |
| 附录 B | 初稿 A4/B4/C5 合并 | 编号去重、状态以 `AGENT_MEMORY §11` 为准 |

**遗留未清**（初稿中明确标注"查不到/未查清"的）：① IGMCG 字母级全称；② Controller 与 Generator 输入表征不一致是否有意；③ Controller `mem_kv` 未来泄漏**无独立编号**（H2 指向 MemoryBank）；④ GDN 循环外提的 dispatch 收益未实测（方案已有，见附录 D-S1）；⑤ `train.py:475` 旧行号在审查报告中指什么（S5 已核：**只出现在 `RESULTS_ab.md` 4 处，6 份原始报告没有**，44 个历史版本逐一比对也无对应 → 汇总串行，真实位置为 `train.py:278-279` 与 `:202/:210/:216/:850`）；⑥ r42 那次训练的 stdout 日志仓库内没有，"epoch2 实际 OFF" 为代码状态机推导。

---

## 附录 D 说明书 × 子代理报告交叉复核

> 2026-09-29 执行：以本文件为准，逐条复核 `review\S5.md`（26 条重训分拣）与 `review\S1.md`（GDN 优化两条路线）。**不重跑子代理**。

### D.1 S5 — 26 条分拣 vs 本文件

| 分组 | 条目 | 与说明书核对结果 |
|---|---|---|
| **进（5 条 / 独立 4）** | 3_data #1 `tokens[:L]` 只留行首 64、#2 EOS 被切（全数据集仅 25 个 EOS）；5_freq #1 Controller 单边关；5_freq #2 = 6_optim B1 LR 每 epoch 重放 | **全部一致**：分别对应 §4.4 第 6/7 步、§4.2、§4.3。<br>⚠ **5_freq #1（M1）2026-09-30 改归「需要设计决定」**，是否仍"进"取决于该决定（见下行） |
| **需要设计决定（1 条）＝ 5_freq #1 / M1「Controller 分阶段开关」**（**2026-09-30 后改，S5 已跑完**） | **原分拣**：与上面 4 条并列归「明确 bug，改成一直联合训」。<br>**改后**：按 §1.3-3 设计意图，Controller 与主干"**有时联合训、有时分开训**"是**故意的**——问题不在"关"，而在**关法**：`train.py:278-279` 单边赋值无 else、唯一能开回的路径在 `validate:415`，靠副作用维持状态机。<br>**修法（不再写"改成一直联合训"）**：把训练日程做成**显式、可配置**——哪些批次联合训 / 哪些批次只训主干 / 要不要有只训 Controller 的阶段，而不是代码里这种半途关、靠 validate 副作用兜底的状态机。<br>**倾向方案（只记录不实施）**：逐步交替（一步带、一步不带），比例可配置（先 1:1 再 2:1），不带的步直接跳过 Controller 计算 | 依 §1.3-3（用户 2026-09-30 口述）。**§4.2 的事实描述原样保留不动**（状态机/行号/33% 联合步丢失都是事实，本轮不改）；本行只改**分类与修法**。<br>⚠ **S5 已跑完，本行是事后改的分类**——原 S5 报告已随 `Temp\opencode` 清理丢失，此处以本清单为准 |
| **明确 bug（4 条 / 独立 4）** | 3_data #1 `tokens[:L]` 只留行首 64、#2 EOS 被切（全数据集仅 25 个 EOS）；5_freq #2 = 6_optim B1 LR 每 epoch 重放；**6_optim A1 `label_smoothing` 前提错误**（原列 5 条含 5_freq #1，2026-09-30 移出 → 上行） | ⚠ **本文件已改**：§4.1 原写"与 ignore_index 冲突"（照抄代码注释）→ 更正为**前提是错的，torch 2.4.1 实测可开** |
| **待定（7 条）** | 3_data #3（vocab 12000/9186 死参数）、#6（merged_8k 只取 0.22% 最长行）、4_compute #1（开 chunk_scan）、6_optim A1/B3/B5/B6 | #3 → §3.1 死参数 720,384 已记；**#6 本文件未覆盖 → 补记为"语料选择性偏差，仅 3 行抽查推测，需全量包含性校验"（新发现，OUT_OF_SCOPE）**；4_compute #1 → §3 Controller 行已补"**chunk_scan=True 改数值、需重训**" |
| **不进（14 条）** | 3_data #4、4_compute #4/#13、5_freq #5/#7/#15、6_optim A2/B2/B4/B7/B8/B9/B10/B11 | **无冲突**；其中 #12（val 等权有偏）S5 已据 `train.py:872-873` 判"best 已改 token 加权，旧口径只入 history" → 对应 **M11 ✅**，与本文件一致 |

**S5 里本文件原本没记、现已知的 3 条（均属取舍、非 bug，不改本轮动作）**：B7 `gpas(0.5)×output_gate(0.5)` 双重乘性衰减初始残差只剩 0.25×；B9 `mem_kv=0` 仍非中性（零 k 参与 softmax，diff≈10%）与 `controller.py` "保证中性"注释不符；B10 零初始化使 Controller 主体首步梯度恒 0（1 步延迟）—— 与 **T4** 同源。

### D.2 S1 — GDN 优化两路线 vs 本文件

| 路线 | S1 结论 | 与说明书核对 |
|---|---|---|
| ① 时间步循环外提 **v2**（den 外提 −2089 + α/β 预 unsqueeze −252 + unbind 替切片 −2167 = **−4508，循环 8777→4269，−51.4%**） | **建议做**；v1/v2 与原实现**逐位 diff 0.0**【确定】；全模型 11264→≈6756（**−40%**）【推测 DML 分发结构同构】 | 对应本文件 **N5** 的解法。行号抽验：`mixers.py:1264/1265/1271/1272/1287` 全部吻合。**建议：保留为待办，不进本轮**（本轮主任务是 CharMerge） |
| ② `(I−A)⁻¹` 倍增替代 `solve_triangular` | **数学成立但否决**：(a) 只适用标准形，**当前 for-loop 含 Sᵀ 项非标准形，套用必改数值**【确定】；(b) dispatch 452 vs 485 只 −7%【确定】；(c) 算量 0.67 TFLOP = phase1 ×4400，估 +70–130ms > 全前向 125ms【推测】 | ✅ **与说明书一致**（§3 Controller 行） |
| ② 附带发现：Hillis-Steele 仿射对替 phase1 | **228 vs 485（−53%）**，同一半群结合律、**不改数值**【确定计数 / parity 推测 1e-5 内】 | 本文件**未记** → 补记为 N5 的**备选路线**（比 ① 收益小但同样安全） |

**⚠ 冲突核查结论：S1 与 S5 表面矛盾，实际不矛盾**——
- S5 #6 引 `mixers.py:1220-1222` 称 chunk_scan「与增量解码 cache parity 一致」，据此写"待定：先跑 parity，等价则进"；
- S1 引**同一句注释**的前半，称「与原 for-loop 数值不等」，据此判定"开启即改数值"。
- **读源码（`mixers.py:1219-1227`）确认：注释原文两句都在**——"改用标准形式，与原 for-loop 数值不等，但与增量解码 cache parity 一致"。语义为：chunk_scan ≠ for-loop（**训练数值改变 → 需重训**），但 chunk_scan ≡ 增量解码 cache 路径（**训推一致**）。
- **裁决：S5 的"待定"应升为"改数值 → 需重训"**；要跑的 parity 对拍对象是*增量解码*，不是现行 for-loop（对 for-loop 必然失败，S1 §4.2 已测 diff O(1)）。本文件 §3 Controller 行已按此写定。

### D.3 复核后本文件的改动清单

1. §4.1 `label_smoothing` 前提更正（S5 #14）。
2. §3 Controller GDN 行补 `chunk_scan=True` 改数值事实 + N5 挂 S1 路线①/HS 备选。
3. §附录 C 遗留⑤ 由"疑为串行"改为 S5 已证实的结论。
4. §附录 D 新增本节；3_data #6 语料选择性偏差记为新发现 OUT_OF_SCOPE。
5. **2026-09-30（第六批）**：新增 §1.3 设计意图（用户口述，只记录不实施）；D.1 的 5_freq #1 / M1 由「明确 bug」改归「**需要设计决定**」并改写修法（显式可配置训练日程，非"一直联合训"），注明"**S5 已跑完，此处后改**"。§4.2 事实描述**未动**。

