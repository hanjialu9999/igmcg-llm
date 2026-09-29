# H2 记忆读侧未来泄漏 —— 因果化方案（只出方案，不改码）

> 状态：**方案稿**，等用户拍板后再动代码。本轮只做测量与设计。
> 关联缺陷：H1（Controller direction 非因果）、H3（ALiBi 未做 mem_cols 修正）。
> 依据：本轮新增的三口径基准 `baselines/r42_baseline.json` / `r42_baseline_ctrl_off.json`
> 与 24 序列信号消融（见 §2）。

## 1. 代码事实（读侧 / 写侧 / 逐层 / 训推不对称）

| 环节 | 位置 | 事实 |
|------|------|------|
| 读侧恒放行 | `models/mixers.py:832`（全量路径）、`models/mixers.py:1595-1599`（cache 路径） | `mask[..., :mem_cols] = False`：**所有** query 位置都能读前 `mem_cols` 列，永不施加因果惩罚 |
| 写侧一次写整段 | `models/transformer.py:429-430` | `memory.write(x)`，x 形状 (B, T, D)：槽里含 t+1..T-1 的信息 |
| Controller 写侧 | `models/controller.py:219-225` | 取**末层 S 态**（覆盖整段）经 `mem_query`/`mem_proj` 压成 `(mk, mv)`；`models/transformer.py:1550/1671` 注入 Generator，与 MemoryBank 在 `models/transformer.py:341-349` 合并进**同一段** mem_cols |
| 逐层放大 | 同上 | 层 i 的记忆列含层 0..i-1 对全部 T token 的写入（H2 原文） |
| 训推不对称 | `models/transformer.py:1513-1519` | 训练（use_cache=False）每 batch reset 记忆 → 层 0 恒读零槽；推理增量每步只写当前 token → 读到非零累积 |

> 注：r42 `memory_size=0`（MemoryBank 未启用），实际生效的只有 **Controller mem_kv 这一条**。

## 2. 实测量级（本轮新增）

### 2.1 全量 val（800 序列 / 51130 token，CPU fp32）

| 口径 | Controller=on | Controller=off |
|------|--------------|----------------|
| tf（全序列前向，validate 同路径） | **4.6673**（ppl 106.408） | 6.8774（ppl 970.122） |
| prefix（逐 t 喂前缀，无未来信息） | **5.2909**（ppl 198.528） | 6.8774（ppl 970.122） |
| incremental（use_cache 逐 token，真实生成路径） | **10.5258**（ppl 37265） | 6.8798（ppl 972.395） |

- Controller=off 时 **tf == prefix（差 0.0000）** → 因果掩码 + 卷积本身无泄漏。
- Controller=on 时 **tf − prefix = −0.6236 nats** → 泄漏存在，量级 0.62 nats（ppl 虚低 1.86×）。

### 2.2 信号消融（24 序列，定位泄漏来源）

| 变体 | tf | prefix | incremental | tf−prefix（泄漏） |
|------|-----|--------|-------------|------|
| 全开 | 4.6575 | 5.2572 | 10.4623 | **0.5997** |
| 关 direction | 5.2451 | 5.2486 | 5.7829 | **0.0035** |
| 关 mem_kv | 4.6714 | 5.2651 | 10.4695 | 0.5937 |
| 全关 | 6.7058 | 6.7058 | 6.7103 | 0.0000 |

全量 val 交叉验证：tf(direction 关) = **5.2951** ≈ prefix(全开) = 5.2909（差 0.0042）。

**结论**
1. **H2（mem_cols 恒放行 + 整段写入）在 r42 上实测泄漏 ≈ 0.0035–0.0042 nats（占总泄漏 0.7%）**——结构性真风险，当前权重下数值近中性（`mem_proj` 零初始化训练后仍近乎惰性）。
2. **99.3% 的泄漏来自 H1**：`controller.py:240` `x.mean(dim=1)` 是整段均值，非因果。
3. 增量路径崩坏（prefix→incremental 差 5.21 nats）中 **direction 占 4.67 nats（90%）**，film+mem 合计 0.53 nats，非 Controller 的增量边界 **0.0045 nats ≈ 0**。<s>（说明 KV cache + char_merge 的增量实现是正确的）</s> **⚠ 2026-09-29 推翻**：char_merge 增量实现**不正确**，根因 `models/layers.py:42`（T=1 只左补零、无跨步滚动状态），第二处训推不一致。0.0045 只是 **ctrl=off** 口径下的残差，ctrl=on+direction off 时同一项是 **0.5343**。**2026-09-30 已修**：`char_merge_incremental_buffer`（默认关），CPU 实测 `prefix−incremental` → **0.000000**，见 `docs/ARCHITECTURE.md` §2.2 ⚠DIFF-2 与 §11.10。
4. r42 未启用 MemoryBank → **H2 在 MemoryBank 配置上的真实量级尚未测**（见 §3-A）。

## 3. 方案（三层，按优先级）

### A. 先量化，不改码（成本最低，先做）

1. 用**启用 MemoryBank** 的既有配置（`memory_size>0`，如 config_full_dml 的 mem32）重跑 `scripts/baseline_eval.py` 三口径 → 得到 H2 在 MemoryBank 下的真实 nats。
2. 若 < 0.05 nats：H2 降级为"防御性修复"，排在 H1 之后；若显著：与 H1 同批修。
3. 产出物：`baselines/<config>_baseline.json`，与 r42 同格式可比。

### B. 因果化实现（下一次重训前，必须走 config 开关）

用户已定规则：**新 config 默认开，r42 config 保持旧行为**（开关建议名 `causal_memory: bool = False`，进 `model_config.py`）。三个可选实现：

**B1 逐位置前缀状态（推荐）**
- 记忆列语义从"整段摘要"改为"位置 t 的状态"：query t 只读 mem[t]。
- MemoryBank：`memory.write(x)` 改为产出 `(B, T, M, D)` 的前缀扫描结果（项目已有 `_parallel_prefix_scan`，GatedDeltaNet 亦在用）。
- Controller：`controller.py:219-225` 从"末层单一 S"改为"每位置 S"→ `mem_kv` 变 `(B, T, M, ·)`。
- 显存代价：r42 规模 (B=24, T=64, M=4, D=256) ≈ 6 MB，可忽略。
- 同时自动消除 §1 的"训推不对称"（训练读前缀状态、推理累积状态，两者的 t 定义一致）。

**B2 读侧因果遮蔽（改动最小，但要求写侧可定位）**
- 把 `mixers.py:832` 从"恒放行"改为"列 m 只对 `t >= src(m)` 放行"。
- **前提**：每个记忆槽有确定的来源位置。整段摘要没有单一来源位置 → 必须先改写侧为"每槽负责一段/一位置"，否则本方案不可用。
- 单独用在 Controller 的 `direction`/`film` 这类无位置语义的信号上无效（它们不在 mem_cols 里）。

**B3 训练-推理一致化（最省算力，语义最保守）**
- 训练也按增量语义喂记忆（每步只写当前 token），用前缀扫描并行化 → 数学上等价于 B1 的特例。
- 优点：训练与生成完全同分布；缺点：训练侧算力 +1 次扫描。

**明确否决的选项**
- 只改评测口径不改训练：模型继续按泄漏目标训练，与 B 互斥，只能当过渡。
- 只改 `mixers.py:832` 一行而不改写侧：见 B2 前提，整段摘要下会把"读未来"变成"读不到记忆"，训练/推理同时退化。

### C. 与 H1/H3 的合并顺序（单一事实来源）

H1、H2、H3 都作用在**同一段 mem_cols 的位置语义**上，必须共用一套定义，否则修一个偏一个：

1. 先定"记忆列位置语义"（B1：mem[t] 表示"到 t 为止的状态"）。
2. H2 按该语义实现（读侧/写侧）。
3. H3（`mixers.py:459-461` ALiBi 距离按同一语义做 mem_cols 还原）。
4. H1（`controller.py:240` direction 因果化：滚动均值 / last-token / 训练端逐 token 三选一）同批进同一次重训。

## 4. 验收标准（硬指标）

1. **三口径一致**：`causal_memory: true` 下 `eval_tf == eval_prefix == eval_incremental`，|Δ| < 1e-3 nats。
2. **对抗测试**：构造"记忆/信号里埋未来标签"的样本，断言 logits 不随未来标签变化（新增 `tests/test_causal_memory.py`）。
3. **旧权重回归**：开关 off 时 r42 三口径复现 **4.6673 / 5.2909 / 10.5258**（±1e-6），证明开关不改变旧行为。
4. **速度回归**：训练 step 耗时 ≤ +10%（B1 的额外张量与扫描）；同时用 `scripts/_profile_train_steps.py` 复测。
5. 三口径基准 JSON 进 `baselines/`，作为下一次重训的可比锚点。

## 5. 成本与风险

| 项 | 需重训 | 风险 | 预期收益（r42 实测） |
|----|--------|------|--------------------|
| H2 因果化 | 是 | 中（改记忆路径，影响所有记忆类 config） | 泄漏 −0.004 nats；消除结构风险与训推不对称 |
| H1 direction 因果化 | 是 | 中（改控制信号语义） | 泄漏 −0.62 nats；生成路径 −4.68 nats |
| H3 ALiBi mem_cols | 是 | 低（一行） | 位置先验修正（parity 测不出，需三口径验证） |

> 单修 H2 的数值收益极小，**必须与 H1 同批**才有意义——这是本轮测量给出的最重要排序结论。
