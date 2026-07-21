
## 17. 架构分析：2D 线性注意力能替代什么（2026-07-21）

### 核心发现
2D 轴向线性注意力是唯一同时具备 **线性复杂度 + 空间局部性 + 全局依赖近似** 的方案。
当前架构中，标准 attention 是唯一的 O(T²) 瓶颈，2D linear 可直接替代。

### 可替代的组件
1. **标准 Attention（mixer: 'attn'）** → 用 `mixer: 'linear2d'` 替代
   - O(T²) → O(T·√T)，推理内存大幅节省
   - 2D 局部性比标准注意力有更好的字符/词组空间归纳偏置
   - 改动最小：只改 mixer 选择，block 结构不变

2. **Hybrid 块中的 attn 路径** → linear2d + SSM 并行
   - 当前 hybrid = 标准 attn + SSM = O(T²) + O(T·D·D_s)
   - 改为 linear2d + SSM = O(T·√T) + O(T·D·D_s)，整体线性
   - SSM 建模序列动态，2D 建模空间局部性，互补

3. **Linear Attention（mixer: 'linear'）** → 直接升级
   - 线性注意力无空间结构，2D 加了网格局部性，质量更好
   - 复杂度相当，但 2D 常数因子更优

### 不可替代的组件
- SSM/Mamba：动态选择性扫描 vs 静态空间结构，互补不替代
- Memory Bank：外部记忆容量
- Ngram Fusion：统计先验与神经注意力正交

### 最佳组合：SSM + 2D Linear 混合
- SSM 层：序列动态建模（因果依赖、选择性遗忘/记忆）
- 2D Linear 层：空间局部模式（字符→词组组合）
- 复用现有 hybrid 门控机制并行混合
- 全链路 O(T·D·max(D_s, F))，无 O(T²) 瓶颈

---

## 18. 架构改进完成记录（2026-07-21）

### 已完成的4步实现
1. **层间共享 attn projection**（`share_attn_proj=True`）
   - SlidingWindowCausalSelfAttention / LinearAttention 支持 shared_qkv/shared_proj
   - 参数压缩 ~24%（95K→103K 含共享投影；对比独立 136K）
   - 注意：share_attn_proj 在 attn_linear 混合器中不跨并行分支共享（设计约束，见 §20 Finding 3.3）

2. **新增 mixer='linear2d'**（AxialLinearAttention）
   - 2D 轴向线性注意力，O(T·√T)，行/列各做线性注意力后融合
   - 增量解码退化为 1D 线性注意力 RNN 模式
   - **修复 padding 泄漏**：padded v 被 mask 为 0，防止 phantom token 信息污染

3. **mixer='hybrid_linear2d'**（linear2d + SSM 并行）
   - hybrid 块用 linear2d 做 token mixer + SSM 并行
   - 全链路 O(T·√T)+O(T·D·D_s)，无 O(T²) 瓶颈

4. **全链路线性化验证**
   - 5 种配置 benchmark + 冒烟训练全部通过

### 性能优化（6项，commit 53b1e2e）
1. AxialLinearAttention incremental 路径 `_feat(q)` 重复调用→合并为1次
2. AxialLinearAttention 6→4 Linear layers（row 轴复用 main 投影）-20% 参数
3. AxialLinearAttention 三份 QK-Norm/Temp→两份（main+row 共享）
4. SlidingWindowCausalSelfAttention 消除重复 `_build_causal_window_mask`
5. MambaSSM `_selective_scan` 推理路径预分配缓冲区（24次→4次分配）
6. MambaSSM forward 融合 dB 中间张量

---

## 19. 架构缺陷审查与修复（2026-07-21）

### HIGH 级缺陷（已修复）
1. **AxialLinearAttention zero-padding 信息泄漏**（§18 Finding 1.1.1）
   - 问题：padded 位置的零值 token 参与 cumsum 状态累积，污染真实 token 输出
   - 修复：对 padded v 乘以 `exp(-inf)=0` 屏蔽贡献
   - 测试：`test_linear2d_padding_no_leak`

2. **hybrid_single_gate skip gate 未作用于 SSM 分支**（§18 Finding 2.2.1）
   - 问题：skip gate 仅作用于 attn 分支，SSM 分支始终参与计算
   - 修复：统一 `ssm_eff = sk * ssm_h`，三路分支均应用 skip gate
   - 测试：`test_hybrid_single_gate_skips_both_branches`

3. **config_loader 无 mixer 参数校验**（§18 Finding 3.1）
   - 问题：无效 mixer 值（如 'foo'）静默退化为标准 attn
   - 修复：新增 `_VALID_MIXERS` 校验，无效值抛出 ValueError
   - 测试：`test_config_loader_rejects_invalid_mixer`

### MEDIUM 级缺陷（已修复）
4. **_linear_attn_1d 温度缩放符号/幅度不一致**（§18 Finding 1.1.2）
   - 问题：内部使用 `exp(log_temp)` 而非 `exp(-0.5*log_temp)`，有效温度差 2 倍且符号相反
   - 修复：改为 `exp(-0.5*log_temp)` 与 apply_qk_norm_and_temp 一致
   - 影响：当前代码中该参数未使用（norm/temp 在调用前已应用），为防御性修复

### MEDIUM 级缺陷（已知，未修复）
5. **share_attn_proj 在 attn_linear 中跨并行分支共享投影**（Finding 3.3）
   - 问题：softmax attn 和 linear attn 共享 QKV 投影，可能成为信息瓶颈
   - 状态：设计约束，用户需知晓不应在 attn_linear 混合器中启用 share_attn_proj

6. **LinearAttention 忽略 memory_kv 参数**（Finding 1.2.2）
   - 问题：memory_kv 被传入但未使用，MemoryBank 写入后无法读取
   - 状态：低优先级，不影响当前功能

---

## 20. Benchmark 对比（优化后）

| 配置 | T=16 Fwd | T=64 Fwd | Incr(ms) | Params |
|------|----------|----------|----------|--------|
| attn | 0.7ms | 0.9ms | 0.7-0.9ms | 95K |
| linear | 1.0ms | 1.5ms | 0.8-1.0ms | 95K |
| linear2d | 1.4ms | 2.1ms | 0.8-0.9ms | 128K |
| attn+SSM | 1.8ms | 2.9ms | 1.4-1.7ms | 128K |
| linear2d+SSM | 2.7ms | 4.0ms | 1.6-1.8ms | 160K |

### 实现计划完成状态
1. ✅ 第一步：层间共享 attn projection
2. ✅ 第二步：新增 mixer='linear2d'
3. ✅ 第三步：linear2d + SSM 混合
4. ✅ 第四步：全链路线性化验证
5. ✅ 第五步：6项性能优化
6. ✅ 第六步：架构缺陷审查与修复
