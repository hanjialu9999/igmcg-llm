# 调参指南 (Tuning Guide)

生成质量主要受三个采样参数影响，均在 `chat_config.json` 中配置。

| 参数 | 含义 | 经验范围 | 本项目推荐 |
|------|------|----------|------------|
| `temperature` | 采样温度，越高越随机 | 0.3 – 1.2 | 0.65 |
| `top_k` | 仅从概率最高的 k 个 token 采样 | 25 – 60 | 40 |
| `repetition_penalty` | 重复惩罚，>1 抑制复读 | 1.5 – 2.5 | 2.0 |

辅助参数：`min_new_tokens`（最小生成长度）、`max_new_tokens`（最大生成长度）、
`context_rounds`（多轮对话上下文轮数）。

## 系统化扫描

```bash
# 固定 top_k=42 / rep=2.0，扫描 temperature
python scripts/tuning/tune_temperature.py

# 固定 temperature=0.65 / rep=2.0，扫描 top_k
python scripts/tuning/tune_topk.py
```

## 展示最优参数

```bash
python scripts/tuning/showcase_optimal_params.py
```

该脚本会用实验得到的最优 `temperature / top_k / repetition_penalty` **合并回写**
到 `chat_config.json`，供 `dialogue_interactive.py` 与 `scripts/chat.py` 直接使用。

## 调参建议

- **过低温度 / 过小 top_k**：deterministic，容易复读、缺乏多样性。
- **过高温度 / 过大 top_k**：更富创造性，但可能出现不连贯或乱码。
- 推荐在 **temperature 0.6–0.8、top_k 40–50** 区间做小步扫描后再定。
