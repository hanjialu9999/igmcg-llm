# scripts/tuning/

生成质量参数扫描。

| 文件 | 说明 |
|------|------|
| `tune_temperature.py` | 固定 `top_k` / `repetition_penalty`，扫描 `temperature`。 |
| `tune_topk.py` | 固定 `temperature` / `repetition_penalty`，扫描 `top_k`。 |
| `showcase_optimal_params.py` | 汇总扫描结果，将最优 `temperature` / `top_k` / `repetition_penalty` **回写**到 `chat_config.json`，供 `scripts/chat.py` 与 `dialogue_interactive.py` 直接使用。 |

> 推荐区间：temperature 0.6–0.8、top_k 40–50、repetition_penalty 1.4。
