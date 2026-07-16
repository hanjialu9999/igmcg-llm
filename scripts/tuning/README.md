# scripts/tuning/

生成质量参数扫描。

| 文件 | 说明 |
|------|------|
| `tune_temperature.py` | 固定 `top_k` / `repetition_penalty`，扫描 `temperature`。 |
| `tune_topk.py` | 固定 `temperature` / `repetition_penalty`，扫描 `top_k`。 |
| `showcase_optimal_params.py` | 汇总扫描结果，将最优 `temperature` / `top_k` / `repetition_penalty` **回写**到仓库根的 `chat_config.json`，供 `tools/dialogue_interactive.py` 直接使用（`scripts/chat.py` 仍走命令行参数）。 |

> 推荐区间：temperature 0.6–0.8、top_k 40–50、repetition_penalty 1.4。

## 用法

```bash
# 默认读取 checkpoints/final_model.pt 与 checkpoints/vocab.json
python scripts/tuning/tune_topk.py

# 指定权重/词表与设备（如 DML 推理）
python scripts/tuning/tune_topk.py \
    --model archive_unused/checkpoints_backup/_stab_ckpt/final_model.pt \
    --vocab checkpoints_dml/vocab.json \
    --device dml
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `--model` | 模型权重路径 | `checkpoints/final_model.pt` |
| `--vocab` | 词表路径 | `checkpoints/vocab.json` |
| `--device` | 推理设备（`cpu` / `cuda` / `dml`，默认自动） | 自动 |
