# experiments/

实验 / 诊断 / 一次性脚本目录。

## 当前状态

历史上的 `_*.py` 一次性实验脚本已在仓库整理中清理（见 git 历史 commit `b19bbd2` / `e4dd90f`）。
目录保留，供后续新实验脚本使用。

## 历史脚本（已删除，可从 git 历史恢复）

| 文件 | 说明 |
|------|------|
| `_show.py` | 展示 IGMCG 联合解码的生成效果（输出到 `logs/show.txt`）。 |
| `_diag_igmcg.py` | IGMCG 碎片化诊断：对比修改前后的连贯度指标。 |
| `_compare_stab.py` · `_compare_decode.py` | 对比不同模型 / 不同解码策略的生成质量与速度。 |
| `_ngram_test.py` | n-gram 先验（uni/bi/tri 插值）验证。 |
| `_detect_vocab.py` | 词表检测与匹配（定位正确的 `vocab.json`）。 |
| `_inspect_archives.py` · `_test_archives.py` | 检查 `archive_unused/` 中各类归档检查点是否可加载。 |
| `_combined_full.py` · `_combined_demo.py` | n-gram + IGMCG 联合解码完整 / 演示流程。 |
| `_dml_repro.py` · `_dml_ssm_diag.py` · `_dml_hybrid_diag.py` · `_dml_hybrid_small.py` | DML / SSM / hybrid 训练复现与失败诊断。 |
| `_hybrid_smoke.py` · `_hybrid_train_smoke.py` | hybrid 架构冒烟测试。 |
| `_run_train.py` · `_run_train_cpu.py` | 便捷启动训练的封装。 |
| `_gen_opt_test.py` | 推理优化点验证（KV-cache 一致性、速度对比、实际生成）。 |
| `_bench_enh.py` · `_bench_speed.py` | 增强 / 速度基准微基准。 |
| `_cmp_sel_full.py` | SEL 全量受控对比复现脚本（产物 `cmp_sel_full.txt`）。 |
| `_dl_c4.py` | C4 语料下载相关。 |

> 这些不是项目主流程的一部分，仅用于探索与验证；产物通常写到 `logs/`。
> 如需复用，可从对应 commit 的 git 历史中恢复。
