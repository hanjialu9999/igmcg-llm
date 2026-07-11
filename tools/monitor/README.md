# tools/monitor/

训练过程与资源监控。

| 文件 | 说明 |
|------|------|
| `monitor_training.py` | 监控训练进度（loss / 步速等）。 |
| `monitor_gpu_training.py` | 监控 GPU 显存与利用率。 |
| `monitor_live.py` | 实时（动态刷新）监控。 |
| `simple_monitor.py` | 轻量监控。 |

> 在训练脚本后台运行时配合查看，便于及时发现异常或资源瓶颈。
