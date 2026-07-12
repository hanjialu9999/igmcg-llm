"""后台启动训练（脱离父进程，日志写入 logs/pretrain.log）。

用法：python experiments/_run_train.py
依赖：需在含 torch 的虚拟环境中运行（venv 解析逻辑见 run.bat / chat_zh.bat）。
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, 'scripts') not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import subprocess

log = open("logs/pretrain.log", "w", buffering=1, encoding="utf-8")
p = subprocess.Popen(
    [sys.executable, "scripts/train.py", "--config", "configs/pretrain.yaml"],
    cwd=_ROOT,
    stdout=log, stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,
    creationflags=0x00000008 | 0x00000200,  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
)
print("launched training pid:", p.pid)
