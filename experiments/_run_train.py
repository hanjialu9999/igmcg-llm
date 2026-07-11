import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, 'scripts') not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import subprocess, sys, os

log = open("logs/pretrain.log", "w", buffering=1, encoding="utf-8")
p = subprocess.Popen(
    [sys.executable, "scripts/train.py", "--config", "configs/pretrain.yaml"],
    cwd=r"F:\Projects\新项目",
    stdout=log, stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,
    creationflags=0x00000008 | 0x00000200,  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
)
print("launched training pid:", p.pid)
