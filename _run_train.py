import subprocess, sys, os

log = open("logs/pretrain.log", "w", buffering=1, encoding="utf-8")
p = subprocess.Popen(
    [sys.executable, "scripts/train.py", "--config", "config/pretrain.yaml"],
    cwd=r"F:\Projects\新项目",
    stdout=log, stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,
    creationflags=0x00000008 | 0x00000200,  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
)
print("launched training pid:", p.pid)
