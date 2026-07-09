@echo off
chcp 65001 >nul
setlocal
REM =====================================================================
REM  一键启动器：自动选择虚拟环境，无需手动 activate
REM    - 优先 .amd_venv  （AMD / Intel 核显，Windows + DirectML）
REM    - 其次 .my_venv    （NVIDIA CUDA / 仅 CPU 的环境）
REM    - 最后回退系统 python
REM  用法：
REM    run.bat            -> 交互菜单
REM    run.bat train      -> 训练语言模型
REM    run.bat finetune   -> 微调
REM    run.bat chat       -> 对话
REM    run.bat gen "问题"  -> 单条生成
REM =====================================================================
if exist ".amd_venv\Scripts\python.exe" (
    set "PY=.amd_venv\Scripts\python.exe"
) else if exist ".my_venv\Scripts\python.exe" (
    set "PY=.my_venv\Scripts\python.exe"
) else (
    set "PY=python"
)

if "%1"=="" (
    %PY% run.py
) else if /i "%1"=="train" (
    %PY% scripts/train.py --config config/config.yaml
) else if /i "%1"=="finetune" (
    %PY% train_finetune.py
) else if /i "%1"=="chat" (
    %PY% chat.py
) else if /i "%1"=="gen" (
    %PY% scripts/generate.py --prompt "%2" --device auto
) else (
    %PY% %*
)
