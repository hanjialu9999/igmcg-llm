@echo off
setlocal
chcp 65001 >nul
set "ROOT=%~dp0"

REM 解析 Python 解释器（可移植：不写死绝对路径）
REM   优先级：IGMCG_PYTHON 环境变量 > 项目内 .amd_venv > 上一级 .amd_venv
REM           > 项目内 .my_venv > 上一级 .my_venv > 系统 PATH 上的 python
if defined IGMCG_PYTHON (
    set "PY=%IGMCG_PYTHON%"
) else if exist "%ROOT%.amd_venv\Scripts\python.exe" (
    set "PY=%ROOT%.amd_venv\Scripts\python.exe"
) else if exist "%ROOT%..\.amd_venv\Scripts\python.exe" (
    set "PY=%ROOT%..\.amd_venv\Scripts\python.exe"
) else if exist "%ROOT%.my_venv\Scripts\python.exe" (
    set "PY=%ROOT%.my_venv\Scripts\python.exe"
) else if exist "%ROOT%..\.my_venv\Scripts\python.exe" (
    set "PY=%ROOT%..\.my_venv\Scripts\python.exe"
) else (
    set "PY=python"
)

if "%1"=="" (
    "%PY%" "%ROOT%scripts\chat.py"
) else if /i "%1"=="train" (
    "%PY%" "%ROOT%scripts\train.py" --config "%ROOT%configs\pretrain.yaml"
) else if /i "%1"=="finetune" (
    "%PY%" "%ROOT%train_finetune.py"
) else if /i "%1"=="chat" (
    "%PY%" "%ROOT%scripts\chat.py"
) else if /i "%1"=="gen" (
    "%PY%" "%ROOT%scripts\generate.py" --prompt "%2" --device auto
) else (
    "%PY%" %*
)
