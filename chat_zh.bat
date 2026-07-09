@echo off
chcp 65001 >nul
cd /d F:\Projects\新项目
F:\Projects\.amd_venv\Scripts\python.exe scripts\chat.py --model checkpoints_test\final_model.pt --vocab checkpoints_test\vocab.json --device auto
echo.
echo [提示] 若本窗口中文显示乱码，请打开 logs\chat_history.txt（UTF-8）查看正确中文。
pause
