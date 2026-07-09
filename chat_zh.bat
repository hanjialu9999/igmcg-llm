@echo off
set "ROOT=%~dp0"
"F:\Projects\.amd_venv\Scripts\python.exe" "%ROOT%scripts\chat.py" --model "%ROOT%checkpoints_test\final_model.pt" --vocab "%ROOT%checkpoints_test\vocab.json" --device auto
echo.
echo [Tip] If Chinese is garbled here, open logs\chat_history.txt (UTF-8) to see correct Chinese.
pause
