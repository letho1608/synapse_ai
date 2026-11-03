@echo off
title Synapse AI
echo Web UI: http://127.0.0.1:52415  (opens when the node is ready)
start "" /b cmd /c "timeout /t 8 >nul & start http://127.0.0.1:52415"
python main.py %*
pause
