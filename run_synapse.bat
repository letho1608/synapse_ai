@echo off
title Synapse AI Launcher
color 0A

echo ===================================================
echo               SYNAPSE AI LAUNCHER                  
echo ===================================================
echo.
echo Starting Backend Server...
start "Synapse AI Backend" cmd /k "cd backend && set PYTHONPATH=. && python synapse/main.py"

echo Starting Frontend Dev Server...
start "Synapse AI Frontend" cmd /k "cd frontend && npm install && npm run dev"

echo.
echo Both servers are starting in new command prompt windows.
echo - Backend will run on its configured API port (default: 52415)
echo - Frontend will run on Vite's dev port (default: 5173)
echo.
echo You can close this window now.
pause
