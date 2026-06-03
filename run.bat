@echo off
title Synapse AI Launcher
color 0A

echo ===================================================
echo               SYNAPSE AI LAUNCHER                  
echo ===================================================
echo Starting Synapse Server (Backend + Frontend)...
python main.py

echo.

echo.
echo Both servers are starting in the background of this window.
echo - Backend will run on its configured API port (default: 52415)
echo - Frontend will run on Vite's dev port (default: 5173)
echo.
echo Press Ctrl+C to stop both servers.
pause
