@echo off
title Synapse AI Launcher
color 0A

echo ===================================================
echo               SYNAPSE AI LAUNCHER                  
echo ===================================================
echo Starting Synapse Server (Backend, web UI served by backend, no Node.js needed)...
python main.py

echo.

echo.
echo Backend is starting in this window.
echo - Backend API on its configured port (default: 52415)
echo - Web UI is a static page served by the backend itself (GET /); no Vite/Node.js.
echo.
echo Press Ctrl+C to stop the server.
pause
