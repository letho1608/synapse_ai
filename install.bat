@echo off
REM Synapse AI installer (Windows). Thin wrapper, no new runtime.
REM Installs the default node closure (base + Windows-only wheels).
setlocal
cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
  echo [FAIL] python not found on PATH. Install Python 3.10+ first.
  exit /b 1
)
for /f "tokens=2" %%v in ('python -c "import sys; print(sys.version.split()[0])"') do set PYVER=%%v
echo [INFO] python %PYVER%

if not exist ".venv\Scripts\python.exe" (
  echo [INFO] creating .venv ...
  python -m venv .venv
  if errorlevel 1 exit /b 1
)

echo [INFO] pip install -r backend\requirements.txt ...
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
if errorlevel 1 (
  echo [FAIL] pip install failed. See output above.
  echo        "No such file or directory" under torch\include = Windows 260-char path limit:
  echo        enable long paths ^(https://pip.pypa.io/warnings/enable-long-paths^) or clone to a shorter folder.
  exit /b 1
)

echo [INFO] torch backend detect:
.venv\Scripts\python.exe -c "import torch; print('  torch', torch.__version__, '| cuda:', torch.cuda.is_available())"

echo.
echo [OK] installed. Next steps:
echo   .venv\Scripts\activate
echo   python main.py
echo.
echo Optional extras:
echo   .venv\Scripts\python.exe -m pip install -r backend\requirements-dev.txt     ^(packaging/dev^)
echo   DirectML GPU: see backend\requirements-windows.txt ^(separate env only^)
endlocal
