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

rem frontend\dist ships prebuilt in git; rebuilding here would dirty the checkout.
if exist frontend\dist\index.html (
  echo [INFO] web UI: prebuilt frontend\dist ^(rebuild after editing frontend\src: npm --prefix frontend run build^)
  goto :ui_done
)
where npm >nul 2>&1
if errorlevel 1 (
  echo [INFO] Node.js/npm not found: using the prebuilt web UI in frontend\dist.
  echo        To rebuild after editing frontend\src: npm --prefix frontend install ^&^& npm --prefix frontend run build
) else (
  echo [INFO] building the web UI ...
  call npm --prefix frontend install --no-audit --no-fund
  if errorlevel 1 ( echo [WARN] npm install failed: keeping the prebuilt web UI ) else ( call npm --prefix frontend run build )
)
:ui_done

echo.
echo [OK] installed. Next steps:
echo   .venv\Scripts\activate
echo   python main.py            then open http://127.0.0.1:52415
echo.
echo Optional extras:
echo   .venv\Scripts\python.exe -m pip install -r backend\requirements-dev.txt     ^(packaging/dev^)
echo   DirectML GPU: see backend\requirements-windows.txt ^(separate env only^)
endlocal
