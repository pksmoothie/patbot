@echo off
setlocal
cd /d "%~dp0"
set "GM_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%GM_PYTHON%" set "GM_PYTHON=%~dp0..\..\.venv\Scripts\python.exe"
if not exist "%GM_PYTHON%" (
  echo Python environment not found. See README.md for setup commands.
  pause
  exit /b 1
)
"%GM_PYTHON%" refresh.py run
if errorlevel 1 pause
