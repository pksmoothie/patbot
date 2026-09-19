@echo off
setlocal
cd /d "%~dp0"
set "GM_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%GM_PYTHON%" set "GM_PYTHON=%~dp0..\..\.venv\Scripts\python.exe"
if not exist "%GM_PYTHON%" (
  echo Python environment not found. See tools\gm_assistant\README.md for setup commands.
  pause
  exit /b 1
)
"%GM_PYTHON%" -m streamlit run dashboard.py --server.address 127.0.0.1 --server.port 8502
if errorlevel 1 pause
