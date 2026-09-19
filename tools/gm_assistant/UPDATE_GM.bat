@echo off
setlocal
cd /d "%~dp0"

call REFRESH_GM.bat
if errorlevel 1 (
  echo GM update stopped because the refresh did not complete successfully.
  exit /b 1
)

set "GM_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%GM_PYTHON%" set "GM_PYTHON=%~dp0..\..\.venv\Scripts\python.exe"
if not exist "%GM_PYTHON%" (
  echo Python environment not found. See README.md for setup commands.
  exit /b 1
)

"%GM_PYTHON%" gm_packet.py
if errorlevel 1 (
  echo GM update stopped because the packet could not be generated.
  exit /b 1
)

call RUN_GM.bat
