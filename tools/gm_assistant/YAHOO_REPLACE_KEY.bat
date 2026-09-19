@echo off
setlocal
cd /d "%~dp0"
set "GM_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%GM_PYTHON%" set "GM_PYTHON=%~dp0..\..\.venv\Scripts\python.exe"
if not exist "%GM_PYTHON%" (
  echo Python environment not found. See YAHOO.md.
  exit /b 1
)
"%GM_PYTHON%" yahoo_client.py replace-key %*
exit /b %errorlevel%
