@echo off
setlocal
cd /d "%~dp0"
title PatBot - Fast Draft Refresh

echo ==========================================
echo PatBot - Fast Draft-Day Injury/News Refresh
echo ==========================================
echo.

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: PatBot virtual environment not found.
  echo Run SETUP_ONCE.bat first.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" fast_draft_refresh.py
if errorlevel 1 (
  echo.
  echo FAST REFRESH FAILED. Send the error above to ChatGPT.
  pause
  exit /b 1
)

echo.
echo Refresh complete. Return to the PatBot browser window and trigger any rerun.
pause
