@echo off
setlocal
cd /d "%~dp0"
title PatBot - Update and Run

echo ==========================================
echo PatBot - Update and Run
echo ==========================================
echo.

if not exist ".venv\Scripts\python.exe" (
  echo PatBot has not been set up on this PC yet.
  echo Running SETUP_ONCE.bat instead...
  call SETUP_ONCE.bat
  exit /b %errorlevel%
)

where git >nul 2>&1
if errorlevel 1 (
  echo ERROR: Git was not found in PATH.
  echo Install Git for Windows, then run this file again.
  pause
  exit /b 1
)

echo [1/4] Pulling the latest PatBot code from GitHub...
git pull --ff-only origin main
if errorlevel 1 goto :fail

echo [2/4] Syncing dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt -q
if errorlevel 1 goto :fail

echo [3/4] Running tests...
".venv\Scripts\python.exe" -m pytest -q
if errorlevel 1 goto :fail

echo [4/4] Refreshing live player data...
".venv\Scripts\python.exe" refresh_data.py
if errorlevel 1 (
  echo.
  echo WARNING: Live-data refresh failed. Launching PatBot with the most recent local snapshot.
  echo.
)

echo.
echo Launching PatBot...
".venv\Scripts\python.exe" -m streamlit run app.py
exit /b 0

:fail
echo.
echo ==========================================
echo UPDATE FAILED - PatBot was not launched.
echo Send the error above to ChatGPT.
echo ==========================================
pause
exit /b 1
