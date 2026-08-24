@echo off
setlocal
cd /d "%~dp0"
title PatBot - One-Time Setup

echo ==========================================
echo PatBot - One-Time Windows Setup
echo ==========================================
echo.

where py >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python launcher ^("py"^) was not found.
  echo Install Python, then run this file again.
  pause
  exit /b 1
)

where git >nul 2>&1
if errorlevel 1 (
  echo WARNING: Git was not found in PATH.
  echo PatBot can run after setup, but UPDATE_AND_RUN.bat will need Git.
  echo Install Git for Windows before using automatic updates.
  echo.
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/5] Creating PatBot virtual environment...
  py -m venv .venv
  if errorlevel 1 goto :fail
) else (
  echo [1/5] Existing virtual environment found.
)

echo [2/5] Installing / syncing Python packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

if not exist ".env" (
  echo [3/5] Creating local .env file from template...
  copy /Y ".env.example" ".env" >nul
  echo.
  echo IMPORTANT: A local .env file has been created.
  echo Put your FantasyPros API key on the FANTASYPROS_API_KEY= line.
  echo This file is ignored by Git and will stay on this PC.
  echo.
  start "" notepad ".env"
  echo Save and close Notepad, then return here.
  pause
) else (
  echo [3/5] Existing local .env found. Leaving it untouched.
)

echo [4/5] Running PatBot tests...
".venv\Scripts\python.exe" -m pytest -q
if errorlevel 1 goto :fail

echo [5/5] Refreshing live player data...
".venv\Scripts\python.exe" refresh_data.py
if errorlevel 1 (
  echo.
  echo WARNING: Live-data refresh failed. PatBot can still launch using the last snapshot or example data.
  echo.
)

echo.
echo Setup complete. Launching PatBot...
".venv\Scripts\python.exe" -m streamlit run app.py
exit /b 0

:fail
echo.
echo ==========================================
echo SETUP FAILED - see the error above.
echo ==========================================
pause
exit /b 1
