@echo off
setlocal
cd /d "%~dp0"
title PatBot

if not exist ".venv\Scripts\python.exe" (
  echo PatBot has not been set up on this PC yet.
  echo Running SETUP_ONCE.bat...
  call SETUP_ONCE.bat
  exit /b %errorlevel%
)

".venv\Scripts\python.exe" -m streamlit run app.py
