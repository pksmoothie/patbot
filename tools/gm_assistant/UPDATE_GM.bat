@echo off
setlocal
cd /d "%~dp0"

call REFRESH_GM.bat
if errorlevel 1 (
  echo GM update stopped because the refresh did not complete successfully.
  exit /b 1
)

call RUN_GM.bat
