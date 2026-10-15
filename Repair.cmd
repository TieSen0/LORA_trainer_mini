@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\manage.ps1" -Action Repair
set "result=%errorlevel%"
pause
exit /b %result%
