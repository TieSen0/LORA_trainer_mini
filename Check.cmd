@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\manage.ps1" -Action Check
set "result=%errorlevel%"
pause
exit /b %result%
