@echo off
setlocal
python main.py verify
if errorlevel 1 exit /b %errorlevel%
endlocal
