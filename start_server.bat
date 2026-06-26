@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "KMP_DUPLICATE_LIB_OK=TRUE"
set "PATH=%CD%\.runtime\Scripts;%PATH%"

echo Starting AI interview system...
echo Home: http://127.0.0.1:28080/
echo Live: http://127.0.0.1:28080/live
echo.

".runtime\Scripts\python.exe" main.py
