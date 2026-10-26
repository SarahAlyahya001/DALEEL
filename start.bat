@echo off
rem Sets up DALEEL on the first start and starts it: double-click this file (see README.md).
if not exist "%~dp0scripts\start.ps1" (
    echo DALEEL cannot start from inside the ZIP file. Extract the ZIP file first: right-click it,
    echo choose Extract All, then open the extracted folder and double-click start.bat there.
    pause
    exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start.ps1" %*
if errorlevel 1 pause
