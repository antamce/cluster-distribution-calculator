@echo off
setlocal
rem PowerShell performs robust Conda and environment discovery, including custom
rem installation roots and custom envs_dirs. This file remains the double-click
rem entry point used by the Windows desktop shortcut.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch_synpo.ps1"
if errorlevel 1 (
    echo.
    echo Synpo did not start. Review the message above for recovery instructions.
    pause
)
endlocal
