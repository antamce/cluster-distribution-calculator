@echo off
setlocal
set "SYNPO_PROJECT_ROOT=%~dp0"
set "PYTHONPATH=%SYNPO_PROJECT_ROOT%src"

set "SYNPO_ENV_PYTHON=%USERPROFILE%\miniconda3\envs\synpo-microscopy\python.exe"
if not exist "%SYNPO_ENV_PYTHON%" set "SYNPO_ENV_PYTHON=%USERPROFILE%\anaconda3\envs\synpo-microscopy\python.exe"

if not exist "%SYNPO_ENV_PYTHON%" (
    echo Synpo environment Python was not found.
    echo Expected environment name: synpo-microscopy
    echo Recreate it from this project with: conda env create -f environment.yml
    pause
    exit /b 1
)

"%SYNPO_ENV_PYTHON%" -m synpo
if errorlevel 1 pause
endlocal
