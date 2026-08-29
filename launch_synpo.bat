@echo off
setlocal
set "SYNPO_PROJECT_ROOT=%~dp0"
set "PYTHONPATH=%SYNPO_PROJECT_ROOT%src"

set "SYNPO_CONDA_ROOT=%USERPROFILE%\miniconda3"
set "SYNPO_ENV_ROOT=%SYNPO_CONDA_ROOT%\envs\synpo-microscopy"
if not exist "%SYNPO_ENV_ROOT%\python.exe" (
    set "SYNPO_CONDA_ROOT=%USERPROFILE%\anaconda3"
    set "SYNPO_ENV_ROOT=%USERPROFILE%\anaconda3\envs\synpo-microscopy"
)

if not exist "%SYNPO_ENV_ROOT%\python.exe" (
    echo Synpo environment Python was not found.
    echo Expected environment name: synpo-microscopy
    echo Recreate it from this project with: conda env create -f environment.yml
    pause
    exit /b 1
)

if not exist "%SYNPO_CONDA_ROOT%\Scripts\conda.exe" (
    echo Conda executable was not found at:
    echo %SYNPO_CONDA_ROOT%\Scripts\conda.exe
    pause
    exit /b 1
)

rem Use Conda's own runner so native DLL search paths are prepared correctly,
rem even when "conda" is unavailable in the user's Command Prompt.
"%SYNPO_CONDA_ROOT%\Scripts\conda.exe" run --no-capture-output -p "%SYNPO_ENV_ROOT%" python -m synpo
if errorlevel 1 pause
endlocal
