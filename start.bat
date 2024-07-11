@echo off
chcp 65001

set USE_MIRROR=true
set PYTHONPATH=%~dp0
set PYTHON_CMD=%cd%\fishenv\env\python
set API_FLAG_PATH=%~dp0API_FLAGS.txt
set KMP_DUPLICATE_LIB_OK=TRUE

setlocal enabledelayedexpansion

set "HF_ENDPOINT=https://huggingface.co"
set "no_proxy="
if "%USE_MIRROR%" == "true" (
    set "HF_ENDPOINT=https://hf-mirror.com"
    set "no_proxy=localhost, 127.0.0.1, 0.0.0.0"
)
echo "HF_ENDPOINT: !HF_ENDPOINT!"
echo "NO_PROXY: !no_proxy!"
%PYTHON_CMD% .\tools\download_models.py

set "API_FLAGS="
set "flags="

if exist "%API_FLAG_PATH%" (
    for /f "usebackq tokens=*" %%a in ("%API_FLAG_PATH%") do (
        set "line=%%a"
        if not "!line:~0,1!"=="#" (
            set "line=!line: =<SPACE>!"
            set "line=!line:\=!"
            set "line=!line:<SPACE>= !"
            if not "!line!"=="" (
                set "API_FLAGS=!API_FLAGS!!line! "
            )
        )
    )
)


if not "!API_FLAGS!"=="" set "API_FLAGS=!API_FLAGS:~0,-1!"

set "flags="

echo !API_FLAGS! | findstr /C:"--api" >nul 2>&1
if !errorlevel! equ 0 (
    echo.
    echo Start HTTP API...
    set "mode=api"
    goto process_flags
)

echo !API_FLAGS! | findstr /C:"--infer" >nul 2>&1
if !errorlevel! equ 0 (
    echo.
    echo Start WebUI Inference...
    set "mode=infer"
    goto process_flags
)


:process_flags
for %%p in (!API_FLAGS!) do (
    if not "%%p"=="--!mode!" (
        set "flags=!flags! %%p"
    )
)

if not "!flags!"=="" set "flags=!flags:~1!"

echo Debug: flags = !flags!

if "!mode!"=="api" (
    %PYTHON_CMD% -m tools.api !flags!
) else if "!mode!"=="infer" (
    %PYTHON_CMD% -m tools.webui !flags!
)

echo.
echo Next launch the page...
%PYTHON_CMD% fish_speech\webui\manage.py


:end
endlocal
pause
