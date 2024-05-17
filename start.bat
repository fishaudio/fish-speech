@echo off
chcp 65001


set PYTHONPATH=%~dp0
set PYTHON_CMD=%cd%\fishenv\env\python
set API_FLAG_PATH=%~dp0API_FLAGS.txt
%PYTHON_CMD% .\tools\download_models.py

setlocal enabledelayedexpansion

set no_proxy="localhost, 127.0.0.1, 0.0.0.0"
:: 设置Hugging Face镜像源
set HF_ENDPOINT=https://hf-mirror.com


set "API_FLAGS="
set "flags="
:: 检查API_FLAG文件是否存在
if exist "%API_FLAG_PATH%" (
    for /f "usebackq tokens=*" %%a in ("%API_FLAG_PATH%") do (
        set "line=%%a"
        :: 去除行尾的反斜杠和空白字符，并且跳过以#开头的行
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

:: 去除API_FLAGS变量最后的空格
if not "!API_FLAGS!"=="" set "API_FLAGS=!API_FLAGS:~0,-1!"

:: 初始化 flags 变量
set "flags="

:: 检查是否包含 --api 参数
echo !API_FLAGS! | findstr /C:"--api" >nul 2>&1
if !errorlevel! equ 0 (
    echo.
    echo 启动HTTP API推理
    set "mode=api"
    goto process_flags
)

:: 检查是否包含 --infer 参数
echo !API_FLAGS! | findstr /C:"--infer" >nul 2>&1
if !errorlevel! equ 0 (
    echo.
    echo 启动WebUI推理
    set "mode=infer"
    goto process_flags
)


:process_flags
for %%p in (!API_FLAGS!) do (
    if not "%%p"=="--!mode!" (
        set "flags=!flags! %%p"
    )
)

:: 去除 flags 变量开头的空格
if not "!flags!"=="" set "flags=!flags:~1!"

echo Debug: flags = !flags!

:: 根据 mode 变量启动相应的推理
if "!mode!"=="api" (
    %PYTHON_CMD% -m tools.api !flags!
) else if "!mode!"=="infer" (
    %PYTHON_CMD% -m tools.webui !flags!
)

echo.
echo 接下来启动页面
%PYTHON_CMD% fish_speech\webui\manage.py

:end
endlocal
pause
