@echo off
chcp 65001

set no_proxy="127.0.0.1, 0.0.0.0, localhost"
setlocal enabledelayedexpansion

cd /D "%~dp0"

set PATH="%PATH%";%SystemRoot%\system32

:: 安装Miniconda
:: 检查是否有特殊字符
echo "%CD%"| findstr /R /C:"[!#\$%&()\*+,;<=>?@\[\]\^`{|}~\u4E00-\u9FFF ] " >nul && (
    echo.
    echo 当前路径中存在特殊字符，请使fish-speech的路径不含特殊字符后再运行。 && (
        goto end
    )
)
:: 解决跨驱动器安装问题
set TMP=%CD%\fishenv
set TEMP=%CD%\fishenv
:: 取消激活已经激活的环境
(call conda deactivate && call conda deactivate && call conda deactivate) 2>nul
:: 安装路径配置
set INSTALL_DIR=%cd%\fishenv
set CONDA_ROOT_PREFIX=%cd%\fishenv\conda
set INSTALL_ENV_DIR=%cd%\fishenv\env
set PIP_CMD=%cd%\fishenv\env\python -m pip
set PYTHON_CMD=%cd%\fishenv\env\python
set API_FLAG_PATH=%~dp0API_FLAGS.txt
set MINICONDA_DOWNLOAD_URL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py310_23.3.1-0-Windows-x86_64.exe
set MINICONDA_CHECKSUM=307194e1f12bbeb52b083634e89cc67db4f7980bd542254b43d3309eaf7cb358
set conda_exists=F
:: 确定是否要安装conda
call "%CONDA_ROOT_PREFIX%\_conda.exe" --version >nul 2>&1
if "%ERRORLEVEL%" EQU "0" set conda_exists=T
:: 下载Miniconda
if "%conda_exists%" == "F" (
    echo.
    echo 正在下载Miniconda...
    mkdir "%INSTALL_DIR%" 2>nul
    :: 使用curl下载Miniconda安装程序
    call curl -Lk "%MINICONDA_DOWNLOAD_URL%" > "%INSTALL_DIR%\miniconda_installer.exe"
    :: 检查下载是否成功
    if errorlevel 1 (
        echo.
        echo 下载Miniconda失败
        goto end
    )
    :: 哈希校验
    for /f %%a in ('
        certutil -hashfile "%INSTALL_DIR%\miniconda_installer.exe" sha256
        ^| find /i /v " "
        ^| find /i "%MINICONDA_CHECKSUM%"
    ') do (
        :: 如果哈希值匹配预设的校验和，将其存储在变量中
        set "hash=%%a"
    )
    if not defined hash (
        echo.
        echo Miniconda安装程序的哈希值不匹配
        del "%INSTALL_DIR%\miniconda_installer.exe"
        goto end
    ) else (
        echo.
        echo Miniconda安装程序的哈希值成功匹配
    )
    echo 下载完成，接下来安装Miniconda至"%CONDA_ROOT_PREFIX%"
    start /wait "" "%INSTALL_DIR%\miniconda_installer.exe" /InstallationType=JustMe /NoShortcuts=1 /AddToPath=0 /RegisterPython=0 /NoRegistry=1 /S /D=%CONDA_ROOT_PREFIX%
    :: 测试是否成功安装
    call "%CONDA_ROOT_PREFIX%\_conda.exe" --version
    if errorlevel 1 (
        echo.
        echo 未安装Miniconda
        goto end
    ) else (
        echo.
        echo Miniconda安装成功
    )
    :: 删除安装程序
    del "%INSTALL_DIR%\miniconda_installer.exe"
)

:: 创建conda环境
if not exist "%INSTALL_ENV_DIR%" (
    echo.
    echo 正在创建conda环境...
    call "%CONDA_ROOT_PREFIX%\_conda.exe" create --no-shortcuts -y -k --prefix "%INSTALL_ENV_DIR%" -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ python=3.10
    :: 检查环境创建是否成功
    if errorlevel 1 (
        echo.
        echo 创建conda环境失败
        goto end
    )
)
:: 检查是否真的创建了环境
if not exist "%INSTALL_ENV_DIR%\python.exe" (
    echo.
    echo Conda环境不存在
    goto end
)
:: 环境隔离
set PYTHONNOUSERSITE=1
set PYTHONPATH=
set PYTHONHOME=
set "CUDA_PATH=%INSTALL_ENV_DIR%"
set "CUDA_HOME=%CUDA_PATH%"
:: 激活环境
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"
:: 检查环境是否成功激活
if errorlevel 1 (
    echo.
    echo 环境激活失败
    goto end
) else (
    echo.
    echo 环境激活成功
)
:: 安装依赖
%PIP_CMD% show torch >nul 2>&1
if errorlevel 1 (
    echo.
    echo 未安装pytorch，正在安装...
    %PIP_CMD% install torch --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121 --no-warn-script-location
)
%PIP_CMD% show torchvision >nul 2>&1
if errorlevel 1 (
    echo.
    echo 未安装torchvision，正在安装...
    %PIP_CMD% install torchvision --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121 --no-warn-script-location
)
%PIP_CMD% show torchaudio >nul 2>&1
if errorlevel 1 (
    echo.
    echo 未安装torchaudio，正在安装...
    %PIP_CMD% install torchaudio --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121 --no-warn-script-location
)
%PIP_CMD% show openai-whisper >nul 2>&1
if errorlevel 1 (
    echo.
    echo 未安装openai-whisper，正在安装...
    %PIP_CMD% install -i https://pypi.tuna.tsinghua.edu.cn/simple openai-whisper --no-warn-script-location
)
%PIP_CMD% show fish-speech >nul 2>&1
if errorlevel 1 (
    echo.
    echo 未安装fish-speech，正在安装...
    %PIP_CMD% install -e .
)

:: 设置Hugging Face镜像源
set HF_ENDPOINT="https://hf-mirror.com"

:: 设置API_FLAG
:: 初始化API_FLAG
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
:: 看看开不开api推理
echo !API_FLAGS! | findstr /C:"--api" >nul 2>&1
if !errorlevel! equ 0 (
    echo.
    echo 启动HTTP API推理
    set "first_arg=true"
    for %%a in (%*) do (
        if "!first_arg!"=="true" (
            set "first_arg=false"
        ) else (
            if not "%%a"=="--api" (
                set "flags=!flags!%%a"
            )
        )
    )
    %PYTHON_CMD% -m tools.api !flags!
) else (
    if defined flags (
        :: 启动WebUI推理
        echo.
        echo 启动WebUI推理
        %PYTHON_CMD% -m tools.webui !flags!
    )
)
echo.
echo 接下来启动页面
%PYTHON_CMD% fish_speech\webui\manage.py
:end
pause
