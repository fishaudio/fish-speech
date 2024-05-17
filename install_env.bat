@echo off
chcp 65001

:: 设置是否使用镜像站的标志，true 表示使用，false 表示不使用
set USE_MIRROR=true
echo use_mirror = %USE_MIRROR%

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
set "packages=torch torchvision torchaudio openai-whisper fish-speech"

:: 检查包是否已安装，如果没有安装则添加到需要安装的包列表
set "install_packages="
for %%p in (%packages%) do (
    %PIP_CMD% show %%p >nul 2>&1
    if errorlevel 1 (
        set "install_packages=!install_packages! %%p"
    )
)



if not "!install_packages!"=="" (
    echo.
    echo 正在安装以下包: !install_packages!
    :: 针对不同的包使用不同的安装源
    for %%p in (!install_packages!) do (
        if "!USE_MIRROR!"=="true" (
            if "%%p"=="torch" (
                %PIP_CMD% install torch --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121 --no-warn-script-location
            ) else if "%%p"=="torchvision" (
                %PIP_CMD% install torchvision --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121 --no-warn-script-location
            ) else if "%%p"=="torchaudio" (
                %PIP_CMD% install torchaudio --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121 --no-warn-script-location
            ) else if "%%p"=="openai-whisper" (
                %PIP_CMD% install -i https://pypi.tuna.tsinghua.edu.cn/simple openai-whisper --no-warn-script-location
            ) else if "%%p"=="fish-speech" (
                %PIP_CMD% install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
            )
        ) else (
            if "%%p"=="torch" (
                %PIP_CMD% install torch --index-url https://download.pytorch.org/whl/cu121 --no-warn-script-location
            ) else if "%%p"=="torchvision" (
                %PIP_CMD% install torchvision --index-url https://download.pytorch.org/whl/cu121 --no-warn-script-location
            ) else if "%%p"=="torchaudio" (
                %PIP_CMD% install torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-warn-script-location
            ) else if "%%p"=="openai-whisper" (
                %PIP_CMD% install openai-whisper --no-warn-script-location
            ) else if "%%p"=="fish-speech" (
                %PIP_CMD% install -e .
            )
        )
    )
)
echo 环境检查并安装完成

endlocal
pause