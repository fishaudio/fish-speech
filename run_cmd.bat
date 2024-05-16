@echo off
chcp 65001

set no_proxy="127.0.0.1, 0.0.0.0, localhost"
setlocal enabledelayedexpansion

cd /D "%~dp0"

set PATH="%PATH%";%SystemRoot%\system32

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
set CONDA_ROOT_PREFIX=%cd%\fishenv\conda
set INSTALL_ENV_DIR=%cd%\fishenv\env

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

:: 进入cmd
cmd /k "%*"

:end
pause