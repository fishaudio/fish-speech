@echo off
chcp 65001

set USE_MIRROR=true
echo "USE_MIRROR: %USE_MIRROR%"
setlocal enabledelayedexpansion

cd /D "%~dp0"

set PATH="%PATH%";%SystemRoot%\system32

echo %PATH%


echo "%CD%"| findstr /R /C:"[!#\$%&()\*+,;<=>?@\[\]\^`{|}~\u4E00-\u9FFF ] " >nul && (
    echo.
    echo There are special characters in the current path, please make the path of fish-speech free of special characters before running. && (
        goto end
    )
)


set TMP=%CD%\fishenv
set TEMP=%CD%\fishenv

(call conda deactivate && call conda deactivate && call conda deactivate) 2>nul

set INSTALL_DIR=%cd%\fishenv
set CONDA_ROOT_PREFIX=%cd%\fishenv\conda
set INSTALL_ENV_DIR=%cd%\fishenv\env
set PIP_CMD=%cd%\fishenv\env\python -m pip
set PYTHON_CMD=%cd%\fishenv\env\python
set API_FLAG_PATH=%~dp0API_FLAGS.txt
set MINICONDA_DOWNLOAD_URL=https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Windows-x86_64.exe
if "!USE_MIRROR!" == "true" (
    set MINICONDA_DOWNLOAD_URL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py310_23.3.1-0-Windows-x86_64.exe
)
set MINICONDA_CHECKSUM=307194e1f12bbeb52b083634e89cc67db4f7980bd542254b43d3309eaf7cb358
set conda_exists=F

call "%CONDA_ROOT_PREFIX%\_conda.exe" --version >nul 2>&1
if "%ERRORLEVEL%" EQU "0" set conda_exists=T

if "%conda_exists%" == "F" (
    echo.
    echo Downloading Miniconda...
    mkdir "%INSTALL_DIR%" 2>nul
    call curl -Lk "%MINICONDA_DOWNLOAD_URL%" > "%INSTALL_DIR%\miniconda_installer.exe"
    if errorlevel 1 (
        echo.
        echo Failed to download miniconda.
        goto end
    )
    for /f %%a in ('
        certutil -hashfile "%INSTALL_DIR%\miniconda_installer.exe" sha256
        ^| find /i /v " "
        ^| find /i "%MINICONDA_CHECKSUM%"
    ') do (
        set "hash=%%a"
    )
    if not defined hash (
        echo.
        echo Miniconda hash mismatched!
        del "%INSTALL_DIR%\miniconda_installer.exe"
        goto end
    ) else (
        echo.
        echo Miniconda hash matched successfully.
    )
    echo Downloaded "%CONDA_ROOT_PREFIX%"
    start /wait "" "%INSTALL_DIR%\miniconda_installer.exe" /InstallationType=JustMe /NoShortcuts=1 /AddToPath=0 /RegisterPython=0 /NoRegistry=1 /S /D=%CONDA_ROOT_PREFIX%

    call "%CONDA_ROOT_PREFIX%\_conda.exe" --version
    if errorlevel 1 (
        echo.
        echo Cannot install Miniconda.
        goto end
    ) else (
        echo.
        echo Miniconda Install success.
    )

    del "%INSTALL_DIR%\miniconda_installer.exe"
)


if not exist "%INSTALL_ENV_DIR%" (
    echo.
    echo Creating Conda Environment...
    if "!USE_MIRROR!" == "true" (
        call "%CONDA_ROOT_PREFIX%\_conda.exe" create --no-shortcuts -y -k --prefix "%INSTALL_ENV_DIR%" -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ python=3.10
    ) else (
        call "%CONDA_ROOT_PREFIX%\_conda.exe" create --no-shortcuts -y -k --prefix "%INSTALL_ENV_DIR%" python=3.10
    )

    if errorlevel 1 (
        echo.
        echo Failed to Create Environment.
        goto end
    )
)

if not exist "%INSTALL_ENV_DIR%\python.exe" (
    echo.
    echo Conda Env does not exist.
    goto end
)

set PYTHONNOUSERSITE=1
set PYTHONPATH=
set PYTHONHOME=
set "CUDA_PATH=%INSTALL_ENV_DIR%"
set "CUDA_HOME=%CUDA_PATH%"

call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"

if errorlevel 1 (
    echo.
    echo Failed to activate Env.
    goto end
) else (
    echo.
    echo successfully create env.
)

set "HF_ENDPOINT=https://huggingface.co"
set "no_proxy="
if "%USE_MIRROR%"=="true" (
    set "HF_ENDPOINT=https://hf-mirror.com"
    set "no_proxy=localhost,127.0.0.1,0.0.0.0"
)

echo "HF_ENDPOINT: !HF_ENDPOINT!"
echo "NO_PROXY: !no_proxy!"

%PIP_CMD% install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

%PIP_CMD% install -e . --upgrade-strategy only-if-needed

call :download_and_install "triton_windows-0.1.0-py3-none-any.whl" ^
        "%HF_ENDPOINT%/datasets/SpicyqSama007/windows_compile/resolve/main/triton_windows-0.1.0-py3-none-any.whl?download=true" ^
        "2cc998638180f37cf5025ab65e48c7f629aa5a369176cfa32177d2bd9aa26a0a"


endlocal
echo "Environment Check: Success."
:end
pause

goto :EOF


:download_and_install
setlocal

set "WHEEL_FILE=%1"
set "URL=%2"
set "CHKSUM=%3"

:DOWNLOAD
if not exist "%WHEEL_FILE%" (
    call curl -Lk "%URL%" --output "%WHEEL_FILE%"
)

for /f "delims=" %%I in ("certutil -hashfile %WHEEL_FILE% SHA256 ^| find /i %CHKSUM%") do (
    set "FILE_VALID=true"
)

if not defined FILE_VALID (
    echo File checksum does not match, re-downloading...
    del "%WHEEL_FILE%"
    goto DOWNLOAD
)

echo "OK for %WHEEL_FILE%"
%PIP_CMD% install "%WHEEL_FILE%" --no-warn-script-location
del "%WHEEL_FILE%"

endlocal
goto :EOF
