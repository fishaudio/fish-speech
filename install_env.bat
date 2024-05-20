@echo off
chcp 65001

set USE_MIRROR=true
echo use_mirror = %USE_MIRROR%
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
set MINICONDA_DOWNLOAD_URL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py310_23.3.1-0-Windows-x86_64.exe
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
    call "%CONDA_ROOT_PREFIX%\_conda.exe" create --no-shortcuts -y -k --prefix "%INSTALL_ENV_DIR%" -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ python=3.10

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


set "packages=torch torchvision torchaudio openai-whisper fish-speech"


set "install_packages="
for %%p in (%packages%) do (
    %PIP_CMD% show %%p >nul 2>&1
    if errorlevel 1 (
        set "install_packages=!install_packages! %%p"
    )
)


if not "!install_packages!"=="" (
    echo.
    echo Installing: !install_packages!

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
echo Environment Check: Success.

endlocal
pause
