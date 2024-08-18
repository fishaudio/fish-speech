@echo off
chcp 65001

set no_proxy="127.0.0.1, 0.0.0.0, localhost"
setlocal enabledelayedexpansion

cd /D "%~dp0"

set PATH="%PATH%";%SystemRoot%\system32


echo "%CD%"| findstr /R /C:"[!#\$%&()\*+,;<=>?@\[\]\^`{|}~\u4E00-\u9FFF ] " >nul && (
    echo.
    echo There are special characters in the current path, please make the path of fish-speech free of special characters before running. && (
        goto end
    )
)


set TMP=%CD%\fishenv
set TEMP=%CD%\fishenv


(call conda deactivate && call conda deactivate && call conda deactivate) 2>nul


set CONDA_ROOT_PREFIX=%cd%\fishenv\conda
set INSTALL_ENV_DIR=%cd%\fishenv\env


set PYTHONNOUSERSITE=1
set PYTHONPATH=%~dp0
set PYTHONHOME=


call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"

if errorlevel 1 (
    echo.
    echo Environment activation failed.
    goto end
) else (
    echo.
    echo Environment activation succeeded.
)

cmd /k "%*"

:end
pause
