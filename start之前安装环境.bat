@echo off
chcp 65001

set no_proxy="127.0.0.1, 0.0.0.0, localhost"
setlocal
set PIP_CONFIG_FILE=%APPDATA%\pip\pip.ini

:: 确保pip配置目录存在
if not exist "%APPDATA%\pip\" mkdir "%APPDATA%\pip"

:: 创建或修改pip.ini文件
(
echo [global]
echo.
echo index-url = https://pypi.tuna.tsinghua.edu.cn/simple/
echo.
echo [install]
echo.
echo trusted-host = 
echo.
echo    pypi.tuna.tsinghua.edu.cn 

) > "%PIP_CONFIG_FILE%"

echo pip配置文件已更新
endlocal

.\fishenv\python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-warn-script-location
.\fishenv\python -m pip install -e . --no-warn-script-location
.\fishenv\python -m pip install openai-whisper --no-warn-script-location

echo OK!!
pause