@echo off
chcp 65001
echo loading page...
set PYTHONPATH=%~dp0
set no_proxy="localhost, 127.0.0.1, 0.0.0.0"

if exist ".\fishenv\" (
    .\fishenv\python fish_speech\webui\manage.py
) else (
    python fish_speech\webui\manage.py
)
