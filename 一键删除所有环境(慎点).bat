@echo off
chcp 65001

.\fishenv\python -m pip freeze > installed.txt
.\fishenv\python -m pip uninstall -r installed.txt -y
del installed.txt -y

echo OK!!
pause
