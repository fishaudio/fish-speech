"""
logger封装
"""
import sys

from loguru import logger

# 移除所有默认的处理器
logger.remove()

# 自定义格式并添加到标准输出
log_format = (
    "<g>{time:MM-DD HH:mm:ss}</g> <lvl>{level:<9}</lvl>| {file}:{line} | {message}"
)

logger.add(sys.stdout, format=log_format, backtrace=True, diagnose=True)
