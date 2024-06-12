# -*- coding: utf-8 -*-
"""CARDINAL类 (包含小数DECIMAL类)
纯数 <=> 中文字符串 方法
中文字符串 <=> 纯数 方法
"""

__author__ = "Zhiyang Zhou <zyzhou@stu.xmu.edu.cn>"
__data__ = "2019-05-03"

from fish_speech.text.chn_text_norm.basic_util import *


class Cardinal:
    """
    CARDINAL类
    """

    def __init__(self, cardinal=None, chntext=None):
        self.cardinal = cardinal
        self.chntext = chntext

    def chntext2cardinal(self):
        return chn2num(self.chntext)

    def cardinal2chntext(self):
        return num2chn(self.cardinal)


if __name__ == "__main__":

    # 测试程序
    print(Cardinal(cardinal="21357.230").cardinal2chntext())
