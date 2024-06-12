# -*- coding: utf-8 -*-
"""DIGIT类
数字串 <=> 中文字符串 方法
中文字符串 <=> 数字串 方法
"""

__author__ = "Zhiyang Zhou <zyzhou@stu.xmu.edu.cn>"
__data__ = "2019-05-03"

from fish_speech.text.chn_text_norm.basic_util import *


class Digit:
    """
    DIGIT类
    """

    def __init__(self, digit=None, chntext=None):
        self.digit = digit
        self.chntext = chntext

    # def chntext2digit(self):
    #     return chn2num(self.chntext)

    def digit2chntext(self):
        return num2chn(self.digit, alt_two=False, use_units=False)


if __name__ == "__main__":

    # 测试程序
    print(Digit(digit="2016").digit2chntext())
