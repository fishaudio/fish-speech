# -*- coding: utf-8 -*-
"""FRACTION类
分数 <=> 中文字符串 方法
中文字符串 <=> 分数 方法
"""

__author__ = "Zhiyang Zhou <zyzhou@stu.xmu.edu.cn>"
__data__ = "2019-05-03"

from fish_speech.text.chn_text_norm.basic_util import *


class Fraction:
    """
    FRACTION类
    """

    def __init__(self, fraction=None, chntext=None):
        self.fraction = fraction
        self.chntext = chntext

    def chntext2fraction(self):
        denominator, numerator = self.chntext.split("分之")
        return chn2num(numerator) + "/" + chn2num(denominator)

    def fraction2chntext(self):
        numerator, denominator = self.fraction.split("/")
        return num2chn(denominator) + "分之" + num2chn(numerator)


if __name__ == "__main__":

    # 测试程序
    print(Fraction(fraction="2135/7230").fraction2chntext())
    print(Fraction(chntext="五百八十一分之三百六十九").chntext2fraction())
