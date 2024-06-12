# -*- coding: utf-8 -*-
"""PERCENTAGE类
百分数 <=> 中文字符串 方法
中文字符串 <=> 百分数 方法
"""

__author__ = "Zhiyang Zhou <zyzhou@stu.xmu.edu.cn>"
__data__ = "2019-05-06"

from fish_speech.text.chn_text_norm.basic_util import *


class Percentage:
    """
    PERCENTAGE类
    """

    def __init__(self, percentage=None, chntext=None):
        self.percentage = percentage
        self.chntext = chntext

    def chntext2percentage(self):
        return chn2num(self.chntext.strip().strip("百分之")) + "%"

    def percentage2chntext(self):
        return "百分之" + num2chn(self.percentage.strip().strip("%"))


if __name__ == "__main__":

    # 测试程序
    print(Percentage(chntext="百分之五十六点零三").chntext2percentage())
    print(Percentage(percentage="65.3%").percentage2chntext())
