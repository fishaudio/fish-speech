# -*- coding: utf-8 -*-
"""MONEY类
金钱 <=> 中文字符串 方法
中文字符串 <=> 金钱 方法
"""
import re

__author__ = "Zhiyang Zhou <zyzhou@stu.xmu.edu.cn>"
__data__ = "2019-05-08"

from fish_speech.text.chn_text_norm.cardinal import Cardinal


class Money:
    """
    MONEY类
    """

    def __init__(self, money=None, chntext=None):
        self.money = money
        self.chntext = chntext

    # def chntext2money(self):
    #     return self.money

    def money2chntext(self):
        money = self.money
        pattern = re.compile(r"(\d+(\.\d+)?)")
        matchers = pattern.findall(money)
        if matchers:
            for matcher in matchers:
                money = money.replace(
                    matcher[0], Cardinal(cardinal=matcher[0]).cardinal2chntext()
                )
        self.chntext = money
        return self.chntext


if __name__ == "__main__":

    # 测试
    print(Money(money="21.5万元").money2chntext())
    print(Money(money="230块5毛").money2chntext())
