# -*- coding: utf-8 -*-
"""DATE类
日期 <=> 中文字符串 方法
中文字符串 <=> 日期 方法
"""

__author__ = "Zhiyang Zhou <zyzhou@stu.xmu.edu.cn>"
__data__ = "2019-05-07"

from fish_speech.text.chn_text_norm.cardinal import Cardinal
from fish_speech.text.chn_text_norm.digit import Digit


class Date:
    """
    DATE类
    """

    def __init__(self, date=None, chntext=None):
        self.date = date
        self.chntext = chntext

    # def chntext2date(self):
    #     chntext = self.chntext
    #     try:
    #         year, other = chntext.strip().split('年', maxsplit=1)
    #         year = Digit(chntext=year).digit2chntext() + '年'
    #     except ValueError:
    #         other = chntext
    #         year = ''
    #     if other:
    #         try:
    #             month, day = other.strip().split('月', maxsplit=1)
    #             month = Cardinal(chntext=month).chntext2cardinal() + '月'
    #         except ValueError:
    #             day = chntext
    #             month = ''
    #         if day:
    #             day = Cardinal(chntext=day[:-1]).chntext2cardinal() + day[-1]
    #     else:
    #         month = ''
    #         day = ''
    #     date = year + month + day
    #     self.date = date
    #     return self.date

    def date2chntext(self):
        date = self.date
        try:
            year, other = date.strip().split("年", maxsplit=1)
            year = Digit(digit=year).digit2chntext() + "年"
        except ValueError:
            other = date
            year = ""
        if other:
            try:
                month, day = other.strip().split("月", maxsplit=1)
                month = Cardinal(cardinal=month).cardinal2chntext() + "月"
            except ValueError:
                day = date
                month = ""
            if day:
                day = Cardinal(cardinal=day[:-1]).cardinal2chntext() + day[-1]
        else:
            month = ""
            day = ""
        chntext = year + month + day
        self.chntext = chntext
        return self.chntext


if __name__ == "__main__":

    # 测试
    print(Date(date="09年3月16日").date2chntext())
