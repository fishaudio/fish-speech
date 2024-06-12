# -*- coding: utf-8 -*-
"""TELEPHONE类
电话号码 <=> 中文字符串 方法
中文字符串 <=> 电话号码 方法
"""

__author__ = "Zhiyang Zhou <zyzhou@stu.xmu.edu.cn>"
__data__ = "2019-05-03"

from fish_speech.text.chn_text_norm.basic_util import *


class TelePhone:
    """
    TELEPHONE类
    """

    def __init__(self, telephone=None, raw_chntext=None, chntext=None):
        self.telephone = telephone
        self.raw_chntext = raw_chntext
        self.chntext = chntext

    # def chntext2telephone(self):
    #     sil_parts = self.raw_chntext.split('<SIL>')
    #     self.telephone = '-'.join([
    #         str(chn2num(p)) for p in sil_parts
    #     ])
    #     return self.telephone

    def telephone2chntext(self, fixed=False):

        if fixed:
            sil_parts = self.telephone.split("-")
            self.raw_chntext = "<SIL>".join(
                [num2chn(part, alt_two=False, use_units=False) for part in sil_parts]
            )
            self.chntext = self.raw_chntext.replace("<SIL>", "")
        else:
            sp_parts = self.telephone.strip("+").split()
            self.raw_chntext = "<SP>".join(
                [num2chn(part, alt_two=False, use_units=False) for part in sp_parts]
            )
            self.chntext = self.raw_chntext.replace("<SP>", "")
        return self.chntext


if __name__ == "__main__":

    # 测试程序
    print(TelePhone(telephone="0595-23980880").telephone2chntext())
    # print(TelePhone(raw_chntext='零五九五杠二三八六五零九八').chntext2telephone())
