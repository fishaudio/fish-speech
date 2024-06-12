# -*- coding: utf-8 -*-
"""
TEXT类
"""

__author__ = "Zhiyang Zhou <zyzhou@stu.xmu.edu.cn>"
__data__ = "2019-05-03"

import re

from fish_speech.text.chn_text_norm.cardinal import Cardinal
from fish_speech.text.chn_text_norm.date import Date
from fish_speech.text.chn_text_norm.digit import Digit
from fish_speech.text.chn_text_norm.fraction import Fraction
from fish_speech.text.chn_text_norm.money import Money
from fish_speech.text.chn_text_norm.percentage import Percentage
from fish_speech.text.chn_text_norm.telephone import TelePhone

CURRENCY_NAMES = (
    "(人民币|美元|日元|英镑|欧元|马克|法郎|加拿大元|澳元|港币|先令|芬兰马克|爱尔兰镑|"
    "里拉|荷兰盾|埃斯库多|比塞塔|印尼盾|林吉特|新西兰元|比索|卢布|新加坡元|韩元|泰铢)"
)
CURRENCY_UNITS = "((亿|千万|百万|万|千|百)|(亿|千万|百万|万|千|百|)元|(亿|千万|百万|万|千|百|)块|角|毛|分)"
COM_QUANTIFIERS = (
    "(匹|张|座|回|场|尾|条|个|首|阙|阵|网|炮|顶|丘|棵|只|支|袭|辆|挑|担|颗|壳|窠|曲|墙|群|腔|"
    "砣|座|客|贯|扎|捆|刀|令|打|手|罗|坡|山|岭|江|溪|钟|队|单|双|对|出|口|头|脚|板|跳|枝|件|贴|"
    "针|线|管|名|位|身|堂|课|本|页|家|户|层|丝|毫|厘|分|钱|两|斤|担|铢|石|钧|锱|忽|(千|毫|微)克|"
    "毫|厘|分|寸|尺|丈|里|寻|常|铺|程|(千|分|厘|毫|微)米|撮|勺|合|升|斗|石|盘|碗|碟|叠|桶|笼|盆|"
    "盒|杯|钟|斛|锅|簋|篮|盘|桶|罐|瓶|壶|卮|盏|箩|箱|煲|啖|袋|钵|年|月|日|季|刻|时|周|天|秒|分|旬|"
    "纪|岁|世|更|夜|春|夏|秋|冬|代|伏|辈|丸|泡|粒|颗|幢|堆|条|根|支|道|面|片|张|颗|块|人|抽)"
)


class Text:
    """
    Text类
    """

    def __init__(self, raw_text, norm_text=None):
        self.raw_text = "^" + raw_text + "$"
        self.norm_text = norm_text

    def _particular(self):
        text = self.norm_text
        pattern = re.compile(r"(([a-zA-Z]+)二([a-zA-Z]+))")
        matchers = pattern.findall(text)
        if matchers:
            # print('particular')
            for matcher in matchers:
                text = text.replace(matcher[0], matcher[1] + "2" + matcher[2], 1)
        self.norm_text = text
        return self.norm_text

    def normalize(self):
        text = self.raw_text

        # 规范化日期
        pattern = re.compile(
            r"\D+((([089]\d|(19|20)\d{2})年)?(\d{1,2}月(\d{1,2}[日号])?)?)"
        )
        matchers = pattern.findall(text)
        if matchers:
            # print('date')
            for matcher in matchers:
                text = text.replace(matcher[0], Date(date=matcher[0]).date2chntext(), 1)

        # 规范化金钱
        pattern = re.compile(
            r"\D+((\d+(\.\d+)?)[多余几]?"
            + CURRENCY_UNITS
            + "(\d"
            + CURRENCY_UNITS
            + "?)?)"
        )
        matchers = pattern.findall(text)
        if matchers:
            # print('money')
            for matcher in matchers:
                text = text.replace(
                    matcher[0], Money(money=matcher[0]).money2chntext(), 1
                )

        # 规范化固话/手机号码
        # 手机
        # http://www.jihaoba.com/news/show/13680
        # 移动：139、138、137、136、135、134、159、158、157、150、151、152、188、187、182、183、184、178、198
        # 联通：130、131、132、156、155、186、185、176
        # 电信：133、153、189、180、181、177
        pattern = re.compile(r"\D((\+?86 ?)?1([38]\d|5[0-35-9]|7[678]|9[89])\d{8})\D")
        matchers = pattern.findall(text)
        if matchers:
            # print('telephone')
            for matcher in matchers:
                text = text.replace(
                    matcher[0], TelePhone(telephone=matcher[0]).telephone2chntext(), 1
                )
        # 固话
        pattern = re.compile(r"\D((0(10|2[1-3]|[3-9]\d{2})-?)?[1-9]\d{6,7})\D")
        matchers = pattern.findall(text)
        if matchers:
            # print('fixed telephone')
            for matcher in matchers:
                text = text.replace(
                    matcher[0],
                    TelePhone(telephone=matcher[0]).telephone2chntext(fixed=True),
                    1,
                )

        # 规范化分数
        pattern = re.compile(r"(\d+/\d+)")
        matchers = pattern.findall(text)
        if matchers:
            # print('fraction')
            for matcher in matchers:
                text = text.replace(
                    matcher, Fraction(fraction=matcher).fraction2chntext(), 1
                )

        # 规范化百分数
        text = text.replace("％", "%")
        pattern = re.compile(r"(\d+(\.\d+)?%)")
        matchers = pattern.findall(text)
        if matchers:
            # print('percentage')
            for matcher in matchers:
                text = text.replace(
                    matcher[0],
                    Percentage(percentage=matcher[0]).percentage2chntext(),
                    1,
                )

        # 规范化纯数+量词
        pattern = re.compile(r"(\d+(\.\d+)?)[多余几]?" + COM_QUANTIFIERS)
        matchers = pattern.findall(text)
        if matchers:
            # print('cardinal+quantifier')
            for matcher in matchers:
                text = text.replace(
                    matcher[0], Cardinal(cardinal=matcher[0]).cardinal2chntext(), 1
                )

        # 规范化数字编号
        pattern = re.compile(r"(\d{4,32})")
        matchers = pattern.findall(text)
        if matchers:
            # print('digit')
            for matcher in matchers:
                text = text.replace(matcher, Digit(digit=matcher).digit2chntext(), 1)

        # 规范化纯数
        pattern = re.compile(r"(\d+(\.\d+)?)")
        matchers = pattern.findall(text)
        if matchers:
            # print('cardinal')
            for matcher in matchers:
                text = text.replace(
                    matcher[0], Cardinal(cardinal=matcher[0]).cardinal2chntext(), 1
                )

        self.norm_text = text
        self._particular()

        return self.norm_text.lstrip("^").rstrip("$")


if __name__ == "__main__":

    # 测试程序
    print(Text(raw_text="固话：0595-23865596或23880880。").normalize())
    print(Text(raw_text="手机：+86 19859213959或15659451527。").normalize())
    print(Text(raw_text="分数：32477/76391。").normalize())
    print(Text(raw_text="百分数：80.03%。").normalize())
    print(Text(raw_text="编号：31520181154418。").normalize())
    print(Text(raw_text="纯数：2983.07克或12345.60米。").normalize())
    print(Text(raw_text="日期：1999年2月20日或09年3月15号。").normalize())
    print(Text(raw_text="金钱：12块5，34.5元，20.1万").normalize())
    print(Text(raw_text="特殊：O2O或B2C。").normalize())
