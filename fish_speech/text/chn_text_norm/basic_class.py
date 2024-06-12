# -*- coding: utf-8 -*-
"""基本类
中文字符类
中文数字/数位类
中文数字类
中文数位类
中文数字系统类
中文数学符号类
*中文其他符号类
"""

__author__ = "Zhiyang Zhou <zyzhou@stu.xmu.edu.cn>"
__data__ = "2019-05-02"

from fish_speech.text.chn_text_norm.basic_constant import NUMBERING_TYPES


class ChineseChar(object):
    """
    中文字符
    每个字符对应简体和繁体,
    e.g. 简体 = '负', 繁体 = '負'
    转换时可转换为简体或繁体
    """

    def __init__(self, simplified, traditional):
        self.simplified = simplified
        self.traditional = traditional
        self.__repr__ = self.__str__

    def __str__(self):
        return self.simplified or self.traditional or None

    def __repr__(self):
        return self.__str__()


class ChineseNumberUnit(ChineseChar):
    """
    中文数字/数位字符
    每个字符除繁简体外还有一个额外的大写字符
    e.g. '陆' 和 '陸'
    """

    def __init__(self, power, simplified, traditional, big_s, big_t):
        super(ChineseNumberUnit, self).__init__(simplified, traditional)
        self.power = power
        self.big_s = big_s
        self.big_t = big_t

    def __str__(self):
        return "10^{}".format(self.power)

    @classmethod
    def create(cls, index, value, numbering_type=NUMBERING_TYPES[1], small_unit=False):

        if small_unit:
            return ChineseNumberUnit(
                power=index + 1,
                simplified=value[0],
                traditional=value[1],
                big_s=value[1],
                big_t=value[1],
            )
        elif numbering_type == NUMBERING_TYPES[0]:
            return ChineseNumberUnit(
                power=index + 8,
                simplified=value[0],
                traditional=value[1],
                big_s=value[0],
                big_t=value[1],
            )
        elif numbering_type == NUMBERING_TYPES[1]:
            return ChineseNumberUnit(
                power=(index + 2) * 4,
                simplified=value[0],
                traditional=value[1],
                big_s=value[0],
                big_t=value[1],
            )
        elif numbering_type == NUMBERING_TYPES[2]:
            return ChineseNumberUnit(
                power=pow(2, index + 3),
                simplified=value[0],
                traditional=value[1],
                big_s=value[0],
                big_t=value[1],
            )
        else:
            raise ValueError(
                "Counting type should be in {0} ({1} provided).".format(
                    NUMBERING_TYPES, numbering_type
                )
            )


class ChineseNumberDigit(ChineseChar):
    """
    中文数字字符
    """

    def __init__(
        self, value, simplified, traditional, big_s, big_t, alt_s=None, alt_t=None
    ):
        super(ChineseNumberDigit, self).__init__(simplified, traditional)
        self.value = value
        self.big_s = big_s
        self.big_t = big_t
        self.alt_s = alt_s
        self.alt_t = alt_t

    def __str__(self):
        return str(self.value)

    @classmethod
    def create(cls, i, v):
        return ChineseNumberDigit(i, v[0], v[1], v[2], v[3])


class ChineseMath(ChineseChar):
    """
    中文数位字符
    """

    def __init__(self, simplified, traditional, symbol, expression=None):
        super(ChineseMath, self).__init__(simplified, traditional)
        self.symbol = symbol
        self.expression = expression
        self.big_s = simplified
        self.big_t = traditional


CC, CNU, CND, CM = ChineseChar, ChineseNumberUnit, ChineseNumberDigit, ChineseMath


class NumberSystem(object):
    """
    中文数字系统
    """

    pass


class MathSymbol(object):
    """
    用于中文数字系统的数学符号 (繁/简体), e.g.
    positive = ['正', '正']
    negative = ['负', '負']
    point = ['点', '點']
    """

    def __init__(self, positive, negative, point):
        self.positive = positive
        self.negative = negative
        self.point = point

    def __iter__(self):
        for v in self.__dict__.values():
            yield v


# class OtherSymbol(object):
#     """
#     其他符号
#     """
#
#     def __init__(self, sil):
#         self.sil = sil
#
#     def __iter__(self):
#         for v in self.__dict__.values():
#             yield v
