# -*- coding: utf-8 -*-
"""基本方法
创建中文数字系统 方法
中文字符串 <=> 数字串 方法
数字串 <=> 中文字符串 方法
"""

__author__ = "Zhiyang Zhou <zyzhou@stu.xmu.edu.cn>"
__data__ = "2019-05-02"

from fish_speech.text.chn_text_norm.basic_class import *
from fish_speech.text.chn_text_norm.basic_constant import *


def create_system(numbering_type=NUMBERING_TYPES[1]):
    """
    根据数字系统类型返回创建相应的数字系统，默认为 mid
    NUMBERING_TYPES = ['low', 'mid', 'high']: 中文数字系统类型
        low:  '兆' = '亿' * '十' = $10^{9}$,  '京' = '兆' * '十', etc.
        mid:  '兆' = '亿' * '万' = $10^{12}$, '京' = '兆' * '万', etc.
        high: '兆' = '亿' * '亿' = $10^{16}$, '京' = '兆' * '兆', etc.
    返回对应的数字系统
    """

    # chinese number units of '亿' and larger
    all_larger_units = zip(
        LARGER_CHINESE_NUMERING_UNITS_SIMPLIFIED,
        LARGER_CHINESE_NUMERING_UNITS_TRADITIONAL,
    )
    larger_units = [
        CNU.create(i, v, numbering_type, False) for i, v in enumerate(all_larger_units)
    ]
    # chinese number units of '十, 百, 千, 万'
    all_smaller_units = zip(
        SMALLER_CHINESE_NUMERING_UNITS_SIMPLIFIED,
        SMALLER_CHINESE_NUMERING_UNITS_TRADITIONAL,
    )
    smaller_units = [
        CNU.create(i, v, small_unit=True) for i, v in enumerate(all_smaller_units)
    ]
    # digis
    chinese_digis = zip(
        CHINESE_DIGIS,
        CHINESE_DIGIS,
        BIG_CHINESE_DIGIS_SIMPLIFIED,
        BIG_CHINESE_DIGIS_TRADITIONAL,
    )
    digits = [CND.create(i, v) for i, v in enumerate(chinese_digis)]
    digits[0].alt_s, digits[0].alt_t = ZERO_ALT, ZERO_ALT
    digits[1].alt_s, digits[1].alt_t = ONE_ALT, ONE_ALT
    digits[2].alt_s, digits[2].alt_t = TWO_ALTS[0], TWO_ALTS[1]

    # symbols
    positive_cn = CM(POSITIVE[0], POSITIVE[1], "+", lambda x: x)
    negative_cn = CM(NEGATIVE[0], NEGATIVE[1], "-", lambda x: -x)
    point_cn = CM(POINT[0], POINT[1], ".", lambda x, y: float(str(x) + "." + str(y)))
    # sil_cn = CM(SIL[0], SIL[1], '-', lambda x, y: float(str(x) + '-' + str(y)))
    system = NumberSystem()
    system.units = smaller_units + larger_units
    system.digits = digits
    system.math = MathSymbol(positive_cn, negative_cn, point_cn)
    # system.symbols = OtherSymbol(sil_cn)
    return system


def chn2num(chinese_string, numbering_type=NUMBERING_TYPES[1]):

    def get_symbol(char, system):
        for u in system.units:
            if char in [u.traditional, u.simplified, u.big_s, u.big_t]:
                return u
        for d in system.digits:
            if char in [
                d.traditional,
                d.simplified,
                d.big_s,
                d.big_t,
                d.alt_s,
                d.alt_t,
            ]:
                return d
        for m in system.math:
            if char in [m.traditional, m.simplified]:
                return m

    def string2symbols(chinese_string, system):
        int_string, dec_string = chinese_string, ""
        for p in [system.math.point.simplified, system.math.point.traditional]:
            if p in chinese_string:
                int_string, dec_string = chinese_string.split(p)
                break
        return [get_symbol(c, system) for c in int_string], [
            get_symbol(c, system) for c in dec_string
        ]

    def correct_symbols(integer_symbols, system):
        """
        一百八 to 一百八十
        一亿一千三百万 to 一亿 一千万 三百万
        """

        if integer_symbols and isinstance(integer_symbols[0], CNU):
            if integer_symbols[0].power == 1:
                integer_symbols = [system.digits[1]] + integer_symbols

        if len(integer_symbols) > 1:
            if isinstance(integer_symbols[-1], CND) and isinstance(
                integer_symbols[-2], CNU
            ):
                integer_symbols.append(
                    CNU(integer_symbols[-2].power - 1, None, None, None, None)
                )

        result = []
        unit_count = 0
        for s in integer_symbols:
            if isinstance(s, CND):
                result.append(s)
                unit_count = 0
            elif isinstance(s, CNU):
                current_unit = CNU(s.power, None, None, None, None)
                unit_count += 1

            if unit_count == 1:
                result.append(current_unit)
            elif unit_count > 1:
                for i in range(len(result)):
                    if (
                        isinstance(result[-i - 1], CNU)
                        and result[-i - 1].power < current_unit.power
                    ):
                        result[-i - 1] = CNU(
                            result[-i - 1].power + current_unit.power,
                            None,
                            None,
                            None,
                            None,
                        )
        return result

    def compute_value(integer_symbols):
        """
        Compute the value.
        When current unit is larger than previous unit, current unit * all previous units will be used as all previous units.
        e.g. '两千万' = 2000 * 10000 not 2000 + 10000
        """
        value = [0]
        last_power = 0
        for s in integer_symbols:
            if isinstance(s, CND):
                value[-1] = s.value
            elif isinstance(s, CNU):
                value[-1] *= pow(10, s.power)
                if s.power > last_power:
                    value[:-1] = list(map(lambda v: v * pow(10, s.power), value[:-1]))
                    last_power = s.power
                value.append(0)
        return sum(value)

    system = create_system(numbering_type)
    int_part, dec_part = string2symbols(chinese_string, system)
    int_part = correct_symbols(int_part, system)
    int_str = str(compute_value(int_part))
    dec_str = "".join([str(d.value) for d in dec_part])
    if dec_part:
        return "{0}.{1}".format(int_str, dec_str)
    else:
        return int_str


def num2chn(
    number_string,
    numbering_type=NUMBERING_TYPES[1],
    big=False,
    traditional=False,
    alt_zero=False,
    alt_one=False,
    alt_two=True,
    use_zeros=True,
    use_units=True,
):

    def get_value(value_string, use_zeros=True):

        striped_string = value_string.lstrip("0")

        # record nothing if all zeros
        if not striped_string:
            return []

        # record one digits
        elif len(striped_string) == 1:
            if use_zeros and len(value_string) != len(striped_string):
                return [system.digits[0], system.digits[int(striped_string)]]
            else:
                return [system.digits[int(striped_string)]]

        # recursively record multiple digits
        else:
            result_unit = next(
                u for u in reversed(system.units) if u.power < len(striped_string)
            )
            result_string = value_string[: -result_unit.power]
            return (
                get_value(result_string)
                + [result_unit]
                + get_value(striped_string[-result_unit.power :])
            )

    system = create_system(numbering_type)

    int_dec = number_string.split(".")
    if len(int_dec) == 1:
        int_string = int_dec[0]
        dec_string = ""
    elif len(int_dec) == 2:
        int_string = int_dec[0]
        dec_string = int_dec[1]
    else:
        raise ValueError(
            "invalid input num string with more than one dot: {}".format(number_string)
        )

    if use_units and len(int_string) > 1:
        result_symbols = get_value(int_string)
    else:
        result_symbols = [system.digits[int(c)] for c in int_string]
    dec_symbols = [system.digits[int(c)] for c in dec_string]
    if dec_string:
        result_symbols += [system.math.point] + dec_symbols

    if alt_two:
        liang = CND(
            2,
            system.digits[2].alt_s,
            system.digits[2].alt_t,
            system.digits[2].big_s,
            system.digits[2].big_t,
        )
        for i, v in enumerate(result_symbols):
            if isinstance(v, CND) and v.value == 2:
                next_symbol = (
                    result_symbols[i + 1] if i < len(result_symbols) - 1 else None
                )
                previous_symbol = result_symbols[i - 1] if i > 0 else None
                if isinstance(next_symbol, CNU) and isinstance(
                    previous_symbol, (CNU, type(None))
                ):
                    if next_symbol.power != 1 and (
                        (previous_symbol is None) or (previous_symbol.power != 1)
                    ):
                        result_symbols[i] = liang

    # if big is True, '两' will not be used and `alt_two` has no impact on output
    if big:
        attr_name = "big_"
        if traditional:
            attr_name += "t"
        else:
            attr_name += "s"
    else:
        if traditional:
            attr_name = "traditional"
        else:
            attr_name = "simplified"

    result = "".join([getattr(s, attr_name) for s in result_symbols])

    # if not use_zeros:
    #     result = result.strip(getattr(system.digits[0], attr_name))

    if alt_zero:
        result = result.replace(
            getattr(system.digits[0], attr_name), system.digits[0].alt_s
        )

    if alt_one:
        result = result.replace(
            getattr(system.digits[1], attr_name), system.digits[1].alt_s
        )

    for i, p in enumerate(POINT):
        if result.startswith(p):
            return CHINESE_DIGIS[0] + result

    # ^10, 11, .., 19
    if (
        len(result) >= 2
        and result[1]
        in [
            SMALLER_CHINESE_NUMERING_UNITS_SIMPLIFIED[0],
            SMALLER_CHINESE_NUMERING_UNITS_TRADITIONAL[0],
        ]
        and result[0]
        in [
            CHINESE_DIGIS[1],
            BIG_CHINESE_DIGIS_SIMPLIFIED[1],
            BIG_CHINESE_DIGIS_TRADITIONAL[1],
        ]
    ):
        result = result[1:]

    return result


if __name__ == "__main__":

    # 测试程序
    all_chinese_number_string = (
        CHINESE_DIGIS
        + BIG_CHINESE_DIGIS_SIMPLIFIED
        + BIG_CHINESE_DIGIS_TRADITIONAL
        + LARGER_CHINESE_NUMERING_UNITS_SIMPLIFIED
        + LARGER_CHINESE_NUMERING_UNITS_TRADITIONAL
        + SMALLER_CHINESE_NUMERING_UNITS_SIMPLIFIED
        + SMALLER_CHINESE_NUMERING_UNITS_TRADITIONAL
        + ZERO_ALT
        + ONE_ALT
        + "".join(TWO_ALTS + POSITIVE + NEGATIVE + POINT)
    )

    print("num:", chn2num("一万零四百零三点八零五"))
    print("num:", chn2num("一亿六点三"))
    print("num:", chn2num("一亿零六点三"))
    print("num:", chn2num("两千零一亿六点三"))
    # print('num:', chn2num('一零零八六'))
    print("txt:", num2chn("10260.03", alt_zero=True))
    print("txt:", num2chn("20037.090", numbering_type="low", traditional=True))
    print("txt:", num2chn("100860001.77", numbering_type="high", big=True))
    print(
        "txt:",
        num2chn(
            "059523810880",
            alt_one=True,
            alt_two=False,
            use_lzeros=True,
            use_rzeros=True,
            use_units=False,
        ),
    )

    print(all_chinese_number_string)
