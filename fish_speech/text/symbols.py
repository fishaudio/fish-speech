from itertools import chain

punctuation = ["!", "?", "…", ",", ".", "'", "-"]
pu_symbols = punctuation + ["SP", "UNK"]
pad = "_"

# fmt: off
# Chinese (OpenCpop)
zh_symobls_consonants = [
    'AA', 'EE', 'OO', 'b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 
    'sh', 't', 'w', 'x', 'y', 'z', 'zh'
]

zh_symobls_vowels_no_tones = [
    'E', 'En', 'a', 'ai', 'an', 'ang', 'ao', 'e', 'ei', 'en', 'eng', 'er', 'i', 'i0', 'ia', 'ian', 'iang', 
    'iao', 'ie', 'in', 'ing', 'iong', 'ir', 'iu', 'o', 'ong', 'ou', 'u', 'ua', 'uai', 'uan', 'uang', 'ui', 
    'un', 'uo', 'v', 'van', 've', 'vn'
]

zh_symbols = zh_symobls_consonants + [f"{s}{t}" for s in zh_symobls_vowels_no_tones for t in range(1, 6)]


# Japanese (OpenJTalk)
jp_symbols = [
    'I', 'N', 'U', 'a', 'b', 'by', 'ch', 'cl', 'd', 'dy', 'e', 'f', 'g', 'gy', 'h', 'hy', 'i', 'j', 'k', 'ky',
    'm', 'my', 'n', 'ny', 'o', 'p', 'py', 'r', 'ry', 's', 'sh', 't', 'ts', 'u', 'v', 'w', 'y', 'z'
]

# English (ARPA)
en_symbols = [
    'AH0', 'S', 'AH1', 'EY2', 'AE2', 'EH0', 'OW2', 'UH0', 'NG', 'B', 'G', 'AY0', 'M', 'AA0', 'F', 'AO0', 'ER2', 'UH1', 'IY1', 'AH2', 'DH', 'IY0', 'EY1', 'IH0', 'K', 'N', 'W', 'IY2', 'T', 'AA1', 'ER1', 'EH2', 'OY0', 'UH2', 'UW1', 'Z', 'AW2', 'AW1', 'V', 'UW2', 'AA2', 'ER', 'AW0', 'UW0', 'R', 'OW1', 'EH1', 'ZH', 'AE0', 'IH2', 'IH', 'Y', 'JH', 'P', 'AY1', 'EY0', 'OY2', 'TH', 'HH', 'D', 'ER0', 'CH', 'AO1', 'AE1', 'AO2', 'OY1', 'AY2', 'IH1', 'OW0', 'L', 'SH'
]
# fmt: on

symbol_systems = [
    (None, [pad]),
    ("ZH", zh_symbols),
    ("JP", jp_symbols),
    ("EN", en_symbols),
    (None, pu_symbols),
]

symbols = list(
    chain.from_iterable(
        [(lang, s) for s in symbols] for lang, symbols in symbol_systems
    )
)

symbols_to_id = {s: i for i, s in enumerate(symbols)}

# language maps
language_id_map = {pad: 0, "ZH": 1, "JP": 2, "EN": 3}
language_unicode_range_map = {
    "ZH": [(0x4E00, 0x9FFF)],
    "JP": [(0x4E00, 0x9FFF), (0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)],
    "EN": [(0x0000, 0x007F)],
}
num_languages = len(language_id_map.keys())
