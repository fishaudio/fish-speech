import re

SYMBOLS_MAPPING = {
    "\n": ".",
    "…": ".",
    "“": "'",
    "”": "'",
    "‘": "'",
    "’": "'",
    "【": "",
    "】": "",
    "[": "",
    "]": "",
    "（": "",
    "）": "",
    "(": "",
    ")": "",
    "・": "",
    "·": "",
    "「": "'",
    "」": "'",
    "《": "'",
    "》": "'",
    "—": "",
    "～": "",
    "~": "",
    "：": ",",
    "；": ",",
    ";": ",",
    ":": ",",
}

REPLACE_SYMBOL_REGEX = re.compile(
    "|".join(re.escape(p) for p in SYMBOLS_MAPPING.keys())
)


def clean_text(text):
    # Clean the text
    text = text.strip()

    # Replace all chinese symbols with their english counterparts
    text = REPLACE_SYMBOL_REGEX.sub(lambda x: SYMBOLS_MAPPING[x.group()], text)

    return text
