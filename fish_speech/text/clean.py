import re

SYMBOLS_MAPPING = {
    "‘": "'",
    "’": "'",
}

REPLACE_SYMBOL_REGEX = re.compile(
    "|".join(re.escape(p) for p in SYMBOLS_MAPPING.keys())
)

EMOJI_REGEX = re.compile(
    "["
    "\U0001f1e0-\U0001f1ff"  # flags (iOS)
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f700-\U0001f77f"  # alchemical symbols
    "\U0001f780-\U0001f7ff"  # Geometric Shapes Extended
    "\U0001f800-\U0001f8ff"  # Supplemental Arrows-C
    "\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
    "\U0001fa00-\U0001fa6f"  # Chess Symbols
    "\U0001fa70-\U0001faff"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027b0"  # Dingbats
    "\U000024c2-\U0001f251"
    "]+",
    flags=re.UNICODE,
)


def clean_text(text):
    # Clean the text
    text = text.strip()

    # Replace all Chinese symbols with their English counterparts
    text = REPLACE_SYMBOL_REGEX.sub(lambda x: SYMBOLS_MAPPING[x.group()], text)

    # Remove emojis
    text = EMOJI_REGEX.sub(r"", text)

    text = re.sub(r"[←→↑↓⇄⇅]+", "", text)  # Arrows
    text = re.sub(r"[\u0600-\u06FF]+", "", text)  # Arabic
    text = re.sub(r"[\u0590-\u05FF]+", "", text)  # Hebrew

    # Remove continuous periods (...) and commas (,,,)
    text = re.sub(r"[,]{2,}", lambda m: m.group()[0], text)

    return text
