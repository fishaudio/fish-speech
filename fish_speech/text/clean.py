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
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"
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