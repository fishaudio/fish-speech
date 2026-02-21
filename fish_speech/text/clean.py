import re
import unicodedata

SYMBOLS_MAPPING = {
    "‘": "'",
    "’": "'",
}

REPLACE_SYMBOL_REGEX = re.compile(
    "|".join(re.escape(p) for p in SYMBOLS_MAPPING.keys())
)


EMOJI_REGEX = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f1e0-\U0001f1ff"  # flags (iOS)
    "]+",
    flags=re.UNICODE,
)


def clean_text(text):
    # Clean the text
    text = text.strip()

    # Apply NFC Unicode normalization to ensure consistent character representation
    # across all languages, including Arabic and other Unicode-based scripts.
    # Without this, the same word can appear with different Unicode encodings
    # (e.g., composed vs. decomposed forms), causing the tokenizer to produce
    # different tokens for visually identical text and hurting model accuracy.
    text = unicodedata.normalize("NFC", text)

    # Replace all chinese symbols with their english counterparts
    text = REPLACE_SYMBOL_REGEX.sub(lambda x: SYMBOLS_MAPPING[x.group()], text)

    # Remove emojis
    text = EMOJI_REGEX.sub(r"", text)

    # Remove continuous periods (...) and commas (,,,)
    # Also handle the Arabic comma (،) to avoid repeated punctuation
    text = re.sub(r"[,،]{2,}", lambda m: m.group()[0], text)

    return text
