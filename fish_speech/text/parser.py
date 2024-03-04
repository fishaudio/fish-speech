import itertools
import re
import string
from typing import Optional

from fish_speech.text.chinese import g2p as g2p_chinese
from fish_speech.text.english import g2p as g2p_english
from fish_speech.text.japanese import g2p as g2p_japanese
from fish_speech.text.symbols import (
    language_id_map,
    language_unicode_range_map,
    punctuation,
    symbols_to_id,
)

LANGUAGE_TO_MODULE_MAP = {
    "ZH": g2p_chinese,
    "EN": g2p_english,
    "JP": g2p_japanese,
}

# This files is designed to parse the text, doing some normalization,
# and return an annotated text.
# Example: 1, 2, <JP>3</JP>, 4, <ZH>5</ZH>
# For better compatibility, we also support tree-like structure, like:
# 1, 2, <JP>3, <EN>4</EN></JP>, 5, <ZH>6</ZH>


class Segment:
    def __init__(self, text, language=None, phones=None):
        self.text = text
        self.language = language.upper() if language is not None else None

        if phones is None:
            phones = LANGUAGE_TO_MODULE_MAP[self.language](self.text)

        self.phones = phones

    def __repr__(self):
        return f"<Segment {self.language}: '{self.text}' -> '{' '.join(self.phones)}'>"

    def __str__(self):
        return self.text


SYMBOLS_MAPPING = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "“": "'",
    "”": "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "～": "-",
    "~": "-",
    "・": "-",
    "「": "'",
    "」": "'",
    ";": ",",
    ":": ",",
}

REPLACE_SYMBOL_REGEX = re.compile(
    "|".join(re.escape(p) for p in SYMBOLS_MAPPING.keys())
)
ALL_KNOWN_UTF8_RANGE = list(
    itertools.chain.from_iterable(language_unicode_range_map.values())
)
REMOVE_UNKNOWN_SYMBOL_REGEX = re.compile(
    "[^"
    + "".join(
        f"{re.escape(chr(start))}-{re.escape(chr(end))}"
        for start, end in ALL_KNOWN_UTF8_RANGE
    )
    + "]"
)


def clean_text(text):
    # Clean the text
    text = text.strip()
    # Replace <p:(.*?)> with <PPP(.*?)PPP>
    text = re.sub(r"<p:(.*?)>", r"<PPP\1PPP>", text)
    # Replace all chinese symbols with their english counterparts
    text = REPLACE_SYMBOL_REGEX.sub(lambda x: SYMBOLS_MAPPING[x.group()], text)
    text = REMOVE_UNKNOWN_SYMBOL_REGEX.sub("", text)
    # Replace <PPP(.*?)PPP> with <p:(.*?)>
    text = re.sub(r"<PPP(.*?)PPP>", r"<p:\1>", text)

    return text


def parse_text_to_segments(text, order=None):
    """
    Parse the text and return a list of segments.
    :param text: The text to be parsed.
    :param order: The order of languages. If None, use ["ZH", "JP", "EN"].
    :return: A list of segments.
    """

    if order is None:
        order = ["ZH", "JP", "EN"]

    order = [language.upper() for language in order]
    assert all(language in language_id_map for language in order)

    text = clean_text(text)
    texts = re.split(r"(<.*?>)", text)
    texts = [text for text in texts if text.strip() != ""]

    stack = []
    segments = []
    for text in texts:
        if text.startswith("<") and text.endswith(">") and text[1] != "/":
            current_language = text[1:-1]
            # The following line should be updated later
            assert (
                current_language.upper() in language_id_map
            ), f"Unknown language: {current_language}"
            stack.append(current_language)
        elif text.startswith("</") and text.endswith(">"):
            language = stack.pop()
            if language != text[2:-1]:
                raise ValueError(f"Language mismatch: {language} != {text[2:-1]}")
        elif stack:
            segments.append(Segment(text, stack[-1]))
        else:
            segments.extend(
                [i for i in parse_unknown_segment(text, order) if len(i.phones) > 0]
            )

    return segments


def parse_unknown_segment(text, order):
    last_idx, last_language = 0, None

    for idx, char in enumerate(text):
        if char == " " or char in punctuation or char in string.digits:
            # If the punctuation / number is in the middle of the text,
            # we should not split the text.
            detected_language = last_language or order[0]
        else:
            detected_language = None

            _order = order.copy()
            if last_language is not None:
                # Prioritize the last language
                _order.remove(last_language)
                _order.insert(0, last_language)

            for language in _order:
                for start, end in language_unicode_range_map[language]:
                    if start <= ord(char) <= end:
                        detected_language = language
                        break

                if detected_language is not None:
                    break

            assert (
                detected_language is not None
            ), f"Incorrect language: {char}, clean before calling this function."

        if last_language is None:
            last_language = detected_language

        if detected_language != last_language:
            yield Segment(text[last_idx:idx], last_language)
            last_idx = idx
            last_language = detected_language

    if last_idx != len(text):
        yield Segment(text[last_idx:], last_language)


def segments_to_phones(
    segments: list[Segment],
) -> tuple[tuple[Optional[str], str], list[int]]:
    phones = []
    ids = []

    for segment in segments:
        for phone in segment.phones:
            if phone.strip() == "":
                continue

            q0 = (segment.language, phone)
            q1 = (None, phone)

            if q0 in symbols_to_id:
                phones.append(q0)
                ids.append(symbols_to_id[q0])
            elif q1 in symbols_to_id:
                phones.append(q1)
                ids.append(symbols_to_id[q1])
            else:
                raise ValueError(f"Unknown phone: {segment.language} - {phone} -")

    return phones, ids


def g2p(text, order=None):
    segments = parse_text_to_segments(text, order=order)
    phones, _ = segments_to_phones(segments)
    return phones


if __name__ == "__main__":
    segments = parse_text_to_segments(
        "测试一下 Hugging face, BGM声音很大吗？那我改一下. <jp>世界、こんにちは。</jp>"  # noqa: E501
    )
    print(segments)

    segments = parse_text_to_segments(
        "测试一下 Hugging face, BGM声音很大吗？那我改一下. 世界、こんにちは。"  # noqa: E501
    )
    print(segments)

    print(
        clean_text(
            "测试一下 Hugging face, BGM声音很大吗？那我改一下. 世界、こんにちは。<p:123> <p:aH>"
        )
    )
