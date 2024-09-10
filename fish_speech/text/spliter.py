import re
import string

from fish_speech.text.clean import clean_text


def utf_8_len(text):
    return len(text.encode("utf-8"))


def break_text(texts, length, splits: set):
    for text in texts:
        if utf_8_len(text) <= length:
            yield text
            continue

        curr = ""
        for char in text:
            curr += char

            if char in splits:
                yield curr
                curr = ""

        if curr:
            yield curr


def break_text_by_length(texts, length):
    for text in texts:
        if utf_8_len(text) <= length:
            yield text
            continue

        curr = ""
        for char in text:
            curr += char

            if utf_8_len(curr) >= length:
                yield curr
                curr = ""

        if curr:
            yield curr


def add_cleaned(curr, segments):
    curr = curr.strip()
    if curr and not all(c.isspace() or c in string.punctuation for c in curr):
        segments.append(curr)


def protect_float(text):
    # Turns 3.14 into <3_f_14> to prevent splitting
    return re.sub(r"(\d+)\.(\d+)", r"<\1_f_\2>", text)


def unprotect_float(text):
    # Turns <3_f_14> into 3.14
    return re.sub(r"<(\d+)_f_(\d+)>", r"\1.\2", text)


def split_text(text, length):
    text = clean_text(text)

    # Break the text into pieces with following rules:
    # 1. Split the text at ".", "!", "?" if text is NOT a float
    # 2. If the text is longer than length, split at ","
    # 3. If the text is still longer than length, split at " "
    # 4. If the text is still longer than length, split at any character to length

    texts = [text]
    texts = map(protect_float, texts)
    texts = break_text(texts, length, {".", "!", "?", "。", "！", "？"})
    texts = map(unprotect_float, texts)
    texts = break_text(texts, length, {",", "，"})
    texts = break_text(texts, length, {" "})
    texts = list(break_text_by_length(texts, length))

    # Then, merge the texts into segments with length <= length
    segments = []
    curr = ""

    for text in texts:
        if utf_8_len(curr) + utf_8_len(text) <= length:
            curr += text
        else:
            add_cleaned(curr, segments)
            curr = text

    if curr:
        add_cleaned(curr, segments)

    return segments


if __name__ == "__main__":
    # Test the split_text function

    text = "This is a test sentence. This is another test sentence. And a third one."

    assert split_text(text, 50) == [
        "This is a test sentence.",
        "This is another test sentence. And a third one.",
    ]
    assert split_text("a,aaaaaa3.14", 10) == ["a,", "aaaaaa3.14"]
    assert split_text("   ", 10) == []
    assert split_text("a", 10) == ["a"]

    text = "This is a test sentence with only commas, and no dots, and no exclamation marks, and no question marks, and no newlines."
    assert split_text(text, 50) == [
        "This is a test sentence with only commas,",
        "and no dots, and no exclamation marks,",
        "and no question marks, and no newlines.",
    ]

    text = "This is a test sentence This is a test sentence This is a test sentence. This is a test sentence, This is a test sentence, This is a test sentence."
    # First half split at " ", second half split at ","
    assert split_text(text, 50) == [
        "This is a test sentence This is a test sentence",
        "This is a test sentence. This is a test sentence,",
        "This is a test sentence, This is a test sentence.",
    ]

    text = "这是一段很长的中文文本,而且没有句号,也没有感叹号,也没有问号,也没有换行符。"
    assert split_text(text, 50) == [
        "这是一段很长的中文文本,",
        "而且没有句号,也没有感叹号,",
        "也没有问号,也没有换行符.",
    ]
