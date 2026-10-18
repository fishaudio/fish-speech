import unittest

import torch

from fish_speech.models.text2semantic.fish_omni.chunking import (
    Drama3History,
    parse_speaker_segments,
    split_by_length,
    split_drama3_text,
    utf8_len,
)


class SplitByLengthTest(unittest.TestCase):
    def test_short_text_is_one_unchanged_chunk(self) -> None:
        self.assertEqual(split_by_length("  Hello there.  ", 300), ["  Hello there.  "])
        self.assertEqual(split_by_length(" \n ", 300), [])

    def test_prefers_sentence_ends(self) -> None:
        text = " ".join(f"Sentence number {i} is here." for i in range(40))
        chunks = split_by_length(text, 100)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(utf8_len(chunk), 100)
            self.assertTrue(chunk.endswith("."))
        self.assertEqual(" ".join(chunks), text)

        text = "今天天气很好，我们去公园散步。" * 20
        chunks = split_by_length(text, 100)
        for chunk in chunks:
            self.assertLessEqual(utf8_len(chunk), 100)
            self.assertTrue(chunk.endswith("。"))
        self.assertEqual("".join(chunks), text)

    def test_falls_back_to_clauses_spaces_then_length(self) -> None:
        clauses = "，".join(["这是一个很长的分句"] * 20)
        chunks = split_by_length(clauses, 60)
        self.assertTrue(all(chunk.endswith("，") for chunk in chunks[:-1]))
        self.assertEqual("".join(chunks), clauses)

        words = " ".join(["word"] * 100)
        chunks = split_by_length(words, 50)
        self.assertTrue(all(utf8_len(chunk) <= 50 for chunk in chunks))
        self.assertEqual(" ".join(chunks), words)

        unbroken = "字" * 100
        chunks = split_by_length(unbroken, 50)
        self.assertTrue(all(utf8_len(chunk) <= 50 for chunk in chunks))
        self.assertEqual("".join(chunks), unbroken)

    def test_keeps_decimals_and_protected_spans_whole(self) -> None:
        whisper = "<whisper>" + "secret words. " * 10 + "</whisper>"
        text = "Pi is 3.14159 roughly. " * 10 + whisper + " [laughing] done."
        chunks = split_by_length(text, 60)
        self.assertIn(whisper, chunks)
        self.assertGreater(utf8_len(whisper), 60)
        self.assertFalse(any(chunk.endswith("3.") for chunk in chunks))
        self.assertTrue(any("[laughing]" in chunk for chunk in chunks))


class SplitDrama3TextTest(unittest.TestCase):
    def test_parse_speaker_segments(self) -> None:
        self.assertEqual(
            parse_speaker_segments("intro <|speaker:1|>Hi<|speaker:2|> Yo"),
            [("0", "intro "), ("1", "Hi"), ("2", " Yo")],
        )

    def test_a_new_chunk_starts_at_the_fourth_speaker(self) -> None:
        text = (
            "<|speaker:0|>A1<|speaker:1|>B1<|speaker:0|>A2"
            "<|speaker:2|>C1<|speaker:3|>D1"
        )
        self.assertEqual(
            split_drama3_text(text, max_bytes=300, max_speakers=3),
            [
                "<|speaker:0|> A1\n<|speaker:1|> B1\n<|speaker:0|> A2\n<|speaker:2|> C1",
                "<|speaker:3|> D1",
            ],
        )

    def test_long_segments_are_cut_and_keep_their_tag(self) -> None:
        long = "This sentence is long enough. " * 10
        chunks = split_drama3_text(f"<|speaker:0|>{long}<|speaker:0|>Short.", 100)
        self.assertGreater(len(chunks), 2)
        segments = [segment for chunk in chunks for segment in chunk.split("\n")]
        self.assertTrue(all(s.startswith("<|speaker:0|> ") for s in segments))
        self.assertTrue(chunks[-1].endswith("Short."))

    def test_untagged_text_is_split_by_length(self) -> None:
        self.assertEqual(split_drama3_text("Hello."), ["Hello."])
        self.assertEqual(split_drama3_text("  "), [])


class Drama3HistoryTest(unittest.TestCase):
    def test_pins_the_first_chunk_and_keeps_the_latest(self) -> None:
        history = Drama3History(recent_turns=1)
        self.assertEqual(list(history.candidates()), [[]])
        for index in range(3):
            history.push(f"chunk {index}", torch.full((2, 1), index))
        texts = [[text for text, _ in turns] for turns in history.candidates()]
        self.assertEqual(texts, [["chunk 0", "chunk 2"], ["chunk 0"], []])


if __name__ == "__main__":
    unittest.main()
