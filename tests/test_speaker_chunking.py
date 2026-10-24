"""Tests for S2-Pro speaker-tag chunking helpers."""

import unittest

from fish_speech.models.text2semantic.inference import (
    group_turns_into_batches,
    split_text_by_speaker,
)


class SplitTextBySpeakerTest(unittest.TestCase):
    def test_text_before_first_tag_is_kept_as_speaker_0(self):
        # Previously "Hello there" was silently dropped.
        self.assertEqual(
            split_text_by_speaker("Hello there <|speaker:0|>world"),
            ["<|speaker:0|>Hello there", "<|speaker:0|>world"],
        )

    def test_normal_tagged_text_is_unchanged(self):
        self.assertEqual(
            split_text_by_speaker("<|speaker:0|>a<|speaker:1|>b"),
            ["<|speaker:0|>a", "<|speaker:1|>b"],
        )


class GroupTurnsIntoBatchesTest(unittest.TestCase):
    def test_repeated_speaker_is_not_split_by_speaker_cap(self):
        # One distinct speaker, well under max_bytes -> a single batch.
        turns = ["<|speaker:0|> line"] * 6
        self.assertEqual(
            len(group_turns_into_batches(turns, max_speakers=5, max_bytes=300)), 1
        )

    def test_distinct_speakers_still_respect_the_cap(self):
        turns = [f"<|speaker:{i}|> x" for i in range(6)]
        self.assertEqual(
            len(group_turns_into_batches(turns, max_speakers=5, max_bytes=300)), 2
        )

    def test_byte_limit_still_splits(self):
        turns = ["<|speaker:0|> " + "x" * 200, "<|speaker:0|> " + "y" * 200]
        self.assertEqual(
            len(group_turns_into_batches(turns, max_speakers=5, max_bytes=300)), 2
        )


if __name__ == "__main__":
    unittest.main()
