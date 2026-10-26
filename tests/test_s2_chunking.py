import unittest

from fish_speech.models.text2semantic.fish_omni.chunking import utf8_len
from fish_speech.models.text2semantic.inference import split_s2_text


class S2ChunkingTest(unittest.TestCase):
    def test_short_untagged_text_stays_one_chunk(self) -> None:
        turns, batches = split_s2_text("Hello world.", 200)
        self.assertEqual(turns, [])
        self.assertEqual(batches, ["Hello world."])

    def test_untagged_text_honors_chunk_length(self) -> None:
        text = (
            "Good morning. I wanted to let you know that the furniture will "
            "arrive on Friday between nine and eleven. Please make sure the "
            "front gate is open, and call me if the delivery team needs help."
        )
        turns, batches = split_s2_text(text, 100)
        self.assertEqual(turns, [])
        self.assertGreater(len(batches), 1)
        for chunk in batches:
            self.assertLessEqual(utf8_len(chunk), 100)
        self.assertEqual(" ".join(batches).split(), text.split())

    def test_speaker_tagged_text_still_groups_turns(self) -> None:
        text = "<|speaker:0|>Hello there.<|speaker:1|>General Kenobi."
        turns, batches = split_s2_text(text, 200)
        self.assertEqual(len(turns), 2)
        self.assertEqual(len(batches), 1)
        self.assertIn("<|speaker:0|>", batches[0])
        self.assertIn("<|speaker:1|>", batches[0])


if __name__ == "__main__":
    unittest.main()
