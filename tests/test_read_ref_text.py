"""Tests for read_ref_text path-vs-inline handling."""

import unittest

from fish_speech.utils.file import read_ref_text


class ReadRefTextTest(unittest.TestCase):
    def test_long_inline_text_is_returned_verbatim(self):
        # A real reference transcript easily exceeds the filesystem name limit (~255
        # bytes); probing it as a path used to raise OSError (ENAMETOOLONG).
        text = "A normal multi-sentence reference transcript. " * 8
        self.assertGreater(len(text), 255)
        self.assertEqual(read_ref_text(text), text)

    def test_existing_file_is_read(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ref.txt"
            p.write_text("from file", encoding="utf-8")
            self.assertEqual(read_ref_text(str(p)), "from file")

    def test_short_inline_text_is_returned(self):
        self.assertEqual(read_ref_text("just a short transcript"), "just a short transcript")


if __name__ == "__main__":
    unittest.main()
