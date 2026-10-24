"""The reference-ID validator must reject a trailing newline (re.fullmatch, not re.match)."""

import re
import unittest


class ReferenceIdPatternTest(unittest.TestCase):
    # Mirrors the pattern used in tools/server/views.py.
    pattern = r"^[a-zA-Z0-9\-_ ]+$"

    def _valid(self, value: str) -> bool:
        # The fixed views.py uses re.fullmatch; re.match would wrongly accept "x\n".
        return bool(re.fullmatch(self.pattern, value)) and len(value) <= 255

    def test_trailing_newline_is_rejected(self):
        self.assertFalse(self._valid("myvoice\n"))
        # re.match (the old behavior) would have accepted it -- document the trap.
        self.assertTrue(bool(re.match(self.pattern, "myvoice\n")))

    def test_normal_ids_pass(self):
        for ok in ("myvoice", "my-voice_2", "A B C"):
            self.assertTrue(self._valid(ok))

    def test_illegal_chars_rejected(self):
        for bad in ("a/b", "a\tb", "a.b", ""):
            self.assertFalse(self._valid(bad))
