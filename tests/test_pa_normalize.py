import unittest

from fish_speech.text.pa_normalize import normalize_punjabi


class TestPunjabiNormalize(unittest.TestCase):
    def test_addak_gemination(self):
        # ਗੱਲ ("gall") — addak geminates the consonant that follows it,
        # not the one it's written after.
        self.assertEqual(normalize_punjabi("ਗੱਲ"), "gall")

    def test_addak_word_initial(self):
        # ਅੱਜ ("ajj") — addak + word-final bare consonant (schwa dropped).
        self.assertEqual(normalize_punjabi("ਅੱਜ"), "ajj")

    def test_tippi_nasalization_on_vowel(self):
        # ਸਿੰਘ (Singh) — tippi nasalizes the preceding vowel.
        self.assertEqual(normalize_punjabi("ਸਿੰਘ"), "siṁgh")

    def test_tippi_nasalization_on_bare_consonant(self):
        # ਪੰਜਾਬ (Punjab) — tippi attaches directly to a bare
        # consonant's (kept) schwa.
        self.assertEqual(normalize_punjabi("ਪੰਜਾਬ"), "paṁjāb")

    def test_retroflex_consonants(self):
        # ਤੁਹਾਡੇ ("tuhāḍē") exercises the retroflex ਡ -> ḍ.
        self.assertEqual(normalize_punjabi("ਤੁਹਾਡੇ"), "tuhāḍē")

    def test_schwa_kept_isolated_medial_consonant(self):
        # ਵਧੀਆ ("vadhīā") — a single bare consonant not at word end
        # keeps its schwa.
        self.assertEqual(normalize_punjabi("ਵਧੀਆ"), "vadhīā")

    def test_schwa_alternation_in_consonant_run(self):
        # ਕਰਕੇ ("karkē") — two adjacent bare consonants: the one
        # closer to the end of the run drops its schwa, the next
        # one back keeps it.
        self.assertEqual(normalize_punjabi("ਕਰਕੇ"), "karkē")

    def test_word_final_schwa_deletion(self):
        # ਦਿਲ ("dil") — a lone bare consonant at the literal end of
        # the word drops its schwa.
        self.assertEqual(normalize_punjabi("ਦਿਲ"), "dil")

    def test_nukta_consonant(self):
        # ਖੁਸ਼ ("khuś") exercises the nukta-formed ਸ਼ -> ś.
        self.assertEqual(normalize_punjabi("ਖੁਸ਼"), "khuś")

    def test_independent_vowel_after_syllable(self):
        # ਗਿਆ ("giā") — a standalone independent vowel letter forming
        # its own syllable after a consonant-vowel syllable.
        self.assertEqual(normalize_punjabi("ਗਿਆ"), "giā")

    def test_non_gurmukhi_text_passes_through(self):
        self.assertEqual(normalize_punjabi("hello world 123"), "hello world 123")

    def test_mixed_script_input(self):
        self.assertEqual(normalize_punjabi("ਦਿਲ (heart)"), "dil (heart)")

    def test_full_example_sentence_doc_scheme(self):
        # The full worked example, rendered under this scheme's
        # consistent-ē rule.
        # "karke merā" (no macron) while this scheme writes
        # "karkē mērā" (consistent macron for ੇ). See
        # PUNJABI_TRANSLITERATION.md "Open question for
        # native-speaker review" — the difference is unresolved
        # pending native-speaker input, not a bug in this test.
        gurmukhi = (
            "ਅੱਜ ਮੌਸਮ ਬਹੁਤ ਵਧੀਆ ਹੈ, ਤੇ ਤੁਹਾਡੇ ਨਾਲ ਗੱਲ ਕਰਕੇ " "ਮੇਰਾ ਦਿਲ ਖੁਸ਼ ਹੋ ਗਿਆ।"
        )
        expected = (
            "ajj mausam bahut vadhīā hai, tē tuhāḍē nāl gall karkē "
            "mērā dil khuś hō giā."
        )
        self.assertEqual(normalize_punjabi(gurmukhi), expected)

    def test_virama_conjunct(self):
        # ਸ੍ਰੀ ("srī", a common name/loanword pattern) exercises
        # virama-based consonant clustering.
        self.assertEqual(normalize_punjabi("ਸ੍ਰੀ"), "srī")

    def test_control_tokens_pass_through_unchanged(self):
        # <|...|> control tokens must survive bit-for-bit; they are
        # parsed by FishTokenizer.encode with allowed_special="all"
        # and drive the conversation format.
        self.assertEqual(
            normalize_punjabi("<|speaker:0|>ਗੱਲ"),
            "<|speaker:0|>gall",
        )
        self.assertEqual(
            normalize_punjabi("<|im_start|>user\nਗੱਲ<|im_end|>"),
            "<|im_start|>user\ngall<|im_end|>",
        )
        self.assertEqual(
            normalize_punjabi("<|voice|> ਪੰਜਾਬ"),
            "<|voice|> paṁjāb",
        )

    def test_bracket_instruction_tags_pass_through(self):
        # Free-form [tag] syntax (e.g. [whisper], [excited]) is
        # plain text to the tokenizer and must not be altered.
        self.assertEqual(
            normalize_punjabi("[whisper] ਗੱਲ"),
            "[whisper] gall",
        )
        self.assertEqual(
            normalize_punjabi("[excited] [pause] ਅੱਜ"),
            "[excited] [pause] ajj",
        )

    def test_known_limitation_three_consonant_run(self):
        # KNOWN LIMITATION: the right-to-left alternation does not
        # generalize to runs of length >= 3. These tests pin the
        # current (linguistically incorrect) output so a future fix
        # is a visible, deliberate change rather than a silent
        # regression. See PUNJABI_TRANSLITERATION.md.
        self.assertEqual(normalize_punjabi("ਕਮਲ"), "kmal")  # target: "kamal"
        self.assertEqual(normalize_punjabi("ਕਲਮ"), "klam")  # target: "kalam"

    def test_single_bare_consonant_word(self):
        # A lone bare consonant at the literal end of a word drops
        # its schwa.
        self.assertEqual(normalize_punjabi("ਕ"), "k")
        # With a long-vowel matra it becomes a proper CV syllable.
        self.assertEqual(normalize_punjabi("ਕਾ"), "kā")

    def test_bare_consonant_with_nasal_and_deleted_schwa(self):
        # Degenerate but syntactically valid: a bare consonant with
        # only a tippi, at word end. Schwa drops; nasalization
        # remains. Pins current behaviour so a future edit is
        # visible.
        self.assertEqual(normalize_punjabi("ਗੰ"), "gṁ")

    def test_precomposed_nukta_letters(self):
        # Unicode has a single-codepoint form of the nukta letters
        # (e.g. U+0A36) as well as base letter + nukta (U+0A38 U+0A3C).
        # Both encodings should now give the same result.
        self.assertEqual(normalize_punjabi("\u0a16\u0a41\u0a36"), "khuś")
        self.assertEqual(
            normalize_punjabi("\u0a5b\u0a3f\u0a02\u0a26\u0a17\u0a40"), "ziṁdagī"
        )


if __name__ == "__main__":
    unittest.main()
