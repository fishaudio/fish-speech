import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "annotate_funasr.py"


def load_module():
    if not MODULE_PATH.is_file():
        raise AssertionError("FunASR annotation tool is not implemented")
    spec = importlib.util.spec_from_file_location("annotate_funasr", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise AssertionError("Unable to load FunASR annotation tool")
    spec.loader.exec_module(module)
    return module


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"audio")
    return path


class AnnotateFunASRTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_discovers_supported_audio_recursively_in_stable_order(self):
        expected = [
            touch(self.root / "a" / "first.WAV"),
            touch(self.root / "a" / "second.mp3"),
            touch(self.root / "b.flac"),
        ]
        touch(self.root / "ignored.ogg")
        touch(self.root / "note.txt")

        self.assertEqual(self.module.discover_audio_files(self.root), expected)

    def test_skips_existing_labels_without_loading_the_transcriber(self):
        audio = touch(self.root / "sample.wav")
        audio.with_suffix(".lab").write_text("curated text", encoding="utf-8")

        def must_not_run(_path):
            raise AssertionError("transcriber should stay lazy")

        stats = self.module.annotate_dataset(self.root, must_not_run)

        self.assertEqual(stats, self.module.AnnotationStats(discovered=1, skipped=1))
        self.assertEqual(
            audio.with_suffix(".lab").read_text(encoding="utf-8"), "curated text"
        )

    def test_dry_run_reports_work_without_loading_or_writing(self):
        audio = touch(self.root / "sample.mp3")

        def must_not_run(_path):
            raise AssertionError("transcriber should stay lazy")

        stats = self.module.annotate_dataset(self.root, must_not_run, dry_run=True)

        self.assertEqual(stats, self.module.AnnotationStats(discovered=1, planned=1))
        self.assertFalse(audio.with_suffix(".lab").exists())

    def test_writes_utf8_transcript_and_preserves_a_terminal_newline(self):
        audio = touch(self.root / "speaker" / "sample.flac")

        stats = self.module.annotate_dataset(
            self.root, lambda _path: "  你好，世界。  "
        )

        self.assertEqual(stats, self.module.AnnotationStats(discovered=1, processed=1))
        self.assertEqual(
            audio.with_suffix(".lab").read_bytes(), "你好，世界。\n".encode()
        )

    def test_normalizes_transcript_whitespace_to_one_line(self):
        audio = touch(self.root / "sample.wav")

        stats = self.module.annotate_dataset(
            self.root, lambda _path: "  first line\n\nsecond\tline  "
        )

        self.assertEqual(stats, self.module.AnnotationStats(discovered=1, processed=1))
        self.assertEqual(
            audio.with_suffix(".lab").read_text(encoding="utf-8"),
            "first line second line\n",
        )

    def test_overwrite_replaces_an_existing_label(self):
        audio = touch(self.root / "sample.wav")
        label = audio.with_suffix(".lab")
        label.write_text("old\n", encoding="utf-8")

        stats = self.module.annotate_dataset(
            self.root, lambda _path: "new", overwrite=True
        )

        self.assertEqual(stats, self.module.AnnotationStats(discovered=1, processed=1))
        self.assertEqual(label.read_text(encoding="utf-8"), "new\n")

    def test_one_failed_file_does_not_stop_the_remaining_dataset(self):
        first = touch(self.root / "a.wav")
        second = touch(self.root / "b.wav")
        errors = []

        def transcribe(path):
            if path == first:
                raise RuntimeError("damaged audio")
            return "usable"

        stats = self.module.annotate_dataset(
            self.root,
            transcribe,
            on_error=lambda path, error: errors.append((path, str(error))),
        )

        self.assertEqual(
            stats,
            self.module.AnnotationStats(discovered=2, processed=1, failed=1),
        )
        self.assertEqual(errors, [(first, "damaged audio")])
        self.assertFalse(first.with_suffix(".lab").exists())
        self.assertEqual(
            second.with_suffix(".lab").read_text(encoding="utf-8"), "usable\n"
        )

    def test_rejects_malformed_or_empty_funasr_results(self):
        results = [
            None,
            [],
            [{}],
            [{"text": None}],
            [{"text": ""}],
            [{"text": "<|Speech|>"}],
        ]
        for result in results:
            with self.subTest(result=result):
                with self.assertRaisesRegex(ValueError, "transcription text"):
                    self.module.extract_transcript(
                        result, lambda text: text.replace("<|Speech|>", "")
                    )

    def test_extracts_and_postprocesses_funasr_text(self):
        text = self.module.extract_transcript(
            [{"text": "<|zh|><|NEUTRAL|><|Speech|>欢迎使用 Fish Speech"}],
            lambda value: value.split(">", 3)[-1],
        )

        self.assertEqual(text, "欢迎使用 Fish Speech")


if __name__ == "__main__":
    unittest.main()
