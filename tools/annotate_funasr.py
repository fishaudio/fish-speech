import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
from loguru import logger

AUDIO_EXTENSIONS = frozenset({".flac", ".mp3", ".wav"})


@dataclass(frozen=True)
class AnnotationStats:
    discovered: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    planned: int = 0


def discover_audio_files(dataset: Path) -> list[Path]:
    files = (
        path
        for path in dataset.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
    return sorted(
        files, key=lambda path: path.relative_to(dataset).as_posix().casefold()
    )


def extract_transcript(result: Any, postprocess: Callable[[str], str]) -> str:
    if not isinstance(result, list) or not result or not isinstance(result[0], dict):
        raise ValueError("FunASR result did not contain transcription text")

    raw_text = result[0].get("text")
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("FunASR result did not contain transcription text")

    text = postprocess(raw_text).strip()
    if not text:
        raise ValueError("FunASR result did not contain transcription text")
    return text


def write_label(path: Path, text: str) -> None:
    text = " ".join(text.split())
    if not text:
        raise ValueError("Refusing to write empty transcription text")

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(f"{text}\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def annotate_dataset(
    dataset: Path,
    transcribe: Callable[[Path], str],
    *,
    overwrite: bool = False,
    dry_run: bool = False,
    on_error: Callable[[Path, Exception], None] | None = None,
) -> AnnotationStats:
    files = discover_audio_files(dataset)
    processed = skipped = failed = planned = 0

    if on_error is None:
        on_error = lambda path, error: logger.error(
            f"Failed to annotate {path}: {error}"
        )

    for audio_path in files:
        label_path = audio_path.with_suffix(".lab")
        if label_path.exists() and not overwrite:
            skipped += 1
            continue
        if dry_run:
            planned += 1
            continue

        try:
            write_label(label_path, transcribe(audio_path))
            processed += 1
        except Exception as error:
            failed += 1
            on_error(audio_path, error)

    return AnnotationStats(
        discovered=len(files),
        processed=processed,
        skipped=skipped,
        failed=failed,
        planned=planned,
    )


class FunASRTranscriber:
    def __init__(self, model: str, device: str, language: str, use_itn: bool):
        self.model_name = model
        self.device = device
        self.language = language
        self.use_itn = use_itn
        self._model = None
        self._postprocess = None
        self._load_error = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if self._load_error is not None:
            raise self._load_error

        try:
            from funasr import AutoModel
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
        except ImportError as error:
            self._load_error = RuntimeError(
                "FunASR is required for annotation. Install it with "
                "`pip install 'funasr>=1.3.27,<2'`."
            )
            raise self._load_error from error

        self._model = AutoModel(
            model=self.model_name,
            device=self.device,
            disable_update=True,
        )
        self._postprocess = rich_transcription_postprocess

    def __call__(self, audio_path: Path) -> str:
        self._load()
        result = self._model.generate(
            input=str(audio_path),
            cache={},
            language=self.language,
            use_itn=self.use_itn,
        )
        return extract_transcript(result, self._postprocess)


@click.command()
@click.argument(
    "dataset",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--model", default="iic/SenseVoiceSmall", show_default=True)
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--language", default="auto", show_default=True)
@click.option("--itn/--no-itn", default=True, show_default=True)
@click.option("--overwrite", is_flag=True, help="Replace existing .lab files.")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Report files that would be annotated without loading a model.",
)
def main(
    dataset: Path,
    model: str,
    device: str,
    language: str,
    itn: bool,
    overwrite: bool,
    dry_run: bool,
) -> None:
    transcriber = FunASRTranscriber(model, device, language, itn)
    stats = annotate_dataset(
        dataset,
        transcriber,
        overwrite=overwrite,
        dry_run=dry_run,
    )
    click.echo(
        "Annotation summary: "
        f"discovered={stats.discovered}, processed={stats.processed}, "
        f"skipped={stats.skipped}, failed={stats.failed}, planned={stats.planned}"
    )
    if stats.failed:
        raise click.exceptions.Exit(1)


if __name__ == "__main__":
    main()
