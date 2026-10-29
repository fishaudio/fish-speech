from pathlib import Path
from typing import Union

from natsort import natsorted

AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a",
    ".wma",
    ".aac",
    ".aiff",
    ".aif",
    ".aifc",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
}


def audio_to_bytes(file_path):
    if not file_path or not Path(file_path).exists():
        return None
    with open(file_path, "rb") as wav_file:
        wav = wav_file.read()
    return wav


def read_ref_text(ref_text):
    # ref_text may be a path to a transcript file OR an inline transcript. Probing it as
    # a path can raise (e.g. OSError ENAMETOOLONG when an inline transcript exceeds the
    # filesystem's name limit, ~255 bytes) -- a normal reference transcript easily does.
    # Treat any path error as "this is inline text" and return it verbatim.
    try:
        path = Path(ref_text)
        if path.exists() and path.is_file():
            with path.open("r", encoding="utf-8") as file:
                return file.read()
    except OSError:
        pass
    return ref_text


def list_files(
    path: Union[Path, str],
    extensions: set[str] = set(),
    recursive: bool = False,
    sort: bool = True,
) -> list[Path]:
    """List files in a directory.

    Args:
        path (Path): Path to the directory.
        extensions (set, optional): Extensions to filter. Defaults to None.
        recursive (bool, optional): Whether to search recursively. Defaults to False.
        sort (bool, optional): Whether to sort the files. Defaults to True.

    Returns:
        list: List of files.
    """

    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Directory {path} does not exist.")

    globber = path.rglob if recursive else path.glob
    files = [file for ext in extensions for file in globber(f"*{ext}")]

    if sort:
        files = natsorted(files)

    return files
