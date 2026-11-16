import errno
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
    path = Path(ref_text)
    try:
        is_file = path.is_file()
    except OSError as error:
        if error.errno != errno.ENAMETOOLONG:
            raise
        is_file = False
    if is_file:
        with path.open("r", encoding="utf-8") as file:
            return file.read()
    return ref_text


def list_files(
    path: Union[Path, str],
    extensions: set[str] | None = None,
    recursive: bool = False,
    sort: bool = True,
) -> list[Path]:
    """List files in a directory.

    Args:
        path (Path): Path to the directory.
        extensions (set, optional): Case-insensitive suffixes to filter. None or empty includes all files.
        recursive (bool, optional): Whether to search recursively. Defaults to False.
        sort (bool, optional): Whether to sort the files. Defaults to True.

    Returns:
        list: List of files.
    """

    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Directory {path} does not exist.")

    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    suffixes = tuple(extension.lower() for extension in extensions or ())
    globber = path.rglob if recursive else path.glob
    files = [
        file
        for file in globber("*")
        if file.is_file() and (not suffixes or file.name.lower().endswith(suffixes))
    ]

    if sort:
        files = natsorted(files)

    return files
