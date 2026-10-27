# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface-hub>=1.0,<2", "filelock>=3.13"]
# ///
"""Import verified local files into the shared HF Hub cache; never download weights."""

import argparse
import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from filelock import FileLock
from huggingface_hub import HfApi, constants
from huggingface_hub.utils import validate_repo_id


def verify(path, digest, size):
    if path.stat().st_size != size:
        raise ValueError(f"Size mismatch: {path}")
    hasher = hashlib.sha256() if len(digest) == 64 else hashlib.sha1()
    if len(digest) == 40:
        hasher.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            hasher.update(chunk)
    if hasher.hexdigest() != digest:
        raise ValueError(f"Checksum mismatch: {path}")


def safe_relative(name):
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or ":" in name
    ):
        raise ValueError(f"Unsafe Hub path: {name}")
    return Path(*path.parts)


def atomic_copy(source, destination, digest, size):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".import-")
    try:
        with os.fdopen(fd, "wb") as out, source.open("rb") as incoming:
            shutil.copyfileobj(incoming, out, length=8 * 1024 * 1024)
            out.flush()
            os.fsync(out.fileno())
        verify(Path(temporary), digest, size)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def migrate(
    source, repo_id, revision="main", cache_dir=None, move=False, dry_run=False
):
    validate_repo_id(repo_id)
    source = Path(source).expanduser().resolve()
    cache = Path(cache_dir or constants.HF_HUB_CACHE).expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"Not a directory: {source}")
    if source == cache or source in cache.parents or cache in source.parents:
        raise ValueError("Source and cache must be separate, non-nested directories")
    revision_path = safe_relative(revision)
    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    if not re.fullmatch(r"[0-9a-f]{40}", info.sha or ""):
        raise ValueError("Hub did not return an immutable commit hash")
    plans = []
    missing = []
    for entry in info.siblings:
        relative = safe_relative(entry.rfilename)
        local = source / relative
        if not local.exists():
            missing.append(entry.rfilename)
            continue
        if (
            local.is_symlink()
            or source not in local.resolve().parents
            or not local.is_file()
        ):
            raise ValueError(
                f"Source must be a regular file inside the source directory: {local}"
            )
        digest = entry.lfs.sha256 if entry.lfs else entry.blob_id
        if (
            not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", digest or "")
            or entry.size is None
        ):
            raise ValueError(f"Missing trusted checksum/size: {entry.rfilename}")
        verify(local, digest, entry.size)
        plans.append((relative, local, digest, entry.size))
    if not plans:
        raise ValueError(
            "No matching repository files found; check the model ID and source directory"
        )
    print(
        f"{repo_id}@{info.sha}: {len(plans)} verified files; {len(missing)} absent locally"
    )
    print(f"Cache: {cache}")
    if dry_run:
        for relative, _, _, _ in plans:
            print(f'Would {"move" if move else "copy"}: {relative}')
        return
    repo_folder = "models--" + repo_id.replace("/", "--")
    storage = cache / repo_folder
    snapshot = storage / "snapshots" / info.sha
    # Match Hub blob locks so concurrent hf downloads cannot see partial copies.
    for relative, local, digest, size in plans:
        lock = cache / ".locks" / repo_folder / (digest + ".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(lock)):
            blob = storage / "blobs" / digest
            if blob.exists():
                verify(blob, digest, size)
            else:
                atomic_copy(local, blob, digest, size)
            target = snapshot / relative
            if target.exists() or target.is_symlink():
                verify(target, digest, size)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    target.symlink_to(os.path.relpath(blob, target.parent))
                except OSError:
                    # Windows without symlink permission: a verified independent copy.
                    atomic_copy(blob, target, digest, size)
            verify(target, digest, size)
    if revision != info.sha:
        reference = storage / "refs" / revision_path
        reference.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=reference.parent, prefix=".import-")
        try:
            with os.fdopen(fd, "w") as out:
                out.write(info.sha)
            os.replace(temporary, reference)
        finally:
            Path(temporary).unlink(missing_ok=True)
    # No deletion until the entire import has succeeded. Recheck both sides.
    if move:
        for relative, local, digest, size in plans:
            verify(snapshot / relative, digest, size)
            verify(local, digest, size)
        for _, local, _, _ in plans:
            local.unlink()
    print(f"Imported snapshot: {snapshot}")
    print("Unrecognized files and local download metadata were left untouched.")
    if missing:
        print(
            "Partial snapshot: missing files will download on the next online model load."
        )
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--model", required=True, help="Exact Hub repository ID")
    parser.add_argument(
        "--revision", default="main", help="Revision matching the local files"
    )
    parser.add_argument(
        "--cache-dir", type=Path, help="Defaults to HF_HUB_CACHE / HF_HOME"
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Delete verified source files only after successful import",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify and report without writing or deleting",
    )
    args = parser.parse_args()
    try:
        migrate(
            args.source,
            args.model,
            args.revision,
            args.cache_dir,
            args.move,
            args.dry_run,
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Import failed: {exc}\n")


if __name__ == "__main__":
    main()
