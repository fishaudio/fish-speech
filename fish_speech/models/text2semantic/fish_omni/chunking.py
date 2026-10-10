"""Long-text chunking for S3, modeled on the hosted inference server.

Text is cut into chunks of about S3_CHUNK_BYTES UTF-8 bytes with at most
S3_CHUNK_MAX_SPEAKERS speakers, preferring sentence ends, then clause marks, then
spaces. Each chunk is generated on its own; the first chunk and the latest one go
back into the next prompt as user text plus assistant audio, which keeps the voice
consistent across chunks.
"""

import re
from collections import deque
from collections.abc import Iterator

import torch

from fish_speech.models.text2semantic.fish_omni.generate import (
    SPEAKER_TAG_PATTERN,
    UNICODE_WHITE_SPACE,
)

# A threshold, not a hard cap: a protected span longer than this stays whole.
S3_CHUNK_BYTES = 300
S3_CHUNK_MAX_SPEAKERS = 3
# Frames per chunk for max_new_tokens <= 0, about 27.9 s at 2048 samples per
# frame and 44.1 kHz.
S3_CHUNK_MAX_NEW_TOKENS = 600
# History besides the pinned first chunk.
S3_HISTORY_RECENT_TURNS = 1

SENTENCE_ENDS = frozenset(".!?;。！？；…\n")
CLAUSE_MARKS = frozenset(",:，：、")
SPACES = frozenset(" ")
# Never split inside: special tokens, paired tags such as <whisper>…</whisper>, and
# [bracketed] cues.
PROTECTED_SPAN = re.compile(
    r"<\|[^|>]*\|>|<([A-Za-z][\w-]*)\b[^>]*>.*?</\1>|\[[^\[\]]*\]", re.S
)

Piece = tuple[str, bool]  # (text, protected)
HistoryTurn = tuple[str, torch.Tensor]  # (chunk text, codes [num_codebooks, frames])


def utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _protected_pieces(text: str) -> list[Piece]:
    pieces, last = [], 0
    for match in PROTECTED_SPAN.finditer(text):
        if match.start() > last:
            pieces.append((text[last : match.start()], False))
        pieces.append((match.group(0), True))
        last = match.end()
    if last < len(text):
        pieces.append((text[last:], False))
    return pieces


def _breaks_at(text: str, index: int, marks: frozenset[str]) -> bool:
    if text[index] not in marks:
        return False
    # Keep decimals such as 3.14 in one piece.
    return not (
        text[index] == "."
        and 0 < index < len(text) - 1
        and text[index - 1].isdigit()
        and text[index + 1].isdigit()
    )


def _break_at_marks(pieces: list[Piece], length: int, marks: frozenset[str]):
    """Split free pieces longer than length bytes after each mark, keeping it."""
    out = []
    for text, protected in pieces:
        if protected or utf8_len(text) <= length:
            out.append((text, protected))
            continue
        start = 0
        for index in range(len(text)):
            if _breaks_at(text, index, marks):
                out.append((text[start : index + 1], False))
                start = index + 1
        if start < len(text):
            out.append((text[start:], False))
    return out


def _break_by_length(pieces: list[Piece], length: int) -> list[Piece]:
    """Split the free pieces still longer than length on character boundaries."""
    out = []
    for text, protected in pieces:
        if protected or utf8_len(text) <= length:
            out.append((text, protected))
            continue
        current, size = "", 0
        for char in text:
            if current and size + utf8_len(char) > length:
                out.append((current, False))
                current, size = "", 0
            current += char
            size += utf8_len(char)
        out.append((current, False))
    return out


def split_by_length(text: str, length: int = S3_CHUNK_BYTES) -> list[str]:
    """Chunks of about length bytes, packed sentence by sentence.

    Text is cut at every sentence end; a sentence still longer than length is cut
    at clause marks, then spaces, then on raw length. Adjacent pieces are then
    merged back up to length.
    """
    if not text.strip(UNICODE_WHITE_SPACE):
        return []
    if utf8_len(text) <= length:
        return [text]
    pieces = _break_at_marks(_protected_pieces(text), 0, SENTENCE_ENDS)
    for marks in (CLAUSE_MARKS, SPACES):
        pieces = _break_at_marks(pieces, length, marks)
    pieces = _break_by_length(pieces, length)

    chunks, current = [], ""
    for piece, _ in pieces:
        if current and utf8_len(current + piece) > length:
            chunks.append(current)
            current = ""
        current += piece
    chunks.append(current)
    return [
        chunk.strip(UNICODE_WHITE_SPACE)
        for chunk in chunks
        if chunk.strip(UNICODE_WHITE_SPACE)
    ]


def parse_speaker_segments(text: str) -> list[tuple[str, str]]:
    """(speaker id, text) per tagged segment; text before the first tag is speaker 0."""
    parts = SPEAKER_TAG_PATTERN.split(text)
    segments = [("0", parts[0])] if parts[0].strip(UNICODE_WHITE_SPACE) else []
    for speaker, body in zip(parts[1::2], parts[2::2]):
        if body.strip(UNICODE_WHITE_SPACE):
            segments.append((speaker, body))
    return segments


def split_s3_text(
    text: str,
    max_bytes: int = S3_CHUNK_BYTES,
    max_speakers: int = S3_CHUNK_MAX_SPEAKERS,
) -> list[str]:
    """Generation chunks for text, in order.

    Untagged text is split by length. Tagged text is split into speaker segments
    (long ones cut further, each piece keeping its tag) and regrouped: a new chunk
    starts when a segment would bring in one speaker too many or exceed max_bytes,
    so A, B, A can share a chunk.
    """
    if SPEAKER_TAG_PATTERN.search(text) is None:
        return split_by_length(text, max_bytes)

    segments = []
    for speaker, body in parse_speaker_segments(text):
        for piece in split_by_length(body, max_bytes):
            segments.append((speaker, f"<|speaker:{speaker}|> {piece}"))

    chunks, batch, speakers, size = [], [], [], 0
    for speaker, segment in segments:
        new_speaker = speaker not in speakers
        too_many_speakers = new_speaker and len(speakers) >= max_speakers
        too_long = bool(batch) and size + utf8_len(segment) > max_bytes
        if too_many_speakers or too_long:
            chunks.append("\n".join(batch))
            batch, speakers, size = [], [], 0
            new_speaker = True
        batch.append(segment)
        if new_speaker:
            speakers.append(speaker)
        size += utf8_len(segment)
    if batch:
        chunks.append("\n".join(batch))
    return chunks


class S3History:
    """Context for the next chunk: the first chunk stays pinned, plus the latest
    S3_HISTORY_RECENT_TURNS chunks."""

    def __init__(self, recent_turns: int = S3_HISTORY_RECENT_TURNS) -> None:
        self.pinned: HistoryTurn | None = None
        self.recent: deque[HistoryTurn] = deque(maxlen=recent_turns)

    def push(self, text: str, codes: torch.Tensor) -> None:
        if self.pinned is None:
            self.pinned = (text, codes)
        else:
            self.recent.append((text, codes))

    def candidates(self) -> Iterator[list[HistoryTurn]]:
        """Histories to try, longest first: drop the oldest recent turns, then the
        pinned one, ending with no history."""
        pinned = [self.pinned] if self.pinned is not None else []
        recent = list(self.recent)
        for start in range(len(recent) + 1):
            yield pinned + recent[start:]
        if pinned:
            yield []
