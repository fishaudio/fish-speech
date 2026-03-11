import base64
import json
import logging
from pathlib import Path
from typing import Union

import tiktoken
import torch

from fish_speech.tokenizer import (
    AUDIO_EMBED_TOKEN,
    AUDIO_END_TOKEN,
    AUDIO_START_TOKEN,
    IM_END_TOKEN,
    IM_START_TOKEN,
    MODALITY_INTERLEAVE_TOKEN,
    MODALITY_TEXT_TOKEN,
    MODALITY_VOICE_TOKEN,
    PAD_TOKEN,
    PHONEME_END_TOKEN,
    PHONEME_START_TOKEN,
    SEMANTIC_TOKEN_TEMPLATE,
)

logger = logging.getLogger(__name__)

# This is a modified version of the default pattern from GPT-4o, that better handles punctuations.
FISH_TIKTOKEN_PATTERN = "|".join(
    [
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
        r"\p{P}",
        r"[^\r\n\p{L}\p{N}]?\p{L}+",
        r"\p{N}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(\?!\S)",
        r"\s+",
    ]
)
TIKTOKEN_MAX_ENCODE_CHARS = 400_000

BOS_TOKEN = "<|begin_of_text|>"
EOS_TOKEN_S1 = "<|end_of_text|>"

ALL_SPECIAL_TOKENS = [
    BOS_TOKEN,
    EOS_TOKEN_S1,
    PAD_TOKEN,
    IM_START_TOKEN,
    IM_END_TOKEN,
    PHONEME_START_TOKEN,
    PHONEME_END_TOKEN,
    MODALITY_TEXT_TOKEN,
    MODALITY_VOICE_TOKEN,
    MODALITY_INTERLEAVE_TOKEN,
    AUDIO_START_TOKEN,
    AUDIO_END_TOKEN,
    AUDIO_EMBED_TOKEN,
    *[SEMANTIC_TOKEN_TEMPLATE.format(i=i) for i in range(4096)],
]


class FishTokenizerS1:
    def __init__(
        self,
        model_path: Union[str, Path],
        special_tokens: Union[list[str], dict[str, int]] = ALL_SPECIAL_TOKENS,
    ) -> None:
        mergeable_ranks = self.load_tiktoken_bpe(str(model_path))

        if isinstance(special_tokens, dict):
            self.all_special_tokens_with_ids = {
                token: int(token_id) for token, token_id in special_tokens.items()
            }
        else:
            special_token_begin = len(mergeable_ranks)
            self.all_special_tokens_with_ids = {
                token: special_token_begin + i
                for i, token in enumerate(special_tokens)
            }

        self.tkt_model = tiktoken.core.Encoding(
            name=Path(model_path).stem,
            pat_str=FISH_TIKTOKEN_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.all_special_tokens_with_ids,
        )

        self.semantic_id_to_token_id: dict[int, int] = {}
        valid_ids = []
        for code_idx in range(4096):
            token = SEMANTIC_TOKEN_TEMPLATE.format(i=code_idx)
            token_id = self.all_special_tokens_with_ids.get(token)
            if token_id is not None:
                self.semantic_id_to_token_id[code_idx] = token_id
                valid_ids.append(token_id)

        if not valid_ids:
            logger.error(
                "CRITICAL ERROR: No semantic tokens found in vocab! Audio cannot be synthesized."
            )
            self.semantic_begin_id = 0
            self.semantic_end_id = 0
            self.semantic_map_tensor = torch.zeros(4096, dtype=torch.long)
        else:
            self.semantic_begin_id = min(valid_ids)
            self.semantic_end_id = max(valid_ids)
            self.semantic_map_tensor = torch.zeros(4096, dtype=torch.long)
            for semantic_id, token_id in self.semantic_id_to_token_id.items():
                self.semantic_map_tensor[semantic_id] = token_id

        logger.info(
            f"Loaded S1 Tokenizer. Semantic Range: {self.semantic_begin_id} -> {self.semantic_end_id}"
        )

    @property
    def vocab_size(self) -> int:
        if self.tkt_model.max_token_value is not None:
            return int(self.tkt_model.max_token_value) + 1
        return len(self.tkt_model._mergeable_ranks) + len(
            self.all_special_tokens_with_ids
        )

    @property
    def pad_token_id(self) -> int | None:
        return self.all_special_tokens_with_ids.get(PAD_TOKEN)

    @property
    def eos_token_id(self) -> int | None:
        return self.all_special_tokens_with_ids.get(EOS_TOKEN_S1)

    @staticmethod
    def load_tiktoken_bpe(tiktoken_bpe_file: str) -> dict[bytes, int]:
        data = {}
        for line in Path(tiktoken_bpe_file).read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            token, rank = line.split()
            if token == "=":
                continue
            data[base64.b64decode(token)] = int(rank)
        return data

    def get_token_id(self, token: str) -> int:
        if token not in self.all_special_tokens_with_ids:
            raise KeyError(f"Token not found in S1 tokenizer: {token}")
        return self.all_special_tokens_with_ids[token]

    def encode(
        self, text: str, add_special_tokens: bool = False, **kwargs
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        allowed_special = kwargs.pop("allowed_special", self.tkt_model.special_tokens_set)
        if add_special_tokens:
            logger.debug(
                "FishTokenizerS1 ignores add_special_tokens=True; special tokens are parsed inline."
            )

        chunks = []
        for i in range(0, len(text), TIKTOKEN_MAX_ENCODE_CHARS):
            chunks.append(text[i : i + TIKTOKEN_MAX_ENCODE_CHARS])

        return sum(
            self.tkt_model.encode_batch(
                chunks,
                allowed_special=allowed_special,
                disallowed_special=set(),
            ),
            start=[],
        )

    def decode(self, tokens: list[int] | int, **kwargs) -> str:
        if isinstance(tokens, int):
            tokens = [tokens]
        return self.tkt_model.decode(tokens)

    def save_pretrained(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)

        with (target / "tokenizer.tiktoken").open("w", encoding="utf-8") as handle:
            for token, rank in self.tkt_model._mergeable_ranks.items():
                encoded = base64.b64encode(token).decode()
                if encoded == "":
                    encoded = "="
                handle.write(f"{encoded} {rank}\n")

        with (target / "special_tokens.json").open("w", encoding="utf-8") as handle:
            json.dump(
                self.all_special_tokens_with_ids,
                handle,
                indent=2,
                ensure_ascii=False,
            )

    @classmethod
    def from_pretrained(cls, path: str) -> "FishTokenizerS1":
        path_obj = Path(path)
        tokenizer_path = path_obj / "tokenizer.tiktoken"
        special_tokens_path = path_obj / "special_tokens.json"

        if not tokenizer_path.exists():
            raise FileNotFoundError(f"S1 tokenizer file not found: {tokenizer_path}")

        if special_tokens_path.exists():
            special_tokens = json.loads(special_tokens_path.read_text(encoding="utf-8"))
        else:
            special_tokens = ALL_SPECIAL_TOKENS

        return cls(tokenizer_path, special_tokens)
