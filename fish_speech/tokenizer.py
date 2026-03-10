import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import torch
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

# Constants definitions
EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"
PHONEME_START_TOKEN = "<|phoneme_start|>"
PHONEME_END_TOKEN = "<|phoneme_end|>"

MODALITY_TEXT_TOKEN = "<|text|>"
MODALITY_VOICE_TOKEN = "<|voice|>"
MODALITY_INTERLEAVE_TOKEN = "<|interleave|>"
AUDIO_START_TOKEN = "<|audio_start|>"
AUDIO_END_TOKEN = "<|audio_end|>"
AUDIO_EMBED_TOKEN = "<|audio_pad|>"

MODALITY_TOKENS = {
    "text": MODALITY_TEXT_TOKEN,
    "voice": MODALITY_VOICE_TOKEN,
    "interleave": MODALITY_INTERLEAVE_TOKEN,
}

SEMANTIC_TOKEN_TEMPLATE = "<|semantic:{i}|>"
SEMANTIC_TOKENS = [SEMANTIC_TOKEN_TEMPLATE.format(i=i) for i in range(4096)]

ALL_SPECIAL_TOKENS = [
    EOS_TOKEN,
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
    *SEMANTIC_TOKENS,
]


class FishTokenizer:
    def __init__(self, model_path: str):
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.semantic_id_to_token_id = {}

        vocab = self._tokenizer.get_vocab()
        valid_ids = []

        for code_idx in range(4096):
            token = SEMANTIC_TOKEN_TEMPLATE.format(i=code_idx)
            if token in vocab:
                token_id = vocab[token]
                self.semantic_id_to_token_id[code_idx] = token_id
                valid_ids.append(token_id)

        if not valid_ids:
            logger.error(
                "CRITICAL ERROR: No semantic tokens found in vocab! Audio cannot be synthesized."
            )
            self.semantic_begin_id = 0
            self.semantic_end_id = 0
            # Dummy tensor to prevent crash, though generation will fail
            self.semantic_map_tensor = torch.zeros(4096, dtype=torch.long)
        else:
            self.semantic_begin_id = min(valid_ids)
            self.semantic_end_id = max(valid_ids)
            # Create a lookup tensor to handle potential gaps in token IDs safely
            self.semantic_map_tensor = torch.zeros(4096, dtype=torch.long)
            for k, v in self.semantic_id_to_token_id.items():
                self.semantic_map_tensor[k] = v

        logger.info(
            f"Loaded Tokenizer. Semantic Range: {self.semantic_begin_id} -> {self.semantic_end_id}"
        )

    @property
    def vocab_size(self):
        return self._tokenizer.vocab_size

    @property
    def pad_token_id(self):
        return self._tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        return self._tokenizer.eos_token_id

    def get_token_id(self, token: str) -> int:
        return self._tokenizer.convert_tokens_to_ids(token)

    def encode(
        self, text: str, add_special_tokens: bool = False, **kwargs
    ) -> List[int]:
        # [FIX] Force Qwen/Tiktoken backends to parse special tokens inline
        import inspect

        sig = inspect.signature(self._tokenizer.encode)
        if "allowed_special" in sig.parameters and "allowed_special" not in kwargs:
            kwargs["allowed_special"] = "all"
        return self._tokenizer.encode(
            text, add_special_tokens=add_special_tokens, **kwargs
        )

    def decode(self, tokens: Union[List[int], int], **kwargs) -> str:
        return self._tokenizer.decode(tokens, **kwargs)

    def save_pretrained(self, path: str):
        self._tokenizer.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str):
        return cls(path)

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)
