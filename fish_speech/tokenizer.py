import base64
import json
import logging
from pathlib import Path

import tiktoken

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
EOS_TOKEN = "<|end_of_text|>"
PAD_TOKEN = "<|pad|>"
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"

MODALITY_TEXT_TOKEN = "<|text|>"
MODALITY_VOICE_TOKEN = "<|voice|>"
MODALITY_INTERLEAVE_TOKEN = "<|interleave|>"
MODALITY_TOKENS = {
    "text": MODALITY_TEXT_TOKEN,
    "voice": MODALITY_VOICE_TOKEN,
    "interleave": MODALITY_INTERLEAVE_TOKEN,
}

PLACEHOLDER_TOKEN = [""] * 4
for i in range(4):
    PLACEHOLDER_TOKEN[i] = f"<|placeholder:{i}|>"

SEMANTIC_TOKEN_TEMPLATE = "<|semantic:{i}|>"
SEMANTIC_TOKENS = [SEMANTIC_TOKEN_TEMPLATE.format(i=i) for i in range(1024)]

# Warning: when you add a new special token, you should only add it to the end of the list.
ALL_SPECIAL_TOKENS = [
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    IM_START_TOKEN,
    IM_END_TOKEN,
    PLACEHOLDER_TOKEN[0],
    PLACEHOLDER_TOKEN[1],
    PLACEHOLDER_TOKEN[2],
    PLACEHOLDER_TOKEN[3],
    MODALITY_TEXT_TOKEN,
    MODALITY_VOICE_TOKEN,
    MODALITY_INTERLEAVE_TOKEN,
    *SEMANTIC_TOKENS,
]


class FishTokenizer:
    def __init__(self, model_path: str) -> None:
        mergeable_ranks = self.load_tiktoken_bpe(model_path)
        special_token_begin = len(mergeable_ranks)
        self.all_special_tokens_with_ids = {
            token: special_token_begin + i for i, token in enumerate(ALL_SPECIAL_TOKENS)
        }
        self.semantic_id_to_token_id = {
            i: self.all_special_tokens_with_ids[token]
            for i, token in enumerate(SEMANTIC_TOKENS)
        }
        self.semantic_begin_id = self.all_special_tokens_with_ids[SEMANTIC_TOKENS[0]]
        self.semantic_end_id = self.all_special_tokens_with_ids[SEMANTIC_TOKENS[-1]]

        self.tkt_model = tiktoken.core.Encoding(
            name=Path(model_path).stem,
            pat_str=FISH_TIKTOKEN_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.all_special_tokens_with_ids,
        )

    @staticmethod
    def load_tiktoken_bpe(tiktoken_bpe_file: str) -> dict[bytes, int]:
        data = {}
        for line in open(tiktoken_bpe_file).read().splitlines():
            if not line:
                continue
            token, rank = line.split()
            data[base64.b64decode(token)] = int(rank)
        return data

    def get_token_id(self, token: str) -> int:
        return self.all_special_tokens_with_ids[token]

    def encode(self, s: str, allowed_special: bool | set[str] = True) -> list[int]:
        assert isinstance(s, str)

        subs = []
        for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS):
            subs.append(s[i : i + TIKTOKEN_MAX_ENCODE_CHARS])

        if allowed_special is True:
            allowed_special = self.tkt_model.special_tokens_set
        elif allowed_special is False:
            allowed_special = set()

        return sum(
            self.tkt_model.encode_batch(
                subs, allowed_special=allowed_special, disallowed_special=set()
            ),
            start=[],
        )

    def decode(self, tokens: list[int]) -> str:
        return self.tkt_model.decode(tokens)

    def save_pretrained(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "tokenizer.tiktoken", "w") as f:
            for token, rank in self.tkt_model._mergeable_ranks.items():
                f.write(f"{base64.b64encode(token).decode()} {rank}\n")

        with open(path / "special_tokens.json", "w") as f:
            json.dump(
                self.all_special_tokens_with_ids,
                f,
                indent=2,
                ensure_ascii=False,
            )

    @staticmethod
    def from_pretrained(path: str):
        return FishTokenizer(Path(path) / "tokenizer.tiktoken")


if __name__ == "__main__":
    tokenizer = FishTokenizer("data/mpacks/v1.4-pretrain/tokenizer.all.tiktoken")
    tokenizer.save_pretrained("checkpoints/fish-speech-0.5B")
    tokenizer = FishTokenizer.from_pretrained("checkpoints/fish-speech-0.5B")

    print(
        [
            tokenizer.decode([i])
            for i in tokenizer.encode(f"{BOS_TOKEN}你好，世界！{EOS_TOKEN}")
        ]
    )
