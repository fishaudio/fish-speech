from dataclasses import dataclass, field
from typing import Literal

import torch
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedTokenizerFast

IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"
SEMANTIC_TOKEN = "<|semantic|>"
MEL_TOKEN = "<|mel|>"
PHONEME_START_TOKEN = "<|phoneme_start|>"
PHONEME_END_TOKEN = "<|phoneme_end|>"
ALL_SPECIAL_TOKENS = [
    IM_START_TOKEN,
    IM_END_TOKEN,
    SEMANTIC_TOKEN,
    MEL_TOKEN,
    PHONEME_START_TOKEN,
    PHONEME_END_TOKEN,
]

CODEBOOK_PAD_TOKEN_ID = 0


class FishTokenizerConfig(PretrainedConfig):
    share_codebook_embeddings: bool = True
    codebook_size: int = 1024
    num_codebooks: int = 8


class FishTokenizerFast(PreTrainedTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.share_codebook_embeddings = kwargs.pop("share_codebook_embeddings", True)
        self.codebook_size = kwargs.pop("codebook_size", 1024)
        self.num_codebooks = kwargs.pop("num_codebooks", 8)

AutoTokenizer.register(FishTokenizerConfig, fast_tokenizer_class=FishTokenizerFast)

@dataclass(kw_only=True)
class BasePart:
    pass


@dataclass(kw_only=True)
class VQPart(BasePart):
    codes: torch.Tensor


@dataclass(kw_only=True)
class TextPart(BasePart):
    text: str


@dataclass(kw_only=True)
class MelPart(BasePart):
    mels: torch.Tensor


@dataclass(kw_only=True)
class EncodedMessage:
    tokens: torch.Tensor
    labels: torch.Tensor
    vq_parts: list[torch.Tensor]
    mel_parts: list[torch.Tensor]
    vq_require_losses: torch.Tensor | None = None


@dataclass(kw_only=True)
class Message:
    role: Literal["system", "user", "assistant"]
    parts: list[VQPart | TextPart | MelPart] = field(default_factory=list)
    add_im_start: bool = True
    add_im_end: bool = True
    cal_loss: bool = False

    # By default, ignore the loss of the auto-generated im_start token
    ignore_im_start_loss: bool = True

    def encode(
        self: "Message",
        tokenizer: AutoTokenizer,
    ) -> EncodedMessage:
        all_tokens = []
        all_labels = []

        # Multi-modal tokens
        vq_parts = []
        mel_parts = []

        semantic_id, mel_id = tokenizer.convert_tokens_to_ids(
            [SEMANTIC_TOKEN, MEL_TOKEN]
        )

        parts = self.parts.copy()
        if self.add_im_start:
            parts.insert(0, TextPart(text=f"<|im_start|>{self.role}\n"))

        if self.add_im_end:
            parts.append(TextPart(text="<|im_end|>"))

        for part in parts:
            if isinstance(part, TextPart):
                tokens = tokenizer.encode(
                    part.text,
                    add_special_tokens=False,
                    truncation=False,
                    return_tensors="pt",
                ).int()[0]
            elif isinstance(part, VQPart):
                tokens = torch.zeros(part.codes.shape[1], dtype=torch.int) + semantic_id
                codes = part.codes.clone() + 1

                if getattr(tokenizer, "share_codebook_embeddings", True) is False:
                    for i in range(len(codes)):
                        codes[i] += tokenizer.codebook_size * i

                vq_parts.append(codes)
            elif isinstance(part, MelPart):
                tokens = torch.zeros(part.mels.shape[1], dtype=torch.int) + mel_id
                mel_parts.append(part.mels)
            else:
                raise ValueError(f"Unsupported part type: {type(part)}")

            all_tokens.append(tokens)
            if self.cal_loss:
                all_labels.append(tokens.clone())
            else:
                all_labels.append(torch.full_like(tokens, -100))

        tokens = torch.cat(all_tokens, dim=0)
        labels = torch.cat(all_labels, dim=0)
        assert tokens.shape == labels.shape

        if self.ignore_im_start_loss and self.add_im_start:
            labels[: len(all_tokens[0])] = -100

        return EncodedMessage(
            tokens=tokens,
            labels=labels,
            vq_parts=vq_parts,
            mel_parts=mel_parts,
        )


@dataclass
class Conversation:
    messages: list[Message]

    def encode(
        self: "Conversation",
        tokenizer: AutoTokenizer,
        add_shift: bool = True,
    ) -> EncodedMessage:
        # Build the input_ids and labels
        tokens = []
        labels = []
        vq_parts = []
        mel_parts = []
        vq_require_losses = []

        for message in self.messages:
            encoded = message.encode(
                tokenizer,
            )
            tokens.append(encoded.tokens)
            labels.append(encoded.labels)
            vq_parts.extend(encoded.vq_parts)
            mel_parts.extend(encoded.mel_parts)
            vq_require_losses.extend([message.cal_loss] * len(encoded.vq_parts))

        tokens = torch.cat(tokens, dim=0)
        labels = torch.cat(labels, dim=0)
        vq_require_losses = torch.tensor(vq_require_losses, dtype=torch.bool)

        if add_shift:
            tokens = tokens[:-1]
            labels = labels[1:]

        assert tokens.dtype in [
            torch.int,
            torch.long,
        ], f"Invalid dtype: {tokens.dtype}, conv: {conversation}"

        return EncodedMessage(
            tokens=tokens,
            labels=labels,
            vq_parts=vq_parts,
            mel_parts=mel_parts,
            vq_require_losses=vq_require_losses,
        )

    def encode_for_inference(
        self: "Conversation",
        tokenizer: AutoTokenizer,
        num_codebooks: int,
    ) -> EncodedMessage:
        encoded = self.encode(tokenizer, add_shift=False)
        tokens = encoded.tokens
        values = torch.zeros((num_codebooks + 1, len(tokens)), dtype=torch.int)
        values[0] = tokens

        if encoded.vq_parts is None or len(encoded.vq_parts) == 0:
            return values

        semantic_id, mel_id = tokenizer.convert_tokens_to_ids(
            [SEMANTIC_TOKEN, MEL_TOKEN]
        )
        vq_parts = encoded.vq_parts
        vq_parts = torch.cat(vq_parts, dim=1)
        values[1:, tokens == semantic_id] = vq_parts
        return values

    def visualize(self: "Conversation", tokenizer: AutoTokenizer):
        encoded = self.encode(tokenizer, add_shift=False)

        print_in_blue = lambda x: print("\033[94m" + x + "\033[0m", end="")
        print_in_green = lambda x: print("\033[92m" + x + "\033[0m", end="")

        for tok, lab in zip(encoded.tokens, encoded.labels):
            val = tokenizer.decode(tok, skip_special_tokens=False)
            if val == "\n":
                val = "\\n\n"

            if lab == -100:
                print_in_green(val)
            else:
                print_in_blue(val)

        print()


if __name__ == "__main__":
    message0 = Message(
        role="user",
        parts=[
            TextPart(text="Hello, how are you?"),
            VQPart(codes=torch.zeros((4, 10))),
        ],
        cal_loss=False,
    )

    message1 = Message(
        role="assistant",
        parts=[TextPart(text="I'm fine, thank you.")],
        cal_loss=True,
    )
    conversation = Conversation([message0, message1])
    tokenizer = AutoTokenizer.from_pretrained("checkpoints/Qwen2-1.5B-Instruct")
    conversation.visualize(tokenizer)

    encoded = conversation.encode(tokenizer)
    print(encoded)
    print(tokenizer.batch_decode(encoded.tokens))