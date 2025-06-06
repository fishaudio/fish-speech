from dataclasses import dataclass, field
from typing import List, Literal, Union

import numpy as np
import torch

from fish_speech.tokenizer import (
    IM_END_TOKEN,
    MODALITY_TOKENS,
    FishTokenizer,
)


def restore_ndarray(obj, to_tensor: bool = False):
    if isinstance(obj, dict) and "__ndarray__" in obj:
        obj = np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])

    if to_tensor and isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj.copy())

    return obj


@dataclass
class BasePart:
    type: Literal["text", "vq", "audio"] | None = None
    cal_loss: bool = False


@dataclass(kw_only=True)
class VQPart(BasePart):
    type = "vq"
    codes: torch.Tensor

    def __post_init__(self: "VQPart"):
        self.type = "vq"
        self.codes = restore_ndarray(self.codes, to_tensor=True)


@dataclass(kw_only=True)
class TextPart(BasePart):
    type = "text"
    text: str | None = None
    tokens: list[int] | None = None

    def __post_init__(self: "TextPart"):
        self.type = "text"
        if self.text is None and self.tokens is None:
            raise ValueError("Either text or tokens must be provided")


@dataclass(kw_only=True)
class AudioPart(BasePart):
    type = "audio"
    features: torch.Tensor

    def __post_init__(self: "AudioPart"):
        self.type = "audio"
        self.features = restore_ndarray(self.features, to_tensor=True)


@dataclass(kw_only=True)
class EncodedMessage:
    tokens: torch.Tensor
    labels: torch.Tensor
    vq_mask_tokens: torch.Tensor | None = None
    vq_mask_labels: torch.Tensor | None = None
    vq_parts: list[torch.Tensor]
    vq_require_losses: torch.Tensor | None = None
    audio_parts: list[torch.Tensor]
    audio_masks: torch.Tensor | None = None
    metadata: dict | None = None


@dataclass
class ContentSequence:
    """
    Flexible sequence of content parts that supports interleaved multimodal format.
    Example format: <|interleave|><|speaker:1|> TEXT AUDIO <|im_end|><|speaker:2|> TEXT AUDIO <|im_end|>
    """

    parts: list[BasePart] = field(default_factory=list)
    modality: Literal["text", "voice", "interleave"] | None = None
    metadata: dict | None = None

    def __init__(
        self: "ContentSequence",
        parts: list[BasePart | dict] | None = None,
        modality: Literal["text", "voice", "interleave"] | None = None,
        metadata: dict | None = None,
    ):
        self.modality = modality
        self.metadata = metadata or {}

        fixed_parts = []
        for part in parts or []:
            if isinstance(part, dict):
                if part["type"] == "vq":
                    part = VQPart(**part)
                elif part["type"] == "audio":
                    part = AudioPart(**part)
                elif part["type"] == "text":
                    part = TextPart(**part)
                else:
                    raise ValueError(f"Unsupported part type: {part['type']}")
            fixed_parts.append(part)

        self.parts = fixed_parts

        # If modality is specified, add it at the beginning if it's not already there
        if self.modality and not (
            len(self.parts) > 0
            and isinstance(self.parts[0], dict) is False
            and isinstance(self.parts[0], TextPart)
            and self.parts[0].text is not None
            and self.parts[0].text.startswith(MODALITY_TOKENS[self.modality])
        ):
            modality_token = MODALITY_TOKENS[self.modality]
            self.parts.insert(0, TextPart(text=modality_token))

    def append(
        self: "ContentSequence",
        part_or_parts: Union[BasePart, List[BasePart]],
        add_end: bool = False,
        speaker: Union[str, int] | None = None,
    ):
        """
        Append a part or list of parts to the sequence.

        Args:
            part_or_parts: A single part or list of parts to add
            add_end: Whether to add the IM_END_TOKEN after these parts
            speaker: Optional speaker identifier (name or ID) to add before the parts
        """
        # Convert single part to list
        parts_to_add = (
            [part_or_parts] if not isinstance(part_or_parts, list) else part_or_parts
        )

        # Add speaker token if specified
        if speaker is not None:
            speaker_token = f"<|speaker:{speaker}|>"
            self.parts.append(TextPart(text=speaker_token))

        # Add all the parts
        self.parts.extend(parts_to_add)

        # Add end token if requested
        if add_end:
            self.parts.append(
                TextPart(text=IM_END_TOKEN, cal_loss=self.parts[-1].cal_loss)
            )

    def encode(
        self: "ContentSequence",
        tokenizer: FishTokenizer,
        add_shift: bool = True,
        ignore_loss_tokens: list[str] = [],
    ) -> EncodedMessage:
        """
        Encode the sequence parts into tokens for the model.

        Args:
            tokenizer: The tokenizer to use
            add_shift: Whether to shift tokens for next-token prediction
            ignore_loss_tokens: List of token strings to ignore when calculating loss

        Returns:
            EncodedMessage with tensors ready for the model
        """
        all_tokens = []
        all_labels = []

        # Multi-modal elements
        vq_parts = []
        vq_masks = []
        vq_require_losses = []

        audio_parts = []
        audio_masks = []

        ignore_loss_token_ids = [tokenizer.get_token_id(i) for i in ignore_loss_tokens]

        for part in self.parts:
            if isinstance(part, TextPart):
                if part.tokens is None:
                    assert part.text is not None
                    tokens = tokenizer.encode(part.text)
                else:
                    tokens = part.tokens

                tokens = torch.tensor(tokens, dtype=torch.int)
            elif isinstance(part, VQPart):
                curr_codes = part.codes.clone().to(torch.int)
                tokens = torch.tensor(
                    [
                        tokenizer.semantic_id_to_token_id[int(i.item())]
                        for i in curr_codes[0].int()
                    ],
                    dtype=torch.int,
                )
                vq_parts.append(curr_codes)
                vq_require_losses.append(part.cal_loss)
            else:
                raise ValueError(f"Unsupported part type: {type(part)}")

            all_tokens.append(tokens)

            # Set masks for different part types
            if isinstance(part, VQPart):
                vq_masks.append(torch.ones_like(tokens, dtype=torch.bool))
                audio_masks.append(torch.zeros_like(tokens, dtype=torch.bool))
            elif isinstance(part, AudioPart):
                vq_masks.append(torch.zeros_like(tokens, dtype=torch.bool))
                audio_mask = torch.ones_like(tokens, dtype=torch.bool)
                audio_mask[0] = False  # Skip start token
                audio_mask[-1] = False  # Skip end token
                audio_masks.append(audio_mask)
            else:
                vq_masks.append(torch.zeros_like(tokens, dtype=torch.bool))
                audio_masks.append(torch.zeros_like(tokens, dtype=torch.bool))

            # Set labels based on whether we want to calculate loss for this part
            if part.cal_loss and not isinstance(part, AudioPart):
                all_labels.append(tokens.clone())
            else:
                all_labels.append(torch.full_like(tokens, -100))

        # Concatenate all tensors
        tokens = torch.cat(all_tokens, dim=0)
        labels = torch.cat(all_labels, dim=0)
        vq_masks = torch.cat(vq_masks, dim=0)
        audio_masks = torch.cat(audio_masks, dim=0)
        vq_require_losses = torch.tensor(vq_require_losses, dtype=torch.bool)

        # Apply shift if needed for next-token prediction
        vq_mask_tokens = vq_masks
        vq_mask_labels = vq_masks

        if add_shift:
            tokens = tokens[:-1]
            labels = labels[1:]
            vq_masks = vq_masks[:-1]
            vq_mask_tokens = vq_mask_tokens[:-1]
            vq_mask_labels = vq_mask_labels[1:]
            audio_masks = audio_masks[:-1]

        # Ignore specified tokens
        for i in ignore_loss_token_ids:
            assert i != -100 and i is not None
            labels[labels == i] = -100

        assert tokens.dtype in [
            torch.int,
            torch.long,
        ], f"Invalid dtype: {tokens.dtype}"

        return EncodedMessage(
            tokens=tokens,
            labels=labels,
            vq_parts=vq_parts,
            vq_mask_tokens=vq_mask_tokens,
            vq_mask_labels=vq_mask_labels,
            vq_require_losses=vq_require_losses,
            audio_parts=audio_parts,
            audio_masks=audio_masks,
            metadata=self.metadata,
        )

    def encode_for_inference(
        self: "ContentSequence",
        tokenizer: FishTokenizer,
        num_codebooks: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encode(tokenizer, add_shift=False)
        tokens = encoded.tokens
        values = torch.zeros((num_codebooks + 1, len(tokens)), dtype=torch.int)
        values[0] = tokens

        if (encoded.vq_parts is None or len(encoded.vq_parts) == 0) and (
            encoded.audio_parts is None or len(encoded.audio_parts) == 0
        ):
            return values, None, None

        audio_parts = audio_masks = None
        if encoded.vq_parts is not None and len(encoded.vq_parts) > 0:
            vq_parts = encoded.vq_parts
            vq_parts = torch.cat(vq_parts, dim=1)
            values[0, encoded.vq_mask_tokens] = (
                vq_parts[0] + tokenizer.semantic_begin_id
            )
            values[1:, encoded.vq_mask_tokens] = vq_parts

        if encoded.audio_parts is not None and len(encoded.audio_parts) > 0:
            audio_parts = torch.cat(encoded.audio_parts, dim=0)
            audio_masks = encoded.audio_masks[None, :]

        return values, audio_masks, audio_parts

    def visualize(
        self: "ContentSequence",
        tokenizer: FishTokenizer,
        ignore_loss_tokens: list[str] = [],
        merge_semantic_tokens: bool = False,
    ):
        """
        Visualize the encoded sequence with color-coded tokens.
        Blue/cyan tokens contribute to loss, green tokens do not.
        """
        encoded = self.encode(
            tokenizer, add_shift=False, ignore_loss_tokens=ignore_loss_tokens
        )

        # Colors for alternating tokens
        colors = {
            "blue": "\033[94m",  # Light blue
            "cyan": "\033[96m",  # Cyan
            "green": "\033[92m",  # Light green
            "dark_green": "\033[32m",  # Dark green
        }
        blue_idx = 0
        green_idx = 0

        def print_in_blue(x):
            nonlocal blue_idx
            color = colors["blue"] if blue_idx % 2 == 0 else colors["cyan"]
            print(f"{color}{x}\033[0m", end="")
            blue_idx += 1

        def print_in_green(x):
            nonlocal green_idx
            color = colors["green"] if green_idx % 2 == 0 else colors["dark_green"]
            print(f"{color}{x}\033[0m", end="")
            green_idx += 1

        def print_semantic_token(x, count):
            val = f"[<|semantic|>x{count}]"
            if x == -100:
                print_in_green(val)
            else:
                print_in_blue(val)

        count_semantic_tokens = 0
        semantic_label = None

        for tok, lab in zip(encoded.tokens, encoded.labels):
            token_id = int(tok.item())

            if merge_semantic_tokens:
                if (
                    tokenizer.semantic_begin_id <= token_id <= tokenizer.semantic_end_id
                    and (semantic_label is None or semantic_label == lab)
                ):
                    count_semantic_tokens += 1
                    semantic_label = lab
                    continue
                elif count_semantic_tokens > 0:
                    print_semantic_token(semantic_label, count_semantic_tokens)
                    count_semantic_tokens = 0
                    semantic_label = None

            val = tokenizer.decode([int(tok.item())])

            if lab == -100:
                print_in_green(val)
            else:
                print_in_blue(val)

        if merge_semantic_tokens and count_semantic_tokens > 0:
            print_semantic_token(semantic_label, count_semantic_tokens)

        print()
