from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

import torch
from transformers import PreTrainedTokenizerFast

from fish_speech.content_sequence import (
    AudioPart,
    BasePart,
    ContentSequence,
    EncodedMessage,
    TextPart,
    VQPart,
)
from fish_speech.tokenizer import IM_END_TOKEN, IM_START_TOKEN, MODALITY_TOKENS


@dataclass(kw_only=True)
class Message:
    role: Literal["system", "user", "assistant"]
    parts: list[BasePart] = field(default_factory=list)
    add_im_start: bool = True
    add_im_end: bool = True
    cal_loss: bool = False
    modality: Literal["text", "voice", "interleave"] | None = None

    # By default, ignore the loss of the auto-generated im_start token
    ignore_im_start_loss: bool = True


@dataclass
class Conversation:
    messages: list[Message]

    def __init__(self: "Conversation", messages: list[Message] | None = None):
        self.messages = messages or []

    def _build_content_sequence(
        self: "Conversation",
        metadata: dict | None = None,
    ) -> ContentSequence:
        """
        Build a ContentSequence from all messages.
        Handles cal_loss inheritance from message to part level.
        """
        all_parts = []
        for message in self.messages:
            # Add im_start
            if message.add_im_start:
                modality_token = (
                    MODALITY_TOKENS[message.modality] if message.modality else ""
                )
                all_parts.append(
                    TextPart(
                        text=f"{IM_START_TOKEN}{message.role}\n{modality_token}",
                        cal_loss=not message.ignore_im_start_loss,
                    )
                )

            # Add message parts
            for part in message.parts:
                # Inherit cal_loss from message if not set at part level
                if not hasattr(part, "cal_loss") or part.cal_loss is False:
                    new_part = deepcopy(part)
                    new_part.cal_loss = message.cal_loss
                    all_parts.append(new_part)
                else:
                    all_parts.append(part)

            # Add im_end
            if message.add_im_end:
                all_parts.append(
                    TextPart(text=IM_END_TOKEN + "\n", cal_loss=message.cal_loss)
                )

        return ContentSequence(parts=all_parts, modality=None, metadata=metadata)

    def encode(
        self: "Conversation",
        tokenizer: any,
        add_shift: bool = True,
        ignore_loss_tokens: list[str] = [],
        metadata: dict | None = None,
        max_length: int | None = None,
    ) -> EncodedMessage:
        # Build ContentSequence from messages
        content_seq = self._build_content_sequence(metadata=metadata)
        return content_seq.encode(
            tokenizer,
            add_shift=add_shift,
            ignore_loss_tokens=ignore_loss_tokens,
            max_length=max_length,
        )

    def encode_for_inference(
        self: "Conversation",
        tokenizer: any,
        num_codebooks: int,
        metadata: dict | None = None,
    ):
        content_seq = self._build_content_sequence(metadata=metadata)
        return content_seq.encode_for_inference(tokenizer, num_codebooks=num_codebooks)

    def visualize(
        self: "Conversation",
        tokenizer: PreTrainedTokenizerFast,
        ignore_loss_tokens: list[str] = [],
        merge_semantic_tokens: bool = False,
        merge_audio_tokens: bool = False,
        use_color: bool = True,
    ):
        """
        Visualize the encoded sequence with color-coded tokens.
        Blue/cyan tokens contribute to loss, green tokens do not.
        """
        # Build ContentSequence from messages and use its visualize method
        content_seq = self._build_content_sequence()
        content_seq.visualize(
            tokenizer,
            ignore_loss_tokens=ignore_loss_tokens,
            merge_semantic_tokens=merge_semantic_tokens,
        )

    def append(self: "Conversation", message: Message):
        self.messages.append(message)

    def to_content_sequence(
        self: "Conversation",
        metadata: dict | None = None,
    ) -> ContentSequence:
        """
        Convert the Conversation to a ContentSequence.

        This method builds a ContentSequence from all messages,
        handling cal_loss inheritance from message to part level.

        Args:
            metadata: Optional metadata to include in the ContentSequence

        Returns:
            ContentSequence with all messages converted to parts
        """
        return self._build_content_sequence(metadata=metadata)


if __name__ == "__main__":
    # Test the new implementation with the same API
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
    tokenizer = PreTrainedTokenizerFast.from_pretrained("checkpoints/agent-0.6b-debug")

    # Test with enhanced visualization from ContentSequence
    print("Basic visualization:")
    conversation.visualize(tokenizer)

    print("\nWith merged semantic tokens:")
    conversation.visualize(tokenizer, merge_semantic_tokens=True)

    print("\nWithout colors:")
    conversation.visualize(tokenizer, use_color=False)
