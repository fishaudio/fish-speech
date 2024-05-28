from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from transformers import AutoTokenizer

CODEBOOK_PAD_TOKEN_ID = 0
CODEBOOK_EOS_TOKEN_ID = 1
SKIP_TEXT_STRING = "<|skip_text|>"


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    parts: list[str | torch.Tensor]
    mask_labels: bool = False


@dataclass
class Conversation:
    messages: list[Message]


def encode_message(
    message: Message,
    tokenizer: AutoTokenizer,
    num_codebooks: int,
    add_bos: bool = False,
):
    all_tokens = []
    all_codes = []
    semantic_id = tokenizer.convert_tokens_to_ids("<|semantic|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    parts = message.parts.copy()
    parts.insert(0, f"<|im_start|>{message.role}<|im_sep|>")

    if add_bos:
        parts.insert(0, tokenizer.bos_token)

    # Add eos token to the end of the message
    parts.append(f"<|im_end|>{tokenizer.eos_token}")

    for part in parts:
        if isinstance(part, str):
            tokens = tokenizer.encode(
                part,
                add_special_tokens=False,
                max_length=100000,
                truncation=False,
                return_tensors="pt",
            ).int()
            codes = (
                torch.zeros(num_codebooks, tokens.shape[1], dtype=torch.int)
                + CODEBOOK_PAD_TOKEN_ID
            )
            all_tokens.append(tokens)
            all_codes.append(codes)
            continue

        if isinstance(part, np.ndarray):
            part = torch.from_numpy(part)

        if isinstance(part, torch.Tensor):
            tokens = torch.zeros(1, part.shape[1], dtype=torch.int) + semantic_id
            codes = part
            all_tokens.append(tokens)
            all_codes.append(codes)
            continue

        raise ValueError(f"Unsupported part type: {type(part)}")

    tokens = torch.cat(all_tokens, dim=1)
    codes = torch.cat(all_codes, dim=1)
    tokens = torch.cat([tokens, codes], dim=0)

    labels = tokens.clone()
    input_ids = tokens.clone()

    # Mask codebook tokens when CODEBOOK_PAD_TOKEN_ID
    mask0 = (labels[0] == tokenizer.eos_token_id) | (labels[0] == im_end_id)
    labels[1:, mask0] = CODEBOOK_EOS_TOKEN_ID
    input_ids[1:, mask0] = CODEBOOK_EOS_TOKEN_ID

    mask = labels[1:] == CODEBOOK_PAD_TOKEN_ID
    labels[1:][mask] = -100

    # Mask
    if message.mask_labels:
        labels.fill_(-100)

    return input_ids[:, :-1], labels[:, 1:]


def encode_conversation(
    conversation: Conversation, tokenizer: AutoTokenizer, num_codebooks: int
):
    inputs = []
    labels = []

    for idx, message in enumerate(conversation.messages):
        input_ids, label = encode_message(
            message, tokenizer, num_codebooks, add_bos=idx == 0
        )
        inputs.append(input_ids)
        labels.append(label)

    tokens = torch.cat(inputs, dim=1)
    labels = torch.cat(labels, dim=1)

    assert tokens.dtype in [
        torch.int,
        torch.long,
    ], f"Invalid dtype: {tokens.dtype}, conv: {conversation}"

    return tokens, labels


if __name__ == "__main__":
    data = np.load("fake.npy")
    data = torch.from_numpy(data)
    message0 = Message(
        role="user",
        parts=["Hello world, transcribe the following audio:"],  # , data],
    )
    message1 = Message(
        role="assistant",
        parts=[""],
    )
    conversation = Conversation([message0, message1])
    tokenizer = AutoTokenizer.from_pretrained("checkpoints/fish-speech-agent-1")
    a = encode_conversation(conversation, tokenizer, 2)
    print(tokenizer.decode(a[0][0]))
    print(tokenizer.decode(a[1][0]))
    # print(encode_message(message, tokenizer, 2)[1])
