import torch
import torch.nn.functional as F

from fish_speech.conversation import (
    CODEBOOK_PAD_TOKEN_ID,
)


def collate(tokens_list, max_length=torch.inf, end_of_text=None, pad_right=True):
    assert end_of_text
    tokens, attention_masks = [], []
    max_tokens_length = 0
    for _tokens in tokens_list:
        max_tokens_length = max(max_tokens_length, _tokens.size(1))
    max_tokens_length = min(max_tokens_length, max_length)
    for _tokens in tokens_list:
        _attention_mask = torch.ones((max_tokens_length,), dtype=torch.bool)
        tokens_length = _tokens.size(1)
        _attention_mask[:tokens_length] = False
        if tokens_length < max_tokens_length:
            _tokens = F.pad(
                _tokens,
                (0, max_tokens_length - tokens_length),
                value=end_of_text,  # self.tokenizer.get_token_id("<|end_of_text|>"),
            )
            _tokens[1:, tokens_length:] = CODEBOOK_PAD_TOKEN_ID

        tokens.append(_tokens)
        attention_masks.append(_attention_mask)
    tokens = torch.stack(tokens, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)
    return tokens, attention_masks
