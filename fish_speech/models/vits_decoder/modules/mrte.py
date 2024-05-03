import torch
from torch import nn
from torch.nn.utils import remove_weight_norm, weight_norm

from fish_speech.models.vits_decoder.modules.attentions import MultiHeadAttention


class MRTE(nn.Module):
    def __init__(
        self,
        content_enc_channels=192,
        hidden_size=512,
        out_channels=192,
        n_heads=4,
    ):
        super(MRTE, self).__init__()
        self.cross_attention = MultiHeadAttention(hidden_size, hidden_size, n_heads)
        self.c_pre = nn.Conv1d(content_enc_channels, hidden_size, 1)
        self.text_pre = nn.Conv1d(content_enc_channels, hidden_size, 1)
        self.c_post = nn.Conv1d(hidden_size, out_channels, 1)

    def forward(self, ssl_enc, ssl_mask, text, text_mask, ge, test=None):
        if ge == None:
            ge = 0
        attn_mask = text_mask.unsqueeze(2) * ssl_mask.unsqueeze(-1)

        ssl_enc = self.c_pre(ssl_enc * ssl_mask)
        text_enc = self.text_pre(text * text_mask)
        if test != None:
            if test == 0:
                x = (
                    self.cross_attention(
                        ssl_enc * ssl_mask, text_enc * text_mask, attn_mask
                    )
                    + ssl_enc
                    + ge
                )
            elif test == 1:
                x = ssl_enc + ge
            elif test == 2:
                x = (
                    self.cross_attention(
                        ssl_enc * 0 * ssl_mask, text_enc * text_mask, attn_mask
                    )
                    + ge
                )
            else:
                raise ValueError("test should be 0,1,2")
        else:
            x = (
                self.cross_attention(
                    ssl_enc * ssl_mask, text_enc * text_mask, attn_mask
                )
                + ssl_enc
                + ge
            )
        x = self.c_post(x * ssl_mask)
        return x
