from dataclasses import dataclass
from typing import Optional

import torch
from vector_quantize_pytorch import VectorQuantize
from torch import nn
from speech_lm.models.flash_whisper import (
    FlashWhisperForConditionalGeneration,
    FlashWhisperEncoderLayer,
)


@dataclass
class WhisperVQOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]

class WhisperVQ(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = "openai/whisper-medium",
        # Quantization
        codebook_dim: int = 32,
        codebook_size: int = 4096,
        codebook_decay: float = 0.9,
        threshold_ema_dead_code: int = 0,
        use_cosine_similarity: bool = True,
        downsample: bool = True,
        # Attention
        post_attention_depth: int = 2,
    ):
        super().__init__()

        self.whisper = FlashWhisperForConditionalGeneration.from_pretrained(
            model_name_or_path
        )

        # Freeze Whisper
        for param in self.whisper.parameters():
            param.requires_grad = False

        # Store vars
        self.downsample = downsample
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        # Pre-quantization
        whisper_config = self.whisper.model.config
        encoder_width = whisper_config.encoder_attention_heads * 64

        self.pre_ln = nn.LayerNorm(encoder_width)
        self.pre_mlp = nn.Sequential(
            nn.Linear(encoder_width, whisper_config.encoder_ffn_dim),
            nn.GELU(),
            nn.Linear(whisper_config.encoder_ffn_dim, encoder_width),
        )

        # Quantization
        self.quantizer = VectorQuantize(
            dim=encoder_width,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            decay=codebook_decay,
            commitment_weight=1.0,
            threshold_ema_dead_code=threshold_ema_dead_code,
            use_cosine_sim=use_cosine_similarity,
        )
        self.pad_embedding = nn.Parameter(torch.randn(encoder_width))

        # Post-quantization
        self.post_positional_embedding = nn.Embedding(
            whisper_config.max_source_positions, encoder_width
        )
        self.post_attention = nn.Sequential(
            *[
                FlashWhisperEncoderLayer(
                    config=whisper_config,
                )
                for _ in range(post_attention_depth)
            ]
        )
        self.post_ln = nn.LayerNorm(encoder_width)

    def encode(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "Attention mask must be 2D"
        
        # Whisper will downsample by 2
        attention_mask = attention_mask[:, ::2]

        with torch.no_grad():
            hidden_states = self.whisper.model.encoder(
                input_features,
            ).last_hidden_state

            x = hidden_states
            if self.downsample:
                x = x.reshape(x.shape[0], x.shape[1] // 2, 2, x.shape[2]).mean(dim=2)
                attention_mask = attention_mask[:, ::2]

        x = x + self.pre_mlp(self.pre_ln(x))
        quantized, indices, loss = self.quantizer(x, mask=attention_mask.bool())

        # Fill masked positions with pad embedding
        if attention_mask is not None:
            quantized[attention_mask == 0] = self.pad_embedding

        return quantized, indices, loss, hidden_states

    @torch.no_grad()
    def decode(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Upsample
        if self.downsample:
            hidden_states = hidden_states.repeat_interleave(2, dim=1)

        # Inject position embeddings
        positions = torch.arange(0, hidden_states.shape[1], dtype=torch.long, device=hidden_states.device)
        x = hidden_states + self.post_positional_embedding(positions)

        # Decode
        for layer in self.post_attention:
            x = layer(x, None, None)[0]
        hidden_states = self.post_ln(hidden_states)

        return hidden_states

    def forward(
        self,
        input_features: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        labels: torch.Tensor,
        # Audio, not used here
        input_values: Optional[torch.Tensor] = None,
    ) -> WhisperVQOutput:
        quantize, _, vq_loss, teacher_hidden_states = self.encode(
            input_features=input_features,
            attention_mask=encoder_attention_mask,
        )
        vq_hidden_states = self.decode(quantize)

        # student cross entropy loss
        outputs = self.whisper(
            encoder_outputs=(vq_hidden_states,),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        student_ce_loss = outputs.loss
        student_logits = outputs.logits

        # teacher cross entropy loss
        with torch.no_grad():
            outputs = self.whisper(
                encoder_outputs=(teacher_hidden_states,),
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
            )
            teacher_ce_loss = outputs.loss
            teacher_logits = outputs.logits

        # KL divergence
        kl_loss = nn.functional.kl_div(
            nn.functional.log_softmax(student_logits, dim=-1),
            nn.functional.softmax(teacher_logits, dim=-1),
            reduction="batchmean",
        )

        loss = vq_loss + student_ce_loss + kl_loss

        return WhisperVQOutput(loss=loss, metrics={
            "vq_loss": vq_loss,
            "student_ce_loss": student_ce_loss,
            "teacher_ce_loss": teacher_ce_loss,
            "kl_loss": kl_loss,
        })


if __name__ == "__main__":
    from transformers import WhisperProcessor
    from speech_lm.datasets.whisper_vq import WhisperVQDataset, WhisperVQCollator
    from torch.utils.data import DataLoader

    processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
    model = WhisperVQ()

    ds = WhisperVQDataset("filelists/whisper-vq.train.test.filelist", "openai/whisper-medium")
    loader = DataLoader(ds, batch_size=8, collate_fn=WhisperVQCollator())

    for batch in loader:
        output = model(**batch)
        print(output)
        break
