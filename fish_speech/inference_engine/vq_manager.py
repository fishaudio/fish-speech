from typing import Callable

import torch
from loguru import logger

from fish_speech.models.dac.modded_dac import DAC


class VQManager:

    def __init__(self):
        # Make Pylance happy (attribut/method not defined...)
        self.decoder_model: DAC
        self.load_audio: Callable

    def decode_vq_tokens(self, codes):
        chunk_len = codes.shape[1] if codes.dim() >= 2 else 0
        logger.info("VQ features: {} (stream chunk={})", codes.shape, chunk_len)

        if isinstance(self.decoder_model, DAC):
            return self.decoder_model.from_indices(codes[None])[0].squeeze()

        raise ValueError(f"Unknown model type: {type(self.decoder_model)}")

    def encode_reference(self, reference_audio, enable_reference_audio):
        if enable_reference_audio and reference_audio is not None:
            # Load audios, and prepare basic info here
            if hasattr(self.decoder_model, "spec_transform"):
                sample_rate = self.decoder_model.spec_transform.sample_rate
            else:
                sample_rate = self.decoder_model.sample_rate
            reference_audio_content = self.load_audio(reference_audio, sample_rate)

            # Keep audio on CPU; run encoder on CPU to avoid OOM (encoder on long ref
            # can use ~14+ GB GPU and we're already near 16 GB after warmup).
            audios = torch.from_numpy(reference_audio_content)[None, None, :]
            audio_lengths = torch.tensor(
                [audios.shape[2]], dtype=torch.long
            )
            logger.info(
                f"Loaded audio with {audios.shape[2] / sample_rate:.2f} seconds"
            )

            if isinstance(self.decoder_model, DAC):
                device = getattr(self.decoder_model, "device", None)
                on_cuda = device is not None and str(device).startswith("cuda")
                if on_cuda:
                    torch.cuda.empty_cache()
                    self.decoder_model.to("cpu")
                    try:
                        prompt_tokens = self.decoder_model.encode(audios, audio_lengths)[0][0]
                    finally:
                        self.decoder_model.to(device)
                else:
                    prompt_tokens = self.decoder_model.encode(audios, audio_lengths)[0][0]
                logger.info("Encoded prompt: {}", prompt_tokens.shape)
            else:
                raise ValueError(f"Unknown model type: {type(self.decoder_model)}")
        else:
            prompt_tokens = None
            logger.info("No reference audio provided")

        return prompt_tokens
