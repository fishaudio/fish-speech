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
        logger.info(f"VQ features: {codes.shape}")

        if isinstance(self.decoder_model, DAC):
            return self.decoder_model.from_indices(codes[None])[0].squeeze()

        raise ValueError(f"Unknown model type: {type(self.decoder_model)}")

    def encode_reference(self, reference_audio, enable_reference_audio, encoder_device="gpu"):
        if enable_reference_audio and reference_audio is not None:
            # Load audios, and prepare basic info here
            if hasattr(self.decoder_model, "spec_transform"):
                sample_rate = self.decoder_model.spec_transform.sample_rate
            else:
                sample_rate = self.decoder_model.sample_rate
            reference_audio_content = self.load_audio(reference_audio, sample_rate)

            encoder = self.decoder_model
            if encoder_device == "cpu":
                encoder = getattr(self, "_cpu_reference_encoder", None)
                if encoder is None:
                    settings = getattr(self, "_reference_codec_config", None)
                    if settings is None:
                        raise ValueError("CPU reference encoding requires configured Codec checkpoint metadata")
                    from fish_speech.models.dac.inference import load_model
                    encoder = load_model(settings[1], settings[0], device="cpu").float().eval()
                    # Only this newly loaded CPU copy is trimmed. The original
                    # decoder/LLM stay on their configured devices, untouched.
                    del encoder.decoder
                    self._cpu_reference_encoder = encoder
            elif encoder_device != "gpu":
                raise ValueError("Unknown reference encoder device")
            audios = torch.from_numpy(reference_audio_content).to(
                device=encoder.device, dtype=next(encoder.parameters()).dtype,
            )[None, None, :]
            audio_lengths = torch.tensor(
                [audios.shape[2]], device=encoder.device, dtype=torch.long
            )
            logger.info(
                f"Loaded audio with {audios.shape[2] / sample_rate:.2f} seconds"
            )

            # VQ Encoder
            if isinstance(encoder, DAC):
                with torch.inference_mode():
                    prompt_tokens = encoder.encode(audios, audio_lengths)[0][0]
                logger.info(f"Encoded prompt: {prompt_tokens.shape}")
            else:
                raise ValueError(f"Unknown model type: {type(self.decoder_model)}")
        else:
            prompt_tokens = None
            logger.info("No reference audio provided")

        return prompt_tokens
