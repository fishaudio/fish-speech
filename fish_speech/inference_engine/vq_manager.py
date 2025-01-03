from typing import Callable

import torch
from loguru import logger

from fish_speech.models.vqgan.modules.firefly import FireflyArchitecture


class VQManager:

    def __init__(self):
        # Make Pylance happy (attribut/method not defined...)
        self.decoder_model: FireflyArchitecture
        self.load_audio: Callable

    def decode_vq_tokens(self, codes):
        feature_lengths = torch.tensor(
            [codes.shape[1]], device=self.decoder_model.device
        )
        logger.info(f"VQ features: {codes.shape}")

        if isinstance(self.decoder_model, FireflyArchitecture):
            return self.decoder_model.decode(
                indices=codes[None],
                feature_lengths=feature_lengths,
            )[0].squeeze()

        raise ValueError(f"Unknown model type: {type(self.decoder_model)}")

    def encode_reference(self, reference_audio, enable_reference_audio):
        if enable_reference_audio and reference_audio is not None:
            # Load audios, and prepare basic info here
            reference_audio_content = self.load_audio(
                reference_audio, self.decoder_model.spec_transform.sample_rate
            )

            audios = torch.from_numpy(reference_audio_content).to(
                self.decoder_model.device
            )[None, None, :]
            audio_lengths = torch.tensor(
                [audios.shape[2]], device=self.decoder_model.device, dtype=torch.long
            )
            logger.info(
                f"Loaded audio with {audios.shape[2] / self.decoder_model.spec_transform.sample_rate:.2f} seconds"
            )

            # VQ Encoder
            if isinstance(self.decoder_model, FireflyArchitecture):
                prompt_tokens = self.decoder_model.encode(audios, audio_lengths)[0][0]
                logger.info(f"Encoded prompt: {prompt_tokens.shape}")
            else:
                raise ValueError(f"Unknown model type: {type(self.decoder_model)}")
        else:
            prompt_tokens = None
            logger.info("No reference audio provided")

        return prompt_tokens
