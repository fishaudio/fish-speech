from hashlib import sha256
from pathlib import Path
from typing import Literal, Tuple

import torch
from loguru import logger

from tools.api_server import encode_reference
from tools.file import AUDIO_EXTENSIONS, audio_to_bytes, list_files, read_ref_text
from tools.schema import ServeReferenceAudio


class ReferenceLoader:

    def __init__(self) -> None:
        """
        Component of the InferenceEngine class.
        Loads and manages the cache for the reference audio and text.
        """
        self.ref_by_id: dict = {}
        self.ref_by_hash: dict = {}

        # Make Pylance happy (attribut not defined...)
        self.decoder_model: torch.nn.Module

    def load_by_id(self, id: str, use_cache: Literal["never", "on-demand"]) -> Tuple:
        # Load the references audio and text by id
        ref_folder = Path("references") / id
        ref_folder.mkdir(parents=True, exist_ok=True)
        ref_audios = list_files(
            ref_folder, AUDIO_EXTENSIONS, recursive=True, sort=False
        )

        if use_cache == "never" or id not in self.ref_by_id:
            # If the references are not already loaded, encode them
            prompt_tokens = [
                encode_reference(
                    decoder_model=self.decoder_model,
                    reference_audio=audio_to_bytes(str(ref_audio)),
                    enable_reference_audio=True,
                )
                for ref_audio in ref_audios
            ]
            prompt_texts = [
                read_ref_text(str(ref_audio.with_suffix(".lab")))
                for ref_audio in ref_audios
            ]
            self.ref_by_id[id] = (prompt_tokens, prompt_texts)

        else:
            # Reuse already encoded references
            logger.info("Use same references")
            prompt_tokens, prompt_texts = self.ref_by_id[id]

        return prompt_tokens, prompt_texts

    def load_by_hash(
        self,
        references: list[ServeReferenceAudio],
        use_cache: Literal["never", "on-demand"],
    ) -> Tuple:
        # Load the references audio and text by hash
        audio_hashes = [sha256(ref.audio).hexdigest() for ref in references]

        cache_used = False
        prompt_tokens, prompt_texts = [], []
        for i, ref in enumerate(references):
            if use_cache == "never" or audio_hashes[i] not in self.ref_by_hash:
                # If the references are not already loaded, encode them
                prompt_tokens.append(
                    encode_reference(
                        decoder_model=self.decoder_model,
                        reference_audio=ref.audio,
                        enable_reference_audio=True,
                    )
                )
                prompt_texts.append(ref.text)
                self.ref_by_hash[audio_hashes[i]] = (prompt_tokens, prompt_texts)

            else:
                # Reuse already encoded references
                prompt_text, prompt_token = self.ref_by_hash[audio_hashes[i]]
                prompt_texts.append(prompt_text)
                prompt_tokens.append(prompt_token)
                cache_used = True

        if cache_used:
            logger.info("Use same references")

        return prompt_tokens, prompt_texts
