import io
from hashlib import sha256
from pathlib import Path
from typing import Callable, Literal, Tuple

import torch
import torchaudio
from loguru import logger

from fish_speech.models.dac.modded_dac import DAC
from fish_speech.utils.file import (
    AUDIO_EXTENSIONS,
    audio_to_bytes,
    list_files,
    read_ref_text,
)
from fish_speech.utils.schema import ServeReferenceAudio


class ReferenceLoader:
    def __init__(self) -> None:
        """
        Component of the TTSInferenceEngine class.
        Loads and manages the cache for the reference audio and text.
        """
        self.ref_by_id: dict = {}
        self.ref_by_hash: dict = {}

        # Make Pylance happy (attribut/method not defined...)
        self.decoder_model: DAC
        self.encode_reference: Callable

        # Define the torchaudio backend
        backends = torchaudio.list_audio_backends()
        if "ffmpeg" in backends:
            self.backend = "ffmpeg"
        else:
            self.backend = "soundfile"

    def load_by_id(
        self,
        id: str,
        use_cache: Literal["on", "off"],
    ) -> Tuple:
        # Load the references audio and text by id
        ref_folder = Path("references") / id
        ref_folder.mkdir(parents=True, exist_ok=True)
        ref_audios = list_files(
            ref_folder, AUDIO_EXTENSIONS, recursive=True, sort=False
        )

        if use_cache == "off" or id not in self.ref_by_id:
            # If the references are not already loaded, encode them
            prompt_tokens = [
                self.encode_reference(
                    # decoder_model=self.decoder_model,
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
        use_cache: Literal["on", "off"],
    ) -> Tuple:
        # Load the references audio and text by hash
        audio_hashes = [sha256(ref.audio).hexdigest() for ref in references]

        cache_used = False
        prompt_tokens, prompt_texts = [], []
        for i, ref in enumerate(references):
            if use_cache == "off" or audio_hashes[i] not in self.ref_by_hash:
                # If the references are not already loaded, encode them
                prompt_tokens.append(
                    self.encode_reference(
                        reference_audio=ref.audio,
                        enable_reference_audio=True,
                    )
                )
                prompt_texts.append(ref.text)
                self.ref_by_hash[audio_hashes[i]] = (prompt_tokens[-1], ref.text)

            else:
                # Reuse already encoded references
                cached_token, cached_text = self.ref_by_hash[audio_hashes[i]]
                prompt_tokens.append(cached_token)
                prompt_texts.append(cached_text)
                cache_used = True

        if cache_used:
            logger.info("Use same references")

        return prompt_tokens, prompt_texts

    def load_audio(self, reference_audio: bytes | str, sr: int):
        """
        Load the audio data from a file or bytes.
        """
        if len(reference_audio) > 255 or not Path(reference_audio).exists():
            audio_data = reference_audio
            reference_audio = io.BytesIO(audio_data)

        waveform, original_sr = torchaudio.load(reference_audio, backend=self.backend)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if original_sr != sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sr, new_freq=sr
            )
            waveform = resampler(waveform)

        audio = waveform.squeeze().numpy()
        return audio

    def list_reference_ids(self) -> list[str]:
        """
        List all valid reference IDs (subdirectory names containing valid audio and .lab files).

        Returns:
            list[str]: List of valid reference IDs
        """
        ref_base_path = Path("references")
        if not ref_base_path.exists():
            return []

        valid_ids = []
        for ref_dir in ref_base_path.iterdir():
            if not ref_dir.is_dir():
                continue

            # Check if directory contains at least one audio file and corresponding .lab file
            audio_files = list_files(
                ref_dir, AUDIO_EXTENSIONS, recursive=False, sort=False
            )
            if not audio_files:
                continue

            # Check if corresponding .lab file exists for at least one audio file
            has_valid_pair = False
            for audio_file in audio_files:
                lab_file = audio_file.with_suffix(".lab")
                if lab_file.exists():
                    has_valid_pair = True
                    break

            if has_valid_pair:
                valid_ids.append(ref_dir.name)

        return sorted(valid_ids)

    def add_reference(self, id: str, wav_file_path: str, reference_text: str) -> None:
        """
        Add a new reference voice by creating a new directory and copying files.

        Args:
            id: Reference ID (directory name)
            wav_file_path: Path to the audio file to copy
            reference_text: Text content for the .lab file

        Raises:
            FileExistsError: If the reference ID already exists
            FileNotFoundError: If the audio file doesn't exist
            OSError: If file operations fail
        """
        # Validate ID format
        import re

        if not re.match(r"^[a-zA-Z0-9\-_ ]+$", id):
            raise ValueError(
                "Reference ID contains invalid characters. Only alphanumeric, hyphens, underscores, and spaces are allowed."
            )

        if len(id) > 255:
            raise ValueError(
                "Reference ID is too long. Maximum length is 255 characters."
            )

        # Check if reference already exists
        ref_dir = Path("references") / id
        if ref_dir.exists():
            raise FileExistsError(f"Reference ID '{id}' already exists")

        # Check if audio file exists
        audio_path = Path(wav_file_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_file_path}")

        # Validate audio file extension
        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format: {audio_path.suffix}. Supported formats: {', '.join(AUDIO_EXTENSIONS)}"
            )

        try:
            # Create reference directory
            ref_dir.mkdir(parents=True, exist_ok=False)

            # Determine the target audio filename with original extension
            target_audio_path = ref_dir / f"sample{audio_path.suffix}"

            # Copy audio file
            import shutil

            shutil.copy2(audio_path, target_audio_path)

            # Create .lab file
            lab_path = ref_dir / "sample.lab"
            with open(lab_path, "w", encoding="utf-8") as f:
                f.write(reference_text)

            # Clear cache for this ID if it exists
            if id in self.ref_by_id:
                del self.ref_by_id[id]

            logger.info(f"Successfully added reference voice with ID: {id}")

        except Exception as e:
            # Clean up on failure
            if ref_dir.exists():
                import shutil

                shutil.rmtree(ref_dir)
            raise e

    def delete_reference(self, id: str) -> None:
        """
        Delete a reference voice by removing its directory and files.

        Args:
            id: Reference ID (directory name) to delete

        Raises:
            FileNotFoundError: If the reference ID doesn't exist
            OSError: If file operations fail
        """
        # Check if reference exists
        ref_dir = Path("references") / id
        if not ref_dir.exists():
            raise FileNotFoundError(f"Reference ID '{id}' does not exist")

        try:
            # Remove the entire reference directory
            import shutil

            shutil.rmtree(ref_dir)

            # Clear cache for this ID if it exists
            if id in self.ref_by_id:
                del self.ref_by_id[id]

            logger.info(f"Successfully deleted reference voice with ID: {id}")

        except Exception as e:
            logger.error(f"Failed to delete reference '{id}': {e}")
            raise OSError(f"Failed to delete reference '{id}': {e}")
