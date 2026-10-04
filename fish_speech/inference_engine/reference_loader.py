import inspect
import io
import os
import re
from pathlib import Path
from typing import Callable, Literal, Tuple

import torch
import torchaudio
from loguru import logger

from fish_speech.inference_engine.reference_token_cache import (
    ReferenceTokenCache,
    codec_fingerprint,
)
from fish_speech.models.dac.modded_dac import DAC
from fish_speech.utils.file import (
    AUDIO_EXTENSIONS,
    audio_to_bytes,
    list_files,
    read_ref_text,
)
from fish_speech.utils.schema import ServeReferenceAudio

_ID_PATTERN = re.compile(r"^[a-zA-Z0-9\-_ ]+$")


class ReferenceLoader:
    def __init__(self) -> None:
        """
        Component of the TTSInferenceEngine class.
        Loads and manages the cache for the reference audio and text.
        """
        self.reference_cache_events = []
        self.reference_token_cache = None
        self._reference_codec_config = None
        self.ref_by_id: dict = {}
        self.ref_by_hash: dict = {}

        # Make Pylance happy (attribut/method not defined...)
        self.decoder_model: DAC
        self.encode_reference: Callable

        # Define the torchaudio backend
        # list_audio_backends() was removed in torchaudio 2.9
        try:
            backends = torchaudio.list_audio_backends()
            if "ffmpeg" in backends:
                self.backend = "ffmpeg"
            else:
                self.backend = "soundfile"
        except AttributeError:
            # torchaudio 2.9+ removed list_audio_backends()
            # Try ffmpeg first, fallback to soundfile
            try:
                __import__("torchaudio.io._load_audio_fileobj")

                self.backend = "ffmpeg"
            except (ImportError, ModuleNotFoundError):
                self.backend = "soundfile"

    @staticmethod
    def _validate_id(id: str) -> None:
        if not _ID_PATTERN.match(id) or len(id) > 255:
            raise ValueError(
                "Reference ID contains invalid characters or is too long. "
                "Only alphanumeric, hyphens, underscores, and spaces are allowed."
            )

    def configure_reference_token_cache(self, checkpoint_path, config_name):
        self._reference_codec_config = (str(checkpoint_path), config_name)
        fingerprint, identity = codec_fingerprint(
            checkpoint_path,
            config_name,
            inspect.getsource(type(self).load_audio),
            self.decoder_model,
            self.backend,
        )
        quantizer = self.decoder_model.quantizer
        limits = [quantizer.semantic_quantizer.codebook_size]
        limits += [quantizer.quantizer.codebook_size] * len(
            quantizer.quantizer.quantizers
        )
        self.reference_token_cache = ReferenceTokenCache(
            os.environ.get("REFERENCE_TOKEN_CACHE_DIR", "references/.token_cache"),
            fingerprint,
            limits,
            identity,
        )
        logger.info("Reference Codec Fingerprint: {}", fingerprint)

    def resolve_reference_tokens(
        self,
        audio_bytes,
        encoder_device="gpu",
        use_memory_cache="off",
        read_disk_cache=True,
        save_disk_cache=False,
        force_reencode=False,
    ):
        if self.reference_token_cache is None:
            # Third-party callers that construct an engine without checkpoint
            # metadata retain temporary/memory encoding; never share disk tokens.
            quantizer = self.decoder_model.quantizer
            limits = [quantizer.semantic_quantizer.codebook_size]
            limits += [quantizer.quantizer.codebook_size] * len(
                quantizer.quantizer.quantizers
            )
            self.reference_token_cache = ReferenceTokenCache(
                "references/.token_cache",
                "unconfigured",
                limits,
            )
        if self._reference_codec_config is None:
            read_disk_cache = save_disk_cache = False
        array, event = self.reference_token_cache.resolve(
            audio_bytes,
            lambda audio, device: self.encode_reference(
                audio, True, encoder_device=device
            ),
            encoder_device=encoder_device,
            use_memory_cache=use_memory_cache == "on",
            read_disk_cache=read_disk_cache,
            save_disk_cache=save_disk_cache,
            force_reencode=force_reencode,
        )
        self.reference_cache_events.append(event)
        return torch.from_numpy(array)

    def load_by_id(self, id: str, use_cache: Literal["on", "off"], **options) -> Tuple:
        self._validate_id(id)
        ref_folder = Path("references") / id
        if not ref_folder.is_dir():
            raise FileNotFoundError("Saved reference voice does not exist")
        ref_audios = list_files(ref_folder, AUDIO_EXTENSIONS, recursive=True, sort=True)
        if not ref_audios:
            raise ValueError("Saved reference voice contains no audio")
        # IDs locate materials only; text is read afresh for every request.
        tokens, texts = [], []
        for path in ref_audios:
            tokens.append(
                self.resolve_reference_tokens(
                    path.read_bytes(), use_memory_cache=use_cache, **options
                )
            )
            texts.append(read_ref_text(str(path.with_suffix(".lab"))))
        return tokens, texts

    def load_by_hash(
        self,
        references: list[ServeReferenceAudio],
        use_cache: Literal["on", "off"],
        **options,
    ) -> Tuple:
        tokens, texts = [], []
        for reference in references:
            tokens.append(
                self.resolve_reference_tokens(
                    reference.audio, use_memory_cache=use_cache, **options
                )
            )
            texts.append(reference.text)
        return tokens, texts

    def load_audio(self, reference_audio: bytes | str, sr: int):
        """
        Load the audio data from a file or bytes.
        """
        if isinstance(reference_audio, bytes):
            reference_audio = io.BytesIO(reference_audio)

        try:
            waveform, original_sr = torchaudio.load(
                reference_audio, backend=self.backend
            )
        except ImportError:
            # New torchaudio versions require optional torchcodec even when
            # soundfile is requested. Keep the existing mono/resample pipeline.
            import soundfile

            data, original_sr = soundfile.read(
                reference_audio, dtype="float32", always_2d=True
            )
            waveform = torch.from_numpy(data.T.copy())

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
        self._validate_id(id)

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
        self._validate_id(id)

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
