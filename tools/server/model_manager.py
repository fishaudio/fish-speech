import json
from hashlib import sha256
from pathlib import Path

import torch
from loguru import logger

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeReferenceCompatibility, ServeTTSRequest
from tools.server.inference import inference_wrapper as inference


class ModelManager:
    def __init__(
        self,
        mode: str,
        device: str,
        half: bool,
        compile: bool,
        llama_checkpoint_path: str,
        decoder_checkpoint_path: str,
        decoder_config_name: str,
    ) -> None:

        self.mode = mode
        self.device = device
        self.half = half
        self.compile = compile
        self.llama_checkpoint_path = Path(llama_checkpoint_path)
        self.decoder_checkpoint_path = Path(decoder_checkpoint_path)
        self.decoder_config_name = decoder_config_name

        self.precision = torch.half if half else torch.bfloat16

        # Check if MPS or CUDA is available
        if torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("mps is available, running on mps.")
        elif not torch.cuda.is_available():
            self.device = "cpu"
            logger.info("CUDA is not available, running on CPU.")

        # Load the TTS models
        self.load_llama_model(
            llama_checkpoint_path, self.device, self.precision, self.compile, self.mode
        )
        self.load_decoder_model(
            decoder_config_name, decoder_checkpoint_path, self.device
        )
        self.tts_inference_engine = TTSInferenceEngine(
            llama_queue=self.llama_queue,
            decoder_model=self.decoder_model,
            precision=self.precision,
            compile=self.compile,
            reference_compatibility=self.get_reference_compatibility_snapshot(),
        )

        # Warm up the models
        if self.mode == "tts":
            self.warm_up(self.tts_inference_engine)

    def load_llama_model(
        self, checkpoint_path, device, precision, compile, mode
    ) -> None:

        if mode == "tts":
            self.llama_queue = launch_thread_safe_queue(
                checkpoint_path=checkpoint_path,
                device=device,
                precision=precision,
                compile=compile,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        logger.info("LLAMA model loaded.")

    def load_decoder_model(self, config_name, checkpoint_path, device) -> None:
        self.decoder_model = load_decoder_model(
            config_name=config_name,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        logger.info("Decoder model loaded.")

    def warm_up(self, tts_inference_engine) -> None:
        request = ServeTTSRequest(
            text="Hello world.",
            references=[],
            reference_id=None,
            max_new_tokens=1024,
            chunk_length=200,
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.7,
            format="wav",
        )
        list(inference(request, tts_inference_engine))
        logger.info("Models warmed up.")

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

    @staticmethod
    def _sha256_manifest(path: Path) -> str:
        digest = sha256()
        for child in sorted(path.rglob("*")):
            if not child.is_file():
                continue
            rel = str(child.relative_to(path)).replace("\\", "/")
            stat = child.stat()
            digest.update(rel.encode("utf-8"))
            digest.update(stat.st_size.to_bytes(8, "little"))
        return f"sha256:{digest.hexdigest()}"

    def _tokenizer_sha256(self) -> str:
        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
        ]
        existing = [
            self.llama_checkpoint_path / name
            for name in tokenizer_files
            if (self.llama_checkpoint_path / name).is_file()
        ]
        if not existing:
            return self._sha256_manifest(self.llama_checkpoint_path)

        digest = sha256()
        for tokenizer_file in existing:
            digest.update(tokenizer_file.name.encode("utf-8"))
            with tokenizer_file.open("rb") as file_handle:
                for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

    def _semantic_begin_id(self) -> int:
        config_path = self.llama_checkpoint_path / "config.json"
        if not config_path.is_file():
            return 0

        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return 0

        return int(data.get("semantic_start_token_id", 0))

    def _reference_num_codebooks(self) -> int:
        quantizer = getattr(self.decoder_model, "quantizer", None)
        if quantizer is None:
            raise AttributeError("Decoder model is missing quantizer")

        direct = getattr(quantizer, "n_codebooks", None)
        if direct is not None:
            return int(direct) + 1

        residual_quantizer = getattr(quantizer, "quantizer", None)
        residual_n_codebooks = getattr(residual_quantizer, "n_codebooks", None)
        if residual_n_codebooks is not None:
            return int(residual_n_codebooks) + 1

        codebooks = getattr(residual_quantizer, "codebooks", None)
        if codebooks is not None:
            return len(codebooks) + 1

        raise AttributeError(
            "Unable to determine decoder quantizer codebook count for reference compatibility"
        )

    def get_reference_compatibility_snapshot(self) -> ServeReferenceCompatibility:
        sample_rate_hz = getattr(self.decoder_model, "sample_rate", None)
        if sample_rate_hz is None and hasattr(self.decoder_model, "spec_transform"):
            sample_rate_hz = self.decoder_model.spec_transform.sample_rate

        return ServeReferenceCompatibility(
            artifact_schema_version=1,
            codec_checkpoint_sha256=self._sha256_file(self.decoder_checkpoint_path),
            decoder_config_name=self.decoder_config_name,
            text2semantic_checkpoint_sha256=self._sha256_manifest(
                self.llama_checkpoint_path
            ),
            tokenizer_sha256=self._tokenizer_sha256(),
            num_codebooks=self._reference_num_codebooks(),
            semantic_begin_id=self._semantic_begin_id(),
            sample_rate_hz=int(sample_rate_hz),
        )
