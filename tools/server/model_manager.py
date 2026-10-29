import torch
from loguru import logger

from fish_speech.inference_engine import ModelFamily, TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.errors import InvalidInputError
from fish_speech.models.text2semantic.fish_omni.generate import (
    resolve_drama3_devices,
    resolve_drama3_dtype,
    resolve_epsilon_cutoff,
)
from fish_speech.models.text2semantic.fish_omni.model import is_fish_omni_checkpoint
from fish_speech.models.text2semantic.fish_omni.tower import ReferenceSamplingFallback
from fish_speech.models.text2semantic.inference import (
    launch_thread_safe_queue,
    load_codec_model,
    resolve_codec_path,
)
from fish_speech.utils.schema import ServeTTSRequest
from tools.server.inference import inference_wrapper as inference


def resolve_model_device(device: str) -> str:
    """Resolve the device the server runs the models on.

    ``cuda`` is the historical default and the only request that is redirected: to MPS when it is
    available, otherwise to CPU. An explicit accelerator choice (``npu``, ``xpu``, ``cpu``, ...) is
    kept as it was requested, so the server does not silently override it.
    """
    requested = str(device).split(":", 1)[0].lower()
    if requested == "cuda":
        if torch.backends.mps.is_available():
            logger.info("mps is available, running on mps.")
            return "mps"
        if not torch.cuda.is_available():
            logger.info("CUDA is not available, running on CPU.")
            return "cpu"
    elif requested == "mps" and not torch.backends.mps.is_available():
        logger.info("mps is not available, running on CPU.")
        return "cpu"
    return device


class ModelManager:
    """Loads one TTS model (S2-Pro or multi-GPU-capable Drama 3) plus the DAC codec."""

    def __init__(
        self,
        mode: str,
        device: str,
        half: bool,
        compile: bool,
        llama_checkpoint_path: str,
        decoder_checkpoint_path: str | None,
        decoder_config_name: str,
        reference_sampling: ReferenceSamplingFallback | None = None,
    ) -> None:

        self.mode = mode
        self.device = device
        self.half = half
        self.compile = compile
        self.reference_sampling = reference_sampling
        self.precision = torch.half if half else torch.bfloat16
        self.model_family: ModelFamily = (
            "drama3" if is_fish_omni_checkpoint(llama_checkpoint_path) else "s2"
        )
        if reference_sampling is not None and self.model_family != "drama3":
            raise ValueError("--reference-sampling needs a Drama 3 checkpoint")

        decoder_checkpoint_path = resolve_codec_path(
            llama_checkpoint_path, decoder_checkpoint_path
        )

        if self.model_family == "drama3":
            # The codec lives with the tower (last device), in the model's dtype.
            devices, tower_device = resolve_drama3_devices(device)
            self.precision = resolve_drama3_dtype(
                self.precision, [*devices, tower_device]
            )
            self.device = tower_device
            logger.info(
                f"Drama 3 checkpoint: backbone on {devices}, tower + codec on "
                f"{tower_device}, {self.precision}"
            )
        else:
            self.device = resolve_model_device(device)

        # Load the TTS models (Drama 3 spreads over the full device list)
        self.load_llama_model(
            llama_checkpoint_path,
            device if self.model_family == "drama3" else self.device,
            self.precision,
            self.compile,
            self.mode,
        )
        self.load_decoder_model(
            decoder_config_name, decoder_checkpoint_path, self.device
        )
        self.tts_inference_engine = TTSInferenceEngine(
            llama_queue=self.llama_queue,
            decoder_model=self.decoder_model,
            precision=self.precision,
            compile=self.compile,
            model_family=self.model_family,
        )

        # Warm up the models
        if self.mode == "tts":
            self.warm_up(self.tts_inference_engine)

    def validate_sampling(self, req: ServeTTSRequest) -> None:
        """Reject incompatible sampling before HTTP response headers are sent."""
        if self.model_family == "drama3":
            resolve_epsilon_cutoff(self.reference_sampling, req.epsilon_cutoff)
        elif req.epsilon_cutoff:
            raise InvalidInputError(
                "epsilon_cutoff applies to Drama 3 reference sampling only"
            )
        if req.top_k == 0 and self.reference_sampling is None:
            raise InvalidInputError(
                "top_k=0 requires a Drama 3 checkpoint with --reference-sampling"
            )

    def load_llama_model(
        self, checkpoint_path, device, precision, compile, mode
    ) -> None:

        if mode == "tts":
            self.llama_queue = launch_thread_safe_queue(
                checkpoint_path=checkpoint_path,
                device=device,
                precision=precision,
                compile=compile,
                reference_sampling=self.reference_sampling,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        logger.info("LLAMA model loaded.")

    def load_decoder_model(self, config_name, checkpoint_path, device) -> None:
        if self.model_family == "drama3":
            # Folds weight norm before the cast, which float16 (pre-SM80) needs.
            self.decoder_model = load_codec_model(
                checkpoint_path, device, self.precision
            )
        else:
            self.decoder_model = load_decoder_model(
                config_name=config_name,
                checkpoint_path=checkpoint_path,
                device=device,
            )
        logger.info("Decoder model loaded.")

    def warm_up(self, tts_inference_engine) -> None:
        # Unset sampling fields take the shared defaults for either model.
        request = ServeTTSRequest(text="Hello world.", format="wav")
        list(inference(request, tts_inference_engine))
        logger.info("Models warmed up.")
