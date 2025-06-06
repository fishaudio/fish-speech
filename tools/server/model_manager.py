import torch
from funasr import AutoModel
from loguru import logger

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest
from tools.server.inference import inference_wrapper as inference

ASR_MODEL_NAME = "iic/SenseVoiceSmall"


class ModelManager:
    def __init__(
        self,
        mode: str,
        device: str,
        half: bool,
        compile: bool,
        asr_enabled: bool,
        llama_checkpoint_path: str,
        decoder_checkpoint_path: str,
        decoder_config_name: str,
    ) -> None:

        self.mode = mode
        self.device = device
        self.half = half
        self.compile = compile

        self.precision = torch.half if half else torch.bfloat16

        # Check if MPS or CUDA is available
        if torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("mps is available, running on mps.")
        elif not torch.cuda.is_available():
            self.device = "cpu"
            logger.info("CUDA is not available, running on CPU.")

        # Load the ASR model if enabled
        if asr_enabled:
            self.load_asr_model(self.device)

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
        )

        # Warm up the models
        if self.mode == "tts":
            self.warm_up(self.tts_inference_engine)

    def load_asr_model(self, device, hub="ms") -> None:
        self.asr_model = AutoModel(
            model=ASR_MODEL_NAME,
            device=device,
            disable_pbar=True,
            hub=hub,
        )
        logger.info("ASR model loaded.")

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
