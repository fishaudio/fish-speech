from typing import List, Tuple, Dict, Any, Optional, Literal, Union
import warnings
import torch
from queue import Queue

from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.models.vqgan.inference import load_model as load_vqgan_model
from fish_speech.models.vqgan.modules.firefly import FireflyArchitecture
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.utils.schema import ServeTTSRequest

Device = Literal["cuda", "mps", "cpu"]


class Pipeline:

    def __init__(
            self,
            llama_path: str,
            vqgan_path: str,
            vqgan_config: str = "firefly_gan_vq",
            device: Device = "cpu",
            half: bool = False,
            compile: bool = False,
            ) -> None:
        """
        Initialize the TTS pipeline.

        Args:
            llama_path (str): Path to the LLAMA model.
            vqgan_path (str): Path to the VQ-GAN model.
            vqgan_config (str, optional): VQ-GAN model configuration name. Defaults to base configuration.
            device (Device, optional): Device to run the pipeline on. Defaults to "cpu".
            half (bool, optional): Use half precision. Defaults to False.
            compile (bool, optional): Compile the models. Defaults to False.
        """

        # Validate input
        assert isinstance(llama_path, str), "llama_path must be a string."
        assert isinstance(vqgan_path, str), "vqgan_path must be a string."
        assert isinstance(vqgan_config, str), "vqgan_config must be a string."
        assert isinstance(half, bool), "half must be a boolean."
        assert isinstance(compile, bool), "compile must be a boolean."

        device = self.check_device(device)
        precision = torch.half if half else torch.bfloat16

        self.llama = self.load_llama(llama_path, device, precision, compile)
        self.vqgan = self.load_vqgan(vqgan_config, vqgan_path, device)

        self.inference_engine = TTSInferenceEngine(
            llama_queue=self.llama,
            decoder_model=self.vqgan,
            precision=precision,
            compile=compile,
        )

        self.warmup(self.inference_engine)


    def check_device(self, device: str) -> Device:
        """ Check if the device is available. """
        device = device.lower()

        # If CUDA or MPS chosen, check if available
        match device:
            case "cuda":
                if not torch.cuda.is_available():
                    warnings.warn("CUDA is not available, running on CPU.")
                    device = "cpu"
            case "mps":
                if not torch.backends.mps.is_available():
                    warnings.warn("MPS is not available, running on CPU.")
                    device = "cpu"
            case "cpu":
                pass
            case _:
                raise ValueError("Invalid device, choose from 'cuda', 'mps', 'cpu'.")
        
        return device
    

    def load_llama(self, llama_path: str, device: str, precision: torch.dtype, compile: bool) -> Queue:
        """ Load the LLAMA model. """
        try:
            return launch_thread_safe_queue(
                checkpoint_path=llama_path,
                device=device,
                precision=precision,
                compile=compile,
            )
        except Exception as e:
            raise ValueError(f"Failed to load LLAMA model: {e}")


    def load_vqgan(self, vqgan_config: str, vqgan_path: str, device: str) -> FireflyArchitecture:
        """ Load the VQ-GAN model. """
        try:
            return load_vqgan_model(
                config_name=vqgan_config,
                checkpoint_path=vqgan_path,
                device=device,
            )
        except Exception as e:
            raise ValueError(f"Failed to load VQ-GAN model: {e}")


    def warmup(self, inference_engine: TTSInferenceEngine) -> None:
        """ Warm up the inference engine. """
        try:
            list(
                inference_engine.inference(
                    ServeTTSRequest(
                        text="Hello world.",
                        references=[],
                        reference_id=None,
                        max_new_tokens=1024,
                        chunk_length=200,
                        top_p=0.7,
                        repetition_penalty=1.5,
                        temperature=0.7,
                        format="wav",
                    )
                )
            )
        except Exception as e:
            raise ValueError(f"Failed to warm up the inference engine: {e}")