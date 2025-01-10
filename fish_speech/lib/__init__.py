import torch
import warnings
import numpy as np
from queue import Queue
from typing import List, Optional, Literal, Union, Generator

from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.models.vqgan.inference import load_model as load_vqgan_model
from fish_speech.models.vqgan.modules.firefly import FireflyArchitecture
from fish_speech.utils.schema import ServeTTSRequest, Reference
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.utils.file import audio_to_bytes

Device = Literal["cuda", "mps", "cpu"]

warnings.simplefilter(action='ignore', category=FutureWarning)



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

        llama = self.load_llama(llama_path, device, precision, compile)
        vqgan = self.load_vqgan(vqgan_config, vqgan_path, device)

        self.inference_engine = TTSInferenceEngine(
            llama_queue=llama,
            decoder_model=vqgan,
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
    

    @property
    def sample_rate(self) -> int:
        """ Get the sample rate of the audio. """
        return self.inference_engine.decoder_model.spec_transform.sample_rate


    def make_reference(self, audio_path: str, text: str) -> Reference:
        """ Create a reference object from audio and text. """
        audio_bytes = audio_to_bytes(audio_path)
        if audio_bytes is None:
            raise ValueError("Failed to load audio file.")
        
        tokens = self.inference_engine.encode_reference(audio_bytes, True)
        return Reference(tokens=tokens, text=text)


    def generate_streaming(
            self,
            text: str,
            references: Union[List[Reference], Reference] = [],
            seed: Optional[int] = None,
            streaming: bool = False,
            max_new_tokens: int = 0,
            chunk_length: int = 200,
            top_p: Optional[float] = None,
            repetition_penalty: Optional[float] = None,
            temperature: Optional[float] = None,
        ) -> Generator:
        """
        Generate audio from text.

        Args:
            text (str): Text to generate audio from.
            references (Union[List[Reference], Reference], optional): List of reference audios. Defaults to [].
            seed (Optional[int], optional): Random seed. Defaults to None.
            streaming (bool, optional): Stream the audio. Defaults to False.
            max_new_tokens (int, optional): Maximum number of tokens. Defaults to 0 (no limit).
            chunk_length (int, optional): Chunk length for streaming. Defaults to 200.
            top_p (Optional[float], optional): Top-p sampling. Defaults to None.
            repetition_penalty (Optional[float], optional): Repetition penalty. Defaults to None.
            temperature (Optional[float], optional): Sampling temperature. Defaults to None.
        """
        references = [references] if isinstance(references, Reference) else references

        request = ServeTTSRequest(
            text=text,
            preprocessed_references=references,
            seed=seed,
            streaming=streaming,
            max_new_tokens=max_new_tokens,
            chunk_length=chunk_length,
            top_p=top_p or 0.7,
            repetition_penalty=repetition_penalty or 1.2,
            temperature=temperature or 0.7,
        )

        count = 0
        for result in self.inference_engine.inference(request):
            match result.code:
                case "header":
                    pass    # In this case, we only want to yield the audio (amplitude)
                            # User can save with a library like soundfile if needed

                case "error":
                    if isinstance(result.error, Exception):
                        raise result.error
                    else:
                        raise RuntimeError("Unknown error")

                case "segment":
                    count += 1
                    if isinstance(result.audio, tuple) and streaming:
                        yield result.audio[1]

                case "final":
                    count += 1
                    if isinstance(result.audio, tuple) and not streaming:
                        yield result.audio[1]

        if count == 0:
            raise RuntimeError("No audio generated, please check the input text.")

    
    def generate(
            self,
            text: str,
            references: Union[List[Reference], Reference] = [],
            seed: Optional[int] = None,
            streaming: bool = False,
            max_new_tokens: int = 0,
            chunk_length: int = 200,
            top_p: Optional[float] = None,
            repetition_penalty: Optional[float] = None,
            temperature: Optional[float] = None,
        ) -> Union[Generator, np.ndarray]:
        """
        Wrapper for the generate_streaming method.
        Returns either a generator or directly the final audio.

        Args:
            text (str): Text to generate audio from.
            references (Union[List[Reference], Reference], optional): List of reference audios. Defaults to [].
            seed (Optional[int], optional): Random seed. Defaults to None.
            streaming (bool, optional): Stream the audio. Defaults to False.
            max_new_tokens (int, optional): Maximum number of tokens. Defaults to 0 (no limit).
            chunk_length (int, optional): Chunk length for streaming. Defaults to 200.
            top_p (Optional[float], optional): Top-p sampling. Defaults to None.
            repetition_penalty (Optional[float], optional): Repetition penalty. Defaults to None.
            temperature (Optional[float], optional): Sampling temperature. Defaults to None.
        """

        generator = self.generate_streaming(
            text=text,
            references=references,
            seed=seed,
            streaming=streaming,
            max_new_tokens=max_new_tokens,
            chunk_length=chunk_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
        )

        if streaming:
            return generator
        else:
            audio = np.concatenate(list(generator))
            return audio