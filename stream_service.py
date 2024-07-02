import time

import librosa
import numpy as np
import torch
import torchaudio
from loguru import logger
from torchaudio import functional as AF
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
)

from fish_speech.conversation import (
    CODEBOOK_EOS_TOKEN_ID,
    Conversation,
    Message,
    TokensPart,
    encode_conversation,
)
from fish_speech.models.text2semantic.llama import DualARTransformer
from tools.api import decode_vq_tokens, encode_reference
from tools.llama.generate_test import convert_string
from tools.llama.generate_test import generate as llama_generate
from tools.llama.generate_test import load_model as load_llama_model
from tools.vqgan.inference import load_model as load_decoder_model


class FishStreamVAD:
    def __init__(self) -> None:
        # Args
        self.sample_rate = 16000
        self.threshold = 0.5
        self.neg_threshold = self.threshold - 0.15
        self.min_speech_duration_ms = 100
        self.min_silence_ms = 500
        self.speech_pad_ms = 30
        self.chunk_size = 512

        # Convert to samples
        self.min_speech_duration_samples = (
            self.min_speech_duration_ms * self.sample_rate // 1000
        )
        self.min_silence_samples = self.min_silence_ms * self.sample_rate // 1000
        self.speech_pad_samples = self.speech_pad_ms * self.sample_rate // 1000

        # Core buffers
        self.reset()

        # Load models
        logger.info("Loading VAD model")
        vad_model, vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=True,
            onnx=True,
        )

        self.vad_model = vad_model
        self.get_speech_timestamps = vad_utils[0]
        logger.info("VAD model loaded")

    def reset(self):
        self.audio_chunks = None
        self.vad_pointer = 0
        self.speech_probs = []

        self.triggered = False
        self.start = self.end = self.temp_end = 0
        self.last_seen_end = 0
        self.speech_segments = []

    def add_chunk(self, chunk, sr=None):
        """
        Add a chunk to the buffer
        """

        if isinstance(chunk, np.ndarray):
            chunk = torch.from_numpy(chunk)

        if sr is not None and sr != self.sample_rate:
            chunk = AF.resample(chunk, sr, self.sample_rate)

        # self.audio_chunks.append(chunk)
        if self.audio_chunks is None:
            self.audio_chunks = chunk
        else:
            self.audio_chunks = torch.cat([self.audio_chunks, chunk])

        # Trigger VAD
        yield from self.detect_speech()

    def detect_speech(self):
        """
        Run the VAD model on the current buffer
        """

        speech_prob_start_idx = len(self.speech_probs)
        while len(self.audio_chunks) - self.vad_pointer >= self.chunk_size:
            chunk = self.audio_chunks[
                self.vad_pointer : self.vad_pointer + self.chunk_size
            ]
            speech_prob = self.vad_model(chunk, self.sample_rate)
            self.speech_probs.append(speech_prob)
            self.vad_pointer += self.chunk_size

        # Process speech probs
        for i in range(speech_prob_start_idx, len(self.speech_probs)):
            speech_prob = self.speech_probs[i]

            if speech_prob >= self.threshold and self.temp_end:
                self.temp_end = 0

            if speech_prob >= self.threshold and self.triggered is False:
                self.triggered = True
                self.start = i * self.chunk_size
                continue

            if speech_prob < self.neg_threshold and self.triggered is True:
                if self.temp_end == 0:
                    self.temp_end = i * self.chunk_size

                if i * self.chunk_size - self.temp_end < self.min_silence_samples:
                    continue

                self.end = self.temp_end
                if self.end - self.start > self.min_speech_duration_samples:
                    yield self.audio_chunks[
                        self.start : self.end + self.speech_pad_samples
                    ]

                self.triggered = False
                self.start = self.end = self.temp_end = 0


class FishASR:
    def __init__(self) -> None:
        self.audio_chunks = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16
        model_id = "openai/whisper-medium.en"

        logger.info("Loading ASR model")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, use_safetensors=True
        ).to(self.device)
        processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=256,
            torch_dtype=torch_dtype,
            device=self.device,
        )
        logger.info("ASR model loaded")

    @torch.inference_mode()
    def run(self, chunk):
        return self.pipe(chunk.numpy())


class FishE2EAgent:
    def __init__(self) -> None:
        self.device = device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        decoder_model = load_decoder_model(
            config_name="firefly_gan_vq",
            checkpoint_path="checkpoints/fish-speech-1.2/firefly-gan-vq-fsq-4x1024-42hz-generator.pth",
            device=device,
        )
        self.decoder_model = decoder_model
        logger.info("Decoder model loaded")

        llama_model, decode_one_token = load_llama_model(
            config_name="dual_ar_2_codebook_1.3b",
            checkpoint_path="checkpoints/step_000206000.ckpt",
            device=device,
            precision=torch.bfloat16,
            max_length=2048,
            compile=True,
        )
        self.llama_model: DualARTransformer = llama_model
        self.decode_one_token = decode_one_token
        logger.info("LLAMA model loaded")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "checkpoints/fish-speech-agent-1"
        )
        self.semantic_id = self.tokenizer.convert_tokens_to_ids("<|semantic|>")
        self.im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(
            "fishaudio/fish-speech-1"
        )

        # Control params
        self.temperature = torch.tensor(0.7, device=device, dtype=torch.float)
        self.top_p = torch.tensor(0.7, device=device, dtype=torch.float)
        self.repetition_penalty = torch.tensor(1.2, device=device, dtype=torch.float)

        # This is used to control the timbre of the generated audio
        self.base_messages = [
            # Message(
            #     role="user",
            #     parts=[np.load("example/q0.npy")],
            # ),
            # Message(
            #     role="assistant",
            #     parts=[
            #         "Transcribed: Hi, can you briefly describe what is machine learning?\nResponse: Sure! Machine learning is the process of automating tasks that humans are capable of doing with a computer. It involves training computers to make decisions based on data.",
            #         np.load("example/a0.npy"),
            #     ],
            # ),
        ]
        self.reference = encode_reference(
            decoder_model=self.decoder_model,
            reference_audio="example/a0.wav",
            enable_reference_audio=True,
        )
        self.messages = self.base_messages.copy()

    def reset(self):
        self.messages = self.base_messages.copy()

    @torch.inference_mode()
    def vq_encode(self, audios, sr=None):
        if isinstance(audios, np.ndarray):
            audios = torch.from_numpy(audios)

        if audios.ndim == 1:
            audios = audios[None, None, :]

        audios = audios.to(self.decoder_model.device)
        if sr is not None and sr != self.decoder_model.sampling_rate:
            audios = AF.resample(audios, sr, self.decoder_model.sampling_rate)

        audio_lengths = torch.tensor(
            [audios.shape[2]], device=self.decoder_model.device, dtype=torch.long
        )

        return self.decoder_model.encode(audios, audio_lengths)[0][0]

    @torch.inference_mode()
    def generate(self, audio_chunk, sr=None, text=None):
        vq_output = self.vq_encode(audio_chunk, sr)
        logger.info(f"VQ output: {vq_output.shape}")

        # Encode conversation
        self.messages.append(
            Message(
                role="user",
                parts=[vq_output],
            )
        )

        parts = []
        if text is not None:
            parts.append(f"Transcribed: {text}\nResponse:")

        self.messages.append(
            Message(
                role="assistant",
                parts=parts,
            )
        )
        conversation = Conversation(self.messages)

        # Encode the conversation
        prompt, _ = encode_conversation(
            conversation, self.tokenizer, self.llama_model.config.num_codebooks
        )
        prompt = prompt[:, :-1].to(dtype=torch.int, device=self.device)
        prompt_length = prompt.shape[1]

        # Generate
        y = llama_generate(
            model=self.llama_model,
            prompt=prompt,
            max_new_tokens=0,
            eos_token_id=self.tokenizer.eos_token_id,
            im_end_id=self.im_end_id,
            decode_one_token=self.decode_one_token,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
        )

        tokens = self.tokenizer.decode(
            y[0, prompt_length:].tolist(), skip_special_tokens=False
        )
        logger.info(f"Generated: {convert_string(tokens)}")

        # Put the generated tokens
        # since there is <im_end> and <eos> tokens, we remove last 2 tokens
        code_mask = y[0, prompt_length:-2] == self.semantic_id
        codes = y[1:, prompt_length:-2][:, code_mask].clone()

        codes = codes - 2
        assert (codes >= 0).all(), f"Negative code found"

        decoded = y[:, prompt_length:-1].clone()
        if decoded[0, -1] != self.im_end_id:  # <im_end>
            val = [[self.im_end_id]] + [[CODEBOOK_EOS_TOKEN_ID]] * (decoded.size(0) - 1)
            decoded = torch.cat(
                (decoded, torch.tensor(val, device=self.device, dtype=torch.int)), dim=1
            )

        decoded = decoded.cpu()
        self.messages[-1].parts.append(
            TokensPart(
                tokens=decoded[:1],
                codes=decoded[1:],
            )
        )

        # Less than 5 * 20 = 100ms
        if codes.shape[1] <= 5:
            return

        # Generate audio
        main_tokens = decoded[0]
        text_tokens = main_tokens[main_tokens != self.semantic_id]
        text = self.tokenizer.decode(text_tokens.tolist(), skip_special_tokens=True)
        text_tokens = self.decoder_tokenizer.encode(text, return_tensors="pt").to(
            self.device
        )

        audio = decode_vq_tokens(
            decoder_model=self.decoder_model,
            codes=codes,
            text_tokens=text_tokens,
            reference_embedding=self.reference,
        )

        if sr is not None and sr != self.decoder_model.sampling_rate:
            audio = AF.resample(audio, self.decoder_model.sampling_rate, sr)

        return audio.float()


class FishAgentPipeline:
    def __init__(self) -> None:
        self.vad = FishStreamVAD()
        # Currently use ASR model as intermediate
        self.asr = FishASR()
        self.agent = FishE2EAgent()

        self.vad_segments = []
        self.text_segments = []

    def add_chunk(self, chunk, sr=None):
        use_np = isinstance(chunk, np.ndarray)
        if use_np:
            chunk = torch.from_numpy(chunk)

        if sr is not None and sr != 16000:
            chunk = AF.resample(chunk, sr, 16000)

        for vad_audio in self.vad.add_chunk(chunk, 16000):
            self.vad_segments.append(vad_audio)
            asr_text = self.asr.run(vad_audio)
            self.text_segments.append(asr_text)
            logger.info(f"ASR: {asr_text}")

            # Actually should detect if intent is finished here
            result = self.agent.generate(vad_audio, 16000, text=asr_text)
            if result is None:
                continue

            if sr is not None and sr != 16000:
                result = AF.resample(result, 16000, sr)

            if use_np:
                result = result.cpu().numpy()

            yield result

    def reset(self):
        self.vad.reset()
        self.agent.reset()
        self.vad_segments = []
        self.text_segments = []

    def warmup(self):
        logger.info("Warming up the pipeline")
        audio, sr = librosa.load("example/q0.mp3", sr=16000)
        for i in range(0, len(audio), 882):
            for audio in self.add_chunk(audio[i : i + 882], sr):
                pass
        logger.info("Pipeline warmed up")
        self.reset()


if __name__ == "__main__":
    import soundfile as sf

    service = FishAgentPipeline()
    service.warmup()
    logger.info("Stream service started")

    audio, sr = librosa.load("example/q1.mp3", sr=16000)
    seg = []
    for i in range(0, len(audio), 882):
        for audio in service.add_chunk(audio[i : i + 882], sr):
            seg.append(audio)

    audio = np.concatenate(seg)
    sf.write("output.wav", audio, 16000)
