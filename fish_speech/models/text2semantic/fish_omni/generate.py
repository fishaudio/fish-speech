import os
import re
import time
from collections.abc import Iterator, Sequence
from typing import Literal

import torch
from loguru import logger
from tokenizers import Tokenizer
from tqdm import tqdm

from fish_speech.models.text2semantic.errors import InvalidInputError
from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel
from fish_speech.models.text2semantic.fish_omni.tower import (
    RAS_WINDOW_FRAMES,
    TowerAttention,
)
from fish_speech.models.text2semantic.sampling import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)

DRAMA3_TTS_INSTRUCTION = "convert the provided text to speech"
DRAMA3_REFERENCE_INSTRUCTION = (
    "convert the provided text to speech reference to the following:"
)
DRAMA3_MESSAGE_END = "<|im_end|>\n"
DRAMA3_GENERATION_START = "<|im_start|>assistant\n<|voice|>"
# Only the prompt builder writes these; other special tokens in caller text are content.
DRAMA3_STRUCTURAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|voice|>",
    "<|text|>",
    "<|interleave|>",
    "<|semantic:",
)
SPEAKER_TAG_PATTERN = re.compile(r"<\|speaker:([0-9]+)\|>")
# The tokenizer has one token per speaker, <|speaker:0|> to <|speaker:15|>.
MAX_SPEAKER_ID = 15
# The Unicode White_Space property; str.strip() without arguments also strips
# U+001C..U+001F, which would change the empty check and the speaker-tag rule.
UNICODE_WHITE_SPACE = "".join(
    map(
        chr,
        (*range(0x9, 0xE), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B))
        + (0x2028, 0x2029, 0x202F, 0x205F, 0x3000),
    )
)
DRAMA3_MAX_PROMPT_ROWS = 8192
DRAMA3_DEFAULT_MAX_NEW_TOKENS = 2048
DRAMA3_PREFILL_CHUNK_SIZE = 128
DRAMA3_MAX_DEFAULT_DEVICES = 4
# CUDA-graph decode: caches hold the largest default request, with one graph per
# attended cache length; a step replays the smallest bucket covering its position.
DRAMA3_GRAPH_CAPACITY = DRAMA3_MAX_PROMPT_ROWS + DRAMA3_DEFAULT_MAX_NEW_TOKENS
DRAMA3_GRAPH_KV_BUCKETS = (1024, 2048, 4096, DRAMA3_GRAPH_CAPACITY)

FinishReason = Literal["eos", "length"]


class Drama3PromptTooLongError(InvalidInputError):
    """The prompt has more than DRAMA3_MAX_PROMPT_ROWS rows."""


def validate_drama3_text(text: str, location: str) -> None:
    """Non-empty, no structural tokens, and only canonical speaker tags 0..15."""
    if not text.strip(UNICODE_WHITE_SPACE):
        raise InvalidInputError(f"{location}: text must not be empty")
    for token in DRAMA3_STRUCTURAL_TOKENS:
        if token in text:
            raise InvalidInputError(
                f"{location}: {token} is written by the prompt builder"
            )
    for marker in re.finditer(r"<\|speaker", text):
        tag = SPEAKER_TAG_PATTERN.match(text, marker.start())
        digits = tag.group(1) if tag else ""
        if not digits or str(int(digits)) != digits or int(digits) > MAX_SPEAKER_ID:
            raise InvalidInputError(
                f"{location}: speaker tags must be <|speaker:0|> to "
                f"<|speaker:{MAX_SPEAKER_ID}|>; got {text!r}"
            )


def with_speaker_tag(text: str) -> str:
    """Prefix <|speaker:0|> unless a tag already leads the text after white space."""
    if SPEAKER_TAG_PATTERN.match(text.lstrip(UNICODE_WHITE_SPACE)) is None:
        return f"<|speaker:0|> {text}"
    return text


def build_drama3_prompt(
    text: str,
    tokenizer: Tokenizer,
    config: FishOmniConfig,
    prompt_texts: Sequence[str] = (),
    prompt_tokens: Sequence[torch.Tensor] = (),
    history: Sequence[tuple[str, torch.Tensor]] = (),
) -> torch.Tensor:
    """CPU int64 rows [rows, 1 + num_codebooks] ending in the open assistant voice turn.

    Text rows are [token, 0, ..., 0]; audio rows are [c0 + semantic_start, c0..cN-1].
    Every literal is tokenized on its own: tokenizing the joined string changes ids
    at the seams. history holds earlier (text, codes) chunks, written as completed
    user and assistant voice turns before the current one.
    """
    num_codebooks = config.audio_decoder.num_codebooks
    codebook_size = config.audio_decoder.vocab_size
    for token, expected in (
        ("<|semantic:0|>", config.semantic_start_token_id),
        (f"<|semantic:{codebook_size - 1}|>", config.semantic_end_token_id),
        ("<|im_end|>", config.eos_token_id),
    ):
        if tokenizer.token_to_id(token) != expected:
            raise ValueError(f"tokenizer/config mismatch for {token}")
    if len(prompt_texts) != len(prompt_tokens):
        raise InvalidInputError("every reference needs both a transcript and codes")
    validate_drama3_text(text, "text")

    def check_codes(codes: torch.Tensor, location: str) -> None:
        if (
            codes.dtype.is_floating_point
            or codes.dtype == torch.bool
            or codes.ndim != 2
            or codes.shape[0] != num_codebooks
            or codes.shape[1] == 0
        ):
            raise InvalidInputError(
                f"{location}: codes must be integer [{num_codebooks}, frames >= 1]; "
                f"got {codes.dtype} {tuple(codes.shape)}"
            )
        if codes.min() < 0 or codes.max() >= codebook_size:
            raise InvalidInputError(
                f"{location}: codes must be in [0, {codebook_size})"
            )

    segments: list[str | torch.Tensor] = [
        "<|im_start|>system\n",
        DRAMA3_REFERENCE_INSTRUCTION if prompt_tokens else DRAMA3_TTS_INSTRUCTION,
    ]

    for index, (transcript, codes) in enumerate(zip(prompt_texts, prompt_tokens), 1):
        location = f"reference {index}"
        validate_drama3_text(transcript, location)
        check_codes(codes, location)
        # The tag must follow "Text: " directly; leading white space moves behind it.
        tagged = with_speaker_tag(transcript)
        body = tagged.lstrip(UNICODE_WHITE_SPACE)
        tag = SPEAKER_TAG_PATTERN.match(body)
        leading = tagged[: len(tagged) - len(body)]
        transcript = tag.group(0) + leading + body[tag.end() :]
        segments += [f"\n\n[Reference {index}]\nText: {transcript}\nSpeech: ", codes]
    segments.append(DRAMA3_MESSAGE_END)
    for index, (turn_text, codes) in enumerate(history, 1):
        location = f"history {index}"
        validate_drama3_text(turn_text, location)
        check_codes(codes, location)
        segments += [
            "<|im_start|>user\n",
            with_speaker_tag(turn_text),
            DRAMA3_MESSAGE_END,
            DRAMA3_GENERATION_START,
            codes,
            DRAMA3_MESSAGE_END,
        ]
    segments += [
        "<|im_start|>user\n",
        with_speaker_tag(text),
        DRAMA3_MESSAGE_END,
        DRAMA3_GENERATION_START,
    ]

    rows = []
    for segment in segments:
        if isinstance(segment, str):
            ids = torch.tensor(tokenizer.encode(segment).ids, dtype=torch.long)
            text_rows = torch.zeros((len(ids), 1 + num_codebooks), dtype=torch.long)
            text_rows[:, 0] = ids
            rows.append(text_rows)
        else:
            codes = segment.to(device="cpu", dtype=torch.long)
            main = codes[:1] + config.semantic_start_token_id
            rows.append(torch.cat((main, codes)).T)
    rows = torch.cat(rows)
    if len(rows) > DRAMA3_MAX_PROMPT_ROWS:
        raise Drama3PromptTooLongError(
            f"prompt has {len(rows)} rows; at most {DRAMA3_MAX_PROMPT_ROWS}"
        )
    return rows


@torch.inference_mode()
def generate_drama3(
    model: FishOmniModel,
    prompt: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = DEFAULT_TOP_K,
    top_p: float = DEFAULT_TOP_P,
    prefill_chunk_size: int = DRAMA3_PREFILL_CHUNK_SIZE,
) -> tuple[torch.Tensor, FinishReason]:
    """Sample frames after the prompt rows; returns CPU [frames, 1 + N] without EOS.

    The tower cache of prompt row p is filled with frame p + 1 as its label; the
    last prompt row has none yet, so the first decode step writes that slot.
    """
    # The sampler keeps the top-ranked token whatever top_k is, so top_k < 1 would
    # silently turn into greedy decoding.
    if top_k < 1:
        raise InvalidInputError(f"top_k must be at least 1; got {top_k}")
    backbone, tower = model.backbone, model.audio_decoder
    prompt_rows = len(prompt)
    capacity = prompt_rows + max_new_tokens
    if max_new_tokens < 1 or capacity > model.config.text.max_position_embeddings:
        raise ValueError(
            f"max_new_tokens={max_new_tokens} does not fit after {prompt_rows} "
            "prompt rows"
        )
    graphs = Drama3DecodeGraphs.for_model(model, capacity)
    if graphs is None:
        ensure_drama3_caches(model, capacity)
    frames = torch.zeros(
        (capacity, prompt.shape[1]), dtype=torch.long, device=tower.device
    )
    frames[:prompt_rows] = prompt.to(tower.device)
    positions = torch.arange(capacity, device=tower.device)

    started = time.perf_counter()
    for start in range(0, prompt_rows, prefill_chunk_size):
        end = min(start + prefill_chunk_size, prompt_rows)
        chunk = frames[start:end]
        embeddings = tower.embed_backbone(chunk, backbone.embed(chunk[:, 0]))
        hidden = backbone(
            inputs_embeds=embeddings[None], input_pos=positions[start:end]
        )
        hidden = hidden[0].to(tower.device)
        paired_end = min(end, prompt_rows - 1)
        if paired_end > start:
            tower.prefill(
                hidden[: paired_end - start],
                positions[start:paired_end],
                frames[start:paired_end],
                frames[start + 1 : paired_end + 1],
            )
    logger.info(
        f"Drama 3 prefill: {prompt_rows} rows in {time.perf_counter() - started:.2f}s"
    )

    hidden = hidden[-1:]
    produced, finish_reason = 0, "length"
    if graphs is not None:
        graphs.start(
            hidden, prompt_rows - 1, frames[:prompt_rows], temperature, top_k, top_p
        )
    for step in tqdm(range(max_new_tokens)):
        position = prompt_rows - 1 + step
        if graphs is not None:
            sampled, main = graphs.step(position)
        else:
            logits = backbone.compute_logits(hidden).to(tower.device)
            sampled = tower.decode_step(
                logits,
                hidden,
                positions[position : position + 1],
                frames[position : position + 1],
                frames[: position + 1, 0],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            main = sampled[0, 0].item()
        if main == model.config.eos_token_id:
            finish_reason = "eos"
            break
        frames[position + 1] = sampled[0]
        produced += 1
        if graphs is None and step + 1 < max_new_tokens:
            embeddings = tower.embed_backbone(sampled, backbone.embed(sampled[:, 0]))
            hidden = backbone(
                inputs_embeds=embeddings[None],
                input_pos=positions[position + 1 : position + 2],
            )[0].to(tower.device)
    return frames[prompt_rows : prompt_rows + produced].cpu(), finish_reason


def ensure_drama3_caches(model: FishOmniModel, capacity: int) -> None:
    """Grow the backbone and tower caches to capacity rows, reused across requests.

    Stale rows past a request's positions are masked out.
    """
    if model.cache_capacity >= capacity:
        return
    if model.decode_graphs is not None:
        # The graphs captured the old cache buffers.
        model.decode_graphs = None
        torch.cuda.empty_cache()
    model.backbone.setup_caches(max_batch_size=1, max_seq_len=capacity)
    model.audio_decoder.setup_caches(capacity)
    model.cache_capacity = capacity


class Drama3DecodeGraphs:
    """CUDA graphs of one decode step: lm_head -> tower (sample a frame) -> backbone.

    Eager decode is bound by kernel launches (thousands of small kernels per step).
    Captured once for a model on a single CUDA device; FISH_DRAMA3_CUDA_GRAPH=0 disables.
    """

    def __init__(self, model: FishOmniModel) -> None:
        backbone, tower = model.backbone, model.audio_decoder
        self.model = model
        device = tower.device
        width = 1 + model.config.audio_decoder.num_codebooks
        hidden_size = model.config.text.hidden_size
        self.hidden = torch.zeros((1, hidden_size), dtype=backbone.dtype, device=device)
        self.next_hidden = torch.zeros_like(self.hidden)
        self.position = torch.zeros(1, dtype=torch.long, device=device)
        self.frame = torch.zeros((1, width), dtype=torch.long, device=device)
        self.sampled = torch.zeros_like(self.frame)
        self.history = torch.full(
            (RAS_WINDOW_FRAMES,), -1, dtype=torch.long, device=device
        )
        self.temperature = torch.zeros((), dtype=torch.float32, device=device)
        self.top_k = torch.zeros((), dtype=torch.long, device=device)
        self.top_p = torch.zeros((), dtype=torch.float32, device=device)
        self.finite = torch.ones((), dtype=torch.bool, device=device)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.pool = torch.cuda.graph_pool_handle()
        capacity = self.graph_capacity(model)
        self.buckets = tuple(sorted({min(b, capacity) for b in DRAMA3_GRAPH_KV_BUCKETS}))

    @staticmethod
    def graph_capacity(model: FishOmniModel) -> int:
        return min(DRAMA3_GRAPH_CAPACITY, model.config.text.max_position_embeddings)

    @classmethod
    def for_model(cls, model: FishOmniModel, capacity: int) -> "Drama3DecodeGraphs | None":
        """The model's graphs (captured on first use), or None to decode eagerly."""
        if os.environ.get("FISH_DRAMA3_CUDA_GRAPH", "1") == "0":
            return None
        if capacity > cls.graph_capacity(model):
            return None
        devices = {p.device for p in model.backbone.parameters()}
        devices.add(model.audio_decoder.device)
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            return None
        if model.decode_graphs is None:
            # Captured before the first request's prefill, which resets the caches.
            ensure_drama3_caches(model, cls.graph_capacity(model))
            graphs = cls(model)
            started = time.perf_counter()
            with torch.cuda.device(model.audio_decoder.device):
                for kv_len in graphs.buckets:
                    graphs.capture(kv_len)
            logger.info(
                f"Drama 3 decode CUDA graphs: kv buckets {graphs.buckets} captured in "
                f"{time.perf_counter() - started:.1f}s"
            )
            model.decode_graphs = graphs
        return model.decode_graphs

    def tower_attentions(self) -> Iterator[TowerAttention]:
        tower = self.model.audio_decoder
        for layer in (*tower.shared_layers, *tower.residual_layers):
            yield layer.transformer_layer.attention

    def run(self, kv_len: int) -> None:
        backbone, tower = self.model.backbone, self.model.audio_decoder
        logits = backbone.compute_logits(self.hidden)
        self.finite.copy_(torch.isfinite(logits).all())
        sampled = tower.decode_step(
            logits,
            self.hidden,
            self.position,
            self.frame,
            self.history,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            check_finite=False,
        )
        self.sampled.copy_(sampled)
        embeddings = tower.embed_backbone(sampled, backbone.embed(sampled[:, 0]))
        hidden = backbone(
            inputs_embeds=embeddings[None],
            input_pos=self.position + 1,
            static_kv_len=kv_len,
        )
        self.next_hidden.copy_(hidden[0])

    def capture(self, kv_len: int) -> None:
        for attention in self.tower_attentions():
            attention.static_kv_len = kv_len
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.run(kv_len)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                self.run(kv_len)
            self.graphs[kv_len] = graph
        finally:
            for attention in self.tower_attentions():
                attention.static_kv_len = None

    def start(
        self,
        hidden: torch.Tensor,
        position: int,
        prompt_frames: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> None:
        """Load the state after prefill: last hidden row, its position and frame."""
        self.hidden.copy_(hidden)
        self.position.fill_(position)
        self.frame.copy_(prompt_frames[-1:])
        window = prompt_frames[-RAS_WINDOW_FRAMES:, 0]
        self.history.fill_(-1)
        self.history[RAS_WINDOW_FRAMES - len(window) :].copy_(window)
        self.temperature.fill_(temperature)
        self.top_k.fill_(top_k)
        self.top_p.fill_(top_p)

    def step(self, position: int) -> tuple[torch.Tensor, int]:
        """Sample the frame after position and run the backbone on it.

        Returns the frame [1, 1 + N] on the device and its main token.
        """
        kv_len = next(b for b in self.buckets if b >= position + 2)
        self.graphs[kv_len].replay()
        sampled = self.sampled.clone()
        status = torch.stack((sampled[0, 0], self.finite.long()))
        # Inputs of the next step, queued before the host waits on status.
        self.hidden.copy_(self.next_hidden)
        self.frame.copy_(sampled)
        self.history.copy_(torch.cat((self.history[1:], sampled[0, :1])))
        self.position.add_(1)
        main, finite = status.tolist()
        if not finite:
            raise FloatingPointError(
                "non-finite logits; refusing to sample invalid audio"
            )
        return sampled, main


def resolve_drama3_devices(device: str) -> tuple[list[str], str]:
    """Split "cuda:0,cuda:1,..." into backbone devices and the tower (+ codec) device.

    The last device holds the tower; a single device holds everything. Plain "cuda"
    takes the first min(4, visible) GPUs.
    """
    specs = [spec.strip() for spec in device.split(",") if spec.strip()]
    if not specs:
        raise ValueError("no device given")
    if len(specs) == 1 and specs[0] == "cuda":
        count = min(torch.cuda.device_count(), DRAMA3_MAX_DEFAULT_DEVICES)
        if count == 0:
            raise RuntimeError("Drama 3 inference needs CUDA, but no GPU is visible")
        specs = [f"cuda:{index}" for index in range(count)]
    for spec in specs:
        parsed = torch.device(spec)
        if (
            parsed.type != "cuda"
            or parsed.index is None
            or parsed.index >= torch.cuda.device_count()
        ):
            raise ValueError(f"Drama 3 needs available cuda:N devices; got {spec!r}")
    if len(set(specs)) != len(specs):
        raise ValueError(f"duplicate devices in {device!r}")
    if len(specs) == 1:
        return specs, specs[0]
    return specs[:-1], specs[-1]


def resolve_drama3_dtype(precision: torch.dtype, devices: list[str]) -> torch.dtype:
    """bfloat16 needs SM80+; on older GPUs (V100) Drama 3 runs in float16 instead."""
    if precision not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Drama 3 runs in float16 or bfloat16; got {precision}")
    if precision == torch.bfloat16 and any(
        torch.cuda.get_device_capability(device) < (8, 0) for device in devices
    ):
        logger.warning(
            "bfloat16 needs SM80+; running Drama 3 in float16 with an fp32 residual stream"
        )
        return torch.float16
    return precision
