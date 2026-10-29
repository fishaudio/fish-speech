import contextlib
import io
import re

import librosa
import torch
import torchaudio
from cachetools import LRUCache, cached

CACHE_MAXSIZE = 10000
MICRO_BATCH_SIZE = 8
ASR_SAMPLE_RATE = 16000
HUGE_GAP_THRESHOLD = 4000


def _autocast_context(model):
    device = getattr(model, "device", None)
    device_type = getattr(
        device, "type", str(device).split(":")[0] if device else "cpu"
    )
    if device_type == "cuda" and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.half)
    return contextlib.nullcontext()


@torch.no_grad()
def batch_encode(model, audios_list: list[bytes]):
    with _autocast_context(model):
        # Get sample rate from model
        if hasattr(model, "spec_transform"):
            sample_rate = model.spec_transform.sample_rate
        else:
            sample_rate = model.sample_rate

        audios: list[torch.Tensor] = [
            (
                torch.from_numpy(librosa.load(io.BytesIO(audio), sr=sample_rate)[0])[
                    None
                ]
                if isinstance(audio, bytes)
                else audio
            )
            for audio in audios_list
        ]

        lengths = torch.tensor(
            [audio.shape[-1] for audio in audios], device=model.device
        )
        max_length = lengths.max().item()

        print(f"Encode max length: {max_length / sample_rate:.2f}s")

        padded = torch.stack(
            [
                torch.nn.functional.pad(audio, (0, int(max_length - audio.shape[-1])))
                for audio in audios
            ]
        ).to(model.device)

        features, feature_lengths = model.encode(padded, audio_lengths=lengths)
        features, feature_lengths = features.cpu(), feature_lengths.cpu()

        return [
            feature[..., :length] for feature, length in zip(features, feature_lengths)
        ]


@cached(
    cache=LRUCache(maxsize=CACHE_MAXSIZE),
    key=lambda model, audios: (model.device, tuple(audios)),
)
def cached_vqgan_batch_encode(model, audios: list[bytes]):
    return batch_encode(model, audios)


@torch.no_grad()
def batch_vqgan_decode(model, features):
    with _autocast_context(model):
        lengths = torch.tensor(
            [feature.shape[-1] for feature in features], device=model.device
        )
        max_length = lengths.max().item()
        padded = torch.stack(
            [
                torch.nn.functional.pad(feature, (0, max_length - feature.shape[-1]))
                for feature in features
            ]
        ).to(model.device)

        # If bs too large, we do micro batch decode
        audios = []
        for i in range(0, padded.shape[0], MICRO_BATCH_SIZE):
            audios.append(model.from_indices(padded[i : i + MICRO_BATCH_SIZE]).float())
        audios = torch.cat(audios, dim=0).cpu()
        # Each code frame decodes to frame_length samples (the codec is causal,
        # so the zero padding on the right does not affect the kept part).
        audio_lengths = (lengths * model.frame_length).cpu()

        return [
            audio[..., :length].numpy() for audio, length in zip(audios, audio_lengths)
        ]
