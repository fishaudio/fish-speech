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


@torch.no_grad()
@torch.autocast(device_type="cuda", dtype=torch.half)
def batch_encode(model, audios_list: list[bytes]):
    audios: list[torch.Tensor] = [
        (
            torch.from_numpy(
                librosa.load(io.BytesIO(audio), sr=model.spec_transform.sample_rate)[0]
            )[None]
            if isinstance(audio, bytes)
            else audio
        )
        for audio in audios_list
    ]

    lengths = torch.tensor([audio.shape[-1] for audio in audios], device=model.device)
    max_length = lengths.max().item()

    print(f"Encode max length: {max_length / model.spec_transform.sample_rate:.2f}s")

    padded = torch.stack(
        [
            torch.nn.functional.pad(audio, (0, int(max_length - audio.shape[-1])))
            for audio in audios
        ]
    ).to(model.device)

    features, feature_lengths = model.encode(padded, audio_lengths=lengths)
    features, feature_lengths = features.cpu(), feature_lengths.cpu()

    return [feature[..., :length] for feature, length in zip(features, feature_lengths)]


@cached(
    cache=LRUCache(maxsize=CACHE_MAXSIZE),
    key=lambda model, audios: (model.device, tuple(audios)),
)
def cached_vqgan_batch_encode(model, audios: list[bytes]):
    return batch_encode(model, audios)


@torch.no_grad()
@torch.autocast(device_type="cuda", dtype=torch.half)
def batch_vqgan_decode(model, features):
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
    audios, audio_lengths = [], []
    for i in range(0, padded.shape[0], MICRO_BATCH_SIZE):
        audio, audio_length = model.decode(
            padded[i : i + MICRO_BATCH_SIZE],
            feature_lengths=lengths[i : i + MICRO_BATCH_SIZE],
        )
        audios.append(audio)
        audio_lengths.append(audio_length)
    audios = torch.cat(audios, dim=0)
    audio_lengths = torch.cat(audio_lengths, dim=0)
    audios, audio_lengths = audios.cpu(), audio_lengths.cpu()

    return [audio[..., :length].numpy() for audio, length in zip(audios, audio_lengths)]


@torch.no_grad()
def batch_asr(model, lock, audios, sr, language="auto"):
    resampled_audios = []
    for audio in audios:
        audio = torchaudio.functional.resample(audio, sr, ASR_SAMPLE_RATE)
        assert audio.ndim == 1
        resampled_audios.append(audio)

    with lock:
        res = model.generate(
            input=resampled_audios,
            batch_size=len(resampled_audios),
            language=language,
            use_itn=True,
        )

    results = []
    for r, audio in zip(res, audios):
        text = r["text"]
        text = re.sub(r"<\|.*?\|>", "", text)
        duration = len(audio) / sr * 1000
        huge_gap = False

        if "timestamp" in r and len(r["timestamp"]) > 2:
            for timestamp_a, timestamp_b in zip(
                r["timestamp"][:-1], r["timestamp"][1:]
            ):
                # If there is a gap of more than 4 seconds, we consider it as a huge gap
                if timestamp_b[0] - timestamp_a[1] > HUGE_GAP_THRESHOLD:
                    huge_gap = True
                    break

            # Doesn't make sense to have a huge gap at the end
            if duration - r["timestamp"][-1][1] > HUGE_GAP_THRESHOLD:
                huge_gap = True

        results.append(
            {
                "text": text,
                "duration": duration,
                "huge_gap": huge_gap,
            }
        )

    return results
