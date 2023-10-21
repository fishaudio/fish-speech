import json
import os
from pathlib import Path

import librosa
import torch
from datasets import Dataset
from multiprocess import set_start_method
from transformers import AutoProcessor, EncodecModel

set_start_method("spawn", force=True)

encodec_name = "facebook/encodec_24khz"
encodec_processor = AutoProcessor.from_pretrained(encodec_name)
encodec_model = EncodecModel.from_pretrained(encodec_name)
encodec_model.eval()


def tokenize(text, audio, sr=None, speaker=None):
    assert sr is None or sr == encodec_processor.sampling_rate

    if isinstance(audio, (str, Path)):
        audio, sr = librosa.load(audio, sr=sr, mono=True)

    prompt = "[INST] "
    if speaker:
        prompt += f"[SPK] {speaker} [/SPK] "
    prompt += f"{text} [/INST] "

    inputs = encodec_processor(
        raw_audio=audio, sampling_rate=sr, return_tensors="pt"
    ).to(encodec_model.device)
    outputs = encodec_model.encode(
        inputs["input_values"], inputs["padding_mask"], bandwidth=1.5, return_dict=True
    )

    assert outputs.audio_codes.dim() == 4  # [batch, channel, codebook, code]
    assert outputs.audio_codes.shape[0] == outputs.audio_codes.shape[1] == 1

    codes = outputs.audio_codes[0, 0, 0, :].long()
    codes_str = " ".join([f"<encodec_{int(c)}>" for c in codes.tolist()])
    prompt += codes_str

    return {
        "prompt": prompt,
        "codes": codes,
    }


def wrap_tokenize(x):
    device = torch.device("cuda", 0)

    if encodec_model.device != device:
        encodec_model.to(device)

    return tokenize(
        text=x["text"],
        audio=x["raw_path"],
        sr=encodec_processor.sampling_rate,
        speaker=x["speaker"],
    )


def generator_libritts_r():
    base = Path("dataset/tts/LibriTTS_R")

    for i in base.rglob("*.wav"):
        text_file = i.with_suffix(".normalized.txt")
        if not text_file.exists():
            continue

        text = text_file.read_text().strip()

        yield {
            "text": text,
            "speaker": f"libritts_{i.parent.parent.name}",
            "raw_path": str(i),
            "path": str(i.relative_to(base)),
        }


if __name__ == "__main__":
    dataset = Dataset.from_generator(generator_libritts_r)
    dataset = dataset.map(wrap_tokenize, num_proc=12)
    dataset = dataset.remove_columns(["raw_path"])

    dataset.save_to_disk("dataset/tts/libritts-r-encodec")
    dataset.push_to_hub("fishaudio/libritts-r-encodec", private=True)
