from functools import partial
from pathlib import Path

import numpy as np
from datasets import Dataset


def parse_data(phones, items):
    results = []

    for item_name, semantic_audio in zip(items["item_name"], items["semantic_audio"]):
        wav_file = Path(item_name)
        text_file = wav_file.with_suffix(".txt")

        if not text_file.exists():
            text_file = wav_file.with_suffix(".lab")

        if not text_file.exists():
            print(f"Missing {text_file}")
            return None

        text = text_file.read_text().strip()
        semantic = [f"<semantic_{x}>" for x in semantic_audio.split(" ")]
        semantic = " ".join(semantic)
        results.append(f"[INST] {text} [/INST] {semantic} </s>")
        results.append(f"[INST] {phones[item_name]} [/INST] {semantic} </s>")

    return {
        "text": results,
    }


if __name__ == "__main__":
    phones = np.load("dump/phoneme_train.npy", allow_pickle=True).item()
    phones1 = np.load(
        "/home/fish/hubert-vq-vits/dump/phoneme_train.npy", allow_pickle=True
    ).item()
    phones.update(phones1)
    print(len(phones))

    dataset = Dataset.from_csv(
        [
            "dump/semantic_train.tsv",
            "/home/fish/hubert-vq-vits/dump/semantic_train.tsv",
        ],
        delimiter="\t",
        split="train",
    )
    dataset = dataset.map(
        partial(parse_data, phones),
        num_proc=32,
        remove_columns=dataset.column_names,
        batched=True,
    )
    print(len(dataset), dataset[0])
    dataset.push_to_hub("fishaudio/cn-hubert-25hz-vq", private=True)
