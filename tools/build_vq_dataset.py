from functools import lru_cache
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict


@lru_cache(maxsize=1)
def get_phonemes():
    phones = {}
    phones.update(np.load("dump/phoneme_dev.npy", allow_pickle=True).item())
    phones.update(np.load("dump/phoneme_train.npy", allow_pickle=True).item())
    phones.update(
        np.load(
            "/home/fish/hubert-vq-vits/dump/phoneme_dev.npy", allow_pickle=True
        ).item()
    )
    phones.update(
        np.load(
            "/home/fish/hubert-vq-vits/dump/phoneme_train.npy", allow_pickle=True
        ).item()
    )
    print("Loaded phonemes")

    return phones


def parse_data(items):
    results = []
    phones = get_phonemes()

    for item_name, semantic_audio in zip(items["item_name"], items["semantic_audio"]):
        file_name = item_name
        if item_name.startswith("/wenet-speech-vocals"):
            file_name = "/home/fish/wenetspeech/dsall" + item_name

        wav_file = Path(file_name)
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
    test_dataset = Dataset.from_csv(
        ["dump/semantic_dev.tsv", "/home/fish/hubert-vq-vits/dump/semantic_dev.tsv"],
        delimiter="\t",
        split="test",
    )
    test_dataset = test_dataset.map(
        parse_data,
        num_proc=32,
        remove_columns=test_dataset.column_names,
        batched=True,
        batch_size=10000,
    )

    train_dataset = Dataset.from_csv(
        [
            "dump/semantic_train.tsv",
            "/home/fish/hubert-vq-vits/dump/semantic_train.tsv",
        ],
        delimiter="\t",
        split="train",
    )
    train_dataset = train_dataset.map(
        parse_data,
        num_proc=32,
        remove_columns=train_dataset.column_names,
        batched=True,
        batch_size=10000,
    )

    dataset = DatasetDict(
        {
            "train": train_dataset,
            "test": test_dataset,
        }
    )

    print(
        f"There are {len(dataset['train'])} training examples and {len(dataset['test'])} test examples"
    )
    print(dataset["train"][0])
    print(dataset["test"][1])

    dataset.push_to_hub("fishaudio/cn-hubert-25hz-vq", private=True)
