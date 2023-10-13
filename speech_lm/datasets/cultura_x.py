import random
from dataclasses import dataclass
from logging import getLogger
from random import Random

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datasets.download.streaming_download_manager import xopen
from torch.distributed import get_rank, get_world_size, is_initialized
from torch.utils.data import IterableDataset, get_worker_info
from transformers import AutoTokenizer

from speech_lm.utils.braceexpand import braceexpand

SUBSETS = {
    "en": "en_part_{00000..03071}",
    "zh": "zh_part_{00000..00319}",
    "ja": "ja_part_{00000..00159}",
}

log = getLogger(__name__)


class CulturaXDataset(IterableDataset):
    def __init__(self, lang: str, seed: int = 42, parquet_batch_size: int = 10000):
        super().__init__()

        self.lang = lang
        self.seed = seed
        self.parquet_batch_size = parquet_batch_size

        # Get sharded files
        self.files = sorted(list(braceexpand(f"{lang}/{SUBSETS[lang]}.parquet")))
        Random(seed).shuffle(self.files)

    def get_data_splits(self, files):
        # We need to know the total number of devices
        # to split the data properly

        total_devices = 1
        if is_initialized():
            total_devices = get_world_size()

        worker_info = get_worker_info()
        if worker_info is not None:
            total_devices *= worker_info.num_workers

        if len(files) < total_devices:
            # Repeat the files N times to match the number of devices
            files = files * (total_devices // len(files) + 1)

        # DDP
        if is_initialized():
            files = files[get_rank() :: get_world_size()]

        # Split by worker
        if worker_info is not None:
            files = files[worker_info.id :: worker_info.num_workers]

        return files

    def __iter__(self):
        files = self.get_data_splits(self.files)
        random.shuffle(files)

        for filename in files:
            try:
                yield from self.parse_data(filename)
            except Exception as e:
                log.exception(f"Failed to parse {filename}: {e}")

    def parse_data(self, filename: str):
        url = f"https://huggingface.co/datasets/uonlp/CulturaX/resolve/main/{filename}"

        with xopen(url, mode="rb") as stream:
            parquet_file = pq.ParquetFile(stream)

            for batch in parquet_file.iter_batches(
                batch_size=self.parquet_batch_size, columns=["text"]
            ):
                # In-batch shuffling
                texts = [{"text": text.as_py()} for text in batch["text"]]
                random.shuffle(texts)
                yield from texts


@dataclass
class CulutreXCollator:
    tokenizer: AutoTokenizer
    max_length: int = 512

    def __call__(self, examples):
        texts = []

        for example in examples:
            text = example["text"]

            if len(text) <= self.max_length:
                texts.append(text)
            else:
                start = random.randint(0, len(text) - self.max_length)
                texts.append(text[start : start + self.max_length])

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        data = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        data["labels"] = data["input_ids"].clone()
        data["labels"][data["attention_mask"] == 0] = -100

        return data


class InterleaveDataset(IterableDataset):
    def __init__(
        self,
        datasets: list[IterableDataset],
        probabilities: list[float],
        seed: int = 42,
    ):
        super().__init__()

        self.datasets = datasets
        self.probabilities = probabilities
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        dataset_iterators = [iter(dataset) for dataset in self.datasets]

        while True:
            # Random choice one
            dataset_idx = rng.choice(len(self.datasets), p=self.probabilities)
            dataset_iterator = dataset_iterators[dataset_idx]

            try:
                yield next(dataset_iterator)
            except StopIteration:
                # Exhausted, create a new iterator
                dataset_iterators[dataset_idx] = iter(self.datasets[dataset_idx])
                yield next(dataset_iterators[dataset_idx])


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset_en = CulturaXDataset("en")
    dataset_ja = CulturaXDataset("ja")
    dataset = InterleaveDataset([dataset_en, dataset_ja], [0.5, 0.5])
    collator = CulutreXCollator(AutoTokenizer.from_pretrained("gpt2"))

    for batch in DataLoader(dataset, batch_size=4, collate_fn=collator, num_workers=4):
        print(batch)
