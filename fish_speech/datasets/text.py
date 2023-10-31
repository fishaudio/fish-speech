import random
from dataclasses import dataclass
from itertools import chain
from random import Random
from typing import Optional, Union

import numpy as np
import pyarrow.parquet as pq
from datasets.download.streaming_download_manager import xopen
from huggingface_hub import HfApi
from lightning import LightningDataModule
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from torch.distributed import get_rank, get_world_size, is_initialized
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import AutoTokenizer

from fish_speech.utils import RankedLogger
from fish_speech.utils.braceexpand import braceexpand

log = RankedLogger(__name__, rank_zero_only=True)


class TextDataset(IterableDataset):
    def __init__(
        self,
        files: Optional[Union[list[str], str]] = None,
        prefix: Optional[str] = None,
        seed: int = 42,
        parquet_batch_size: int = 10000,
        repo: str = "uonlp/CulturaX",
    ):
        super().__init__()

        self.seed = seed
        self.parquet_batch_size = parquet_batch_size
        self.repo = repo

        if files is None and prefix is None:
            raise ValueError("Either files or prefix must be specified")

        if prefix is not None:
            files = HfApi().list_repo_files(repo, repo_type="dataset")
            files = [
                f for f in files if f.startswith(prefix) and f.endswith(".parquet")
            ]
            log.info(f"Found {len(files)} files in {repo} with prefix {prefix}")
        else:
            if isinstance(files, str):
                files = [files]

            files = list(chain.from_iterable(map(braceexpand, files)))
            log.info(f"Expanded {len(files)} files in {repo}")

        # Get sharded files
        self.files = sorted(files)
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
        url = f"https://huggingface.co/datasets/{self.repo}/resolve/main/{filename}"

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
class TextDataCollator:
    tokenizer: AutoTokenizer
    max_length: int = 512

    def __call__(self, examples):
        texts = [i["text"] for i in examples]

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


class TextDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: Union[TextDataset, InterleaveDataset],
        val_dataset: Union[TextDataset, InterleaveDataset],
        batch_size: int = 32,
        tokenizer: AutoTokenizer = None,
        max_length: int = 1024,
        num_workers: int = 4,
    ):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=TextDataCollator(self.tokenizer, self.max_length),
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=TextDataCollator(self.tokenizer, self.max_length),
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    dm = TextDataModule(
        InterleaveDataset(
            datasets=[
                TextDataset(
                    prefix="en/en_part_",
                ),
                TextDataset(
                    prefix="zh/zh_part_",
                ),
                TextDataset(
                    prefix="ja/ja_part_",
                ),
            ],
            probabilities=[0.8, 0.1, 0.1],
        ),
        TextDataset(
            files="ja/ja_part_{00000..00159}",
        ),
        batch_size=2,
        tokenizer=AutoTokenizer.from_pretrained("bert-base-multilingual-cased"),
    )

    for batch in dm.train_dataloader():
        print(batch)
        break
