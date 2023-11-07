import json
import random
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from random import Random
from typing import Optional, Union

import numpy as np
import pyarrow.parquet as pq
import torch
from datasets.download.streaming_download_manager import xopen
from huggingface_hub import HfApi
from lightning import LightningDataModule
from torch.distributed import get_rank, get_world_size, is_initialized
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from transformers import AutoTokenizer

from fish_speech.text import clean_text, g2p
from fish_speech.utils import RankedLogger
from fish_speech.utils.braceexpand import braceexpand

log = RankedLogger(__name__, rank_zero_only=True)


def split_by_rank_worker(files):
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


class StreamTextDataset(IterableDataset):
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

    def __iter__(self):
        files = split_by_rank_worker(self.files)
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


# @dataclass
# class DatasetLine:
#     text: str
#     semantic: str
#     speaker: str


class AutoAugTextDataset(IterableDataset):
    """
    Auto Augment Dataset by Speaker

    1. Random concatenate multiple sentences from the same speaker to form a longer sentence
    2. Automatically normalize the text
    3. Mix text and phones
    """

    def __init__(
        self,
        jsonl_files: list[str],
        seed: int = 42,
        phones_prob: float = 0.5,
        max_length: int = 1024,
        order: Optional[list[str]] = None,
        tokenizer: AutoTokenizer = None,
    ):
        super().__init__()

        self.jsonl_files = jsonl_files
        self.seed = seed
        self.phones_prob = phones_prob
        self.max_length = max_length
        self.order = order
        self.tokenizer = tokenizer

        # Read all lines, and group by speaker
        self.speakers = {}
        self.lines = []

        for filename in self.jsonl_files:
            lines = Path(filename).read_text().splitlines()
            for json_line in lines:
                line = json.loads(json_line)
                speaker = line.get("speaker", None)

                if speaker not in self.speakers:
                    self.speakers[speaker] = []

                self.lines.append(line)
                self.speakers[speaker].append(line)

        # Shuffle the lines
        Random(seed).shuffle(self.lines)

    def __iter__(self):
        lines = split_by_rank_worker(self.lines)
        random.shuffle(lines)

        for line in lines:
            yield self.augment(line)

    def tokenize_sentence(
        self, sentence: str, semantic: list[int], mode: str = "sample"
    ):
        sentence = clean_text(sentence)

        if (
            mode == "sample" and (random.random() < self.phones_prob)
        ) or mode == "phones":
            sentence = " ".join([t for _, t in g2p(sentence, order=self.order)])

        semantic = " ".join([f"<semantic_{i}>" for i in semantic])

        tokens = self.tokenizer.encode(
            f"{sentence} {semantic}", max_length=10**6, add_special_tokens=False
        )
        return sentence, semantic, len(tokens)

    def augment(self, line):
        speaker = line.get("speaker", None)

        # 20% to pure text or pure phones
        mode = "sample"
        if random.random() < 0.2:
            mode = random.choice(["text", "phones"])

        if speaker is None:
            a, b, _ = self.tokenize_sentence(line["text"], line["semantic"], mode=mode)
            return {"text": f"[INST] {a} [/INST] {b} </s>"}

        # Random sample based on speaker using a truncated normal distribution
        a = torch.tensor([0], dtype=torch.float32)
        torch.nn.init.trunc_normal_(
            a,
            mean=self.max_length // 2,
            std=self.max_length // 4,
            a=0,
            b=self.max_length,
        )
        remaining_tokens = a.long().item() - 4

        final_text, final_semantic = [], []

        # Shuffle unique lines
        idxs = list(range(len(self.speakers[speaker])))
        random.shuffle(idxs)

        while remaining_tokens > 0 and len(idxs) > 0:
            line = self.speakers[speaker][idxs.pop()]
            text, semantic, length = self.tokenize_sentence(
                line["text"], line["semantic"], mode=mode
            )
            remaining_tokens -= length
            final_text.append(text)
            final_semantic.append(semantic)

        final_text = " ".join(final_text)
        final_semantic = " ".join(final_semantic)

        return {"text": f"[INST] {final_text} [/INST] {final_semantic} </s>"}


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
        train_dataset: Union[StreamTextDataset, AutoAugTextDataset, InterleaveDataset],
        val_dataset: Union[StreamTextDataset, AutoAugTextDataset, InterleaveDataset],
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
    import json

    # data/Genshin/English/Aabid/vo_KVCOP001_1907808_aabid_01.lab
    # all_files = [i for i in Path("data/Genshin/English").rglob("*.lab")]
    # with open("test.jsonl", "w") as f:
    #     for i in all_files:
    #         wav_file = i.with_suffix(".wav")
    #         duration = float(Path(wav_file).stat().st_size) / 2 / 44100
    #         eta_tokens = duration * 25
    #         fake_tokens = [random.randint(0, 2048) for _ in range(int(eta_tokens))]
    #         f.write(json.dumps({"text": Path(i).read_text(), "speaker": i.parent.name, "semantic": fake_tokens}) + "\n")

    ds = AutoAugTextDataset(
        jsonl_files=["test.jsonl"],
        order=["en"],
        tokenizer=AutoTokenizer.from_pretrained(
            "fishaudio/speech-lm-300m", revision="text-pretrain-10k-phones"
        ),
    )

    dm = TextDataModule(
        train_dataset=ds,
        val_dataset=ds,
        tokenizer=ds.tokenizer,
        batch_size=2,
        max_length=1024,
        num_workers=0,
    )

    for batch in dm.train_dataloader():
        print(batch)
        break
