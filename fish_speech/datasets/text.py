import random
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from random import Random
from typing import Optional, Union

import grpc
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from datasets.download.streaming_download_manager import xopen
from huggingface_hub import HfApi
from lightning import LightningDataModule
from torch.distributed import get_rank, get_world_size, is_initialized
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import AutoTokenizer

from fish_speech.datasets.protos.text_data_pb2 import SampledData
from fish_speech.datasets.protos.text_data_stream import read_pb_stream
from fish_speech.text.clean import clean_text
from fish_speech.utils import RankedLogger
from fish_speech.utils.braceexpand import braceexpand

log = RankedLogger(__name__, rank_zero_only=True)

CODEBOOK_PAD_TOKEN_ID = 0
CODEBOOK_EOS_TOKEN_ID = 1


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
        max_length: int = 1024,
        tokenizer: AutoTokenizer = None,
    ):
        super().__init__()

        self.seed = seed
        self.parquet_batch_size = parquet_batch_size
        self.repo = repo
        self.max_length = max_length
        self.tokenizer = tokenizer

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
        for data in self.parse_data_internal(filename):
            text = data["text"]

            # encode
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,
                max_length=10**6,
            )

            # Random choice self.max_length
            if len(tokens) > self.max_length:
                start = random.randint(0, len(tokens) - self.max_length)
                tokens = tokens[start : start + self.max_length - 1]

            tokens = (
                [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
            )
            # Pad dims
            placeholder_multi_codebook = torch.zeros((4, len(tokens)), dtype=torch.long)

            tokens = torch.concat(
                [
                    torch.tensor([tokens], dtype=torch.long),
                    placeholder_multi_codebook,
                ],
                dim=0,
            )
            labels = tokens.clone()
            tokens = tokens[:, :-1]
            labels = labels[:, 1:]
            labels[1:] = -100  # remove all placeholders

            yield {"tokens": tokens, "labels": labels}

    def parse_data_internal(self, filename: str):
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


class AutoAugTextDataset(IterableDataset):
    """
    Auto Augment Dataset by Speaker

    1. Random concatenate multiple sentences from the same speaker to form a longer sentence
    2. Automatically normalize the text

    For interactive mode, we use the following format (multiple sequences):
    <s> [INST] [SPK: speaker] text [/INST] ... [INST] text [/INST] </s>

    For non-interactive mode, we use the following format (one long sequence):
    <s> [INST] text [/INST] ... </s>
    """

    def __init__(
        self,
        proto_files: list[str],
        seed: int = 42,
        interactive_prob: float = 0.5,
        max_length: int = 1024,
        tokenizer: AutoTokenizer = None,
        use_speaker: bool = True,
        causual: bool = True,
        use_negative_samples: bool = False,
        num_codebooks: Optional[int] = None,
    ):
        """
        Args:
            proto_files: proto buf files if using local data
            seed: random seed
            interactive_prob: probability to use interactive mode
            max_length: max length of the text
            tokenizer: tokenizer
            use_speaker: include speaker information in the prompt
            causual: use causual sampling when using local data, disable will lead to random sampling
            use_negative_samples: generate negative samples
            num_codebooks: number of codebooks, if None, it will be automatically detected
        """

        super().__init__()

        assert 0 <= interactive_prob <= 1, "interactive_prob must be in [0, 1]"

        self.seed = seed
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.interactive_prob = interactive_prob
        self.use_speaker = use_speaker
        self.proto_files = proto_files
        self.causual = causual
        self.use_negative_samples = use_negative_samples
        self.num_codebooks = num_codebooks

        self.semantic_token_id = self.tokenizer.convert_tokens_to_ids("<|semantic|>")
        self.groups = None

    def init_mock_data_server(self):
        if self.groups is not None:
            return

        # Expand the proto files
        expanded_proto_files = []
        for filename in self.proto_files:
            for i in braceexpand(filename):
                i = Path(i)
                if i.is_file():
                    expanded_proto_files.append(i)
                elif i.is_dir():
                    expanded_proto_files.extend(i.rglob("*.proto"))
                    expanded_proto_files.extend(i.rglob("*.protos"))
                else:
                    raise ValueError(f"{i} is not a file or directory")

        expanded_proto_files = sorted(expanded_proto_files)
        Random(self.seed).shuffle(expanded_proto_files)

        self.groups = []
        shard_proto_files = split_by_rank_worker(expanded_proto_files)
        log.info(
            f"Reading {len(shard_proto_files)} / {len(expanded_proto_files)} files"
        )

        count = 0
        for filename in shard_proto_files:
            with open(filename, "rb") as f:
                for text_data in read_pb_stream(f):
                    self.groups.append(text_data)
                    count += 1

        log.info(f"Read total {count} groups of data")

        # Shuffle the lines
        Random(self.seed).shuffle(self.groups)
        self.group_weights = [len(i.sentences) for i in self.groups]

    def __iter__(self):
        while True:
            yield self.augment()

    def tokenize_sentence(self, sentence: str):
        sentence = clean_text(sentence)
        tokens = self.tokenizer.encode(
            f"{sentence}",
            max_length=10**6,
            add_special_tokens=False,
            truncation=False,
        )
        return sentence, len(tokens)

    def sample_data(self):
        if self.groups is None:
            self.init_mock_data_server()

        # Shuffle unique lines, estimate that each sample is at least 20 tokens
        num_samples = self.max_length // 20

        # choice group based on their number of samples
        group = random.choices(self.groups, weights=self.group_weights, k=1)[0]

        if self.causual:
            # Sample in order
            if num_samples >= len(group.sentences):
                samples = group.sentences
            else:
                begin = random.randint(0, len(group.sentences) - num_samples)
                samples = group.sentences[begin : begin + num_samples]
        else:
            samples = random.choices(
                group.sentences, k=min(num_samples, len(group.sentences))
            )

        return SampledData(
            source=group.source,
            name=group.name,
            samples=samples,
        )

    def augment(self):
        final_text, final_semantic = [], []
        response = self.sample_data()
        if len(response.samples) == 0:
            # Invalid group
            return None

        samples = list(response.samples)
        idx = 0
        use_interactive = random.random() < self.interactive_prob

        if use_interactive is False:
            # Random sample based on speaker using a truncated normal distribution
            a = torch.tensor([0], dtype=torch.float32)
            torch.nn.init.trunc_normal_(
                a,
                mean=self.max_length // 2,
                std=self.max_length // 4,
                a=10,
                b=self.max_length,
            )
            remaining_tokens = a.long().item() - 4
        else:
            remaining_tokens = self.max_length

        all_tokens, all_labels = [], []
        while remaining_tokens > 0 and len(samples) > 0:
            sentence = samples.pop(0)

            text = random.choice(sentence.texts)
            text, length = self.tokenize_sentence(text)
            remaining_tokens -= length + len(sentence.semantics[0].values)

            if use_interactive is False:
                final_text.append(text)
                final_semantic.append(sentence.semantics)
            else:
                # For interactive mode, we only apply speaker for the first sentence
                # [INST] [SPK: speaker] text [/INST] ... [INST] text [/INST]
                tokens, labels = self.pack_sentences(
                    sentences=[text],
                    semantics=[sentence.semantics],
                    speaker=response.name if (self.use_speaker and idx == 0) else None,
                    add_bos=idx == 0,
                )

                all_tokens.append(tokens)
                all_labels.append(labels)

            idx += 1

        if use_interactive is False:
            tokens, labels = self.pack_sentences(
                final_text,
                semantics=final_semantic,
                speaker=response.name if self.use_speaker else None,
                add_bos=True,
            )
            all_tokens.append(tokens)
            all_labels.append(labels)

        tokens = torch.cat(all_tokens, dim=1)
        labels = torch.cat(all_labels, dim=1)

        # Verify that the length is correct
        assert tokens.size(1) == labels.size(1), f"{tokens.size(1)} != {labels.size(1)}"

        # Verify bos token
        assert tokens[0, 0] == self.tokenizer.bos_token_id

        data = {"tokens": tokens, "labels": labels}

        if self.use_negative_samples:
            negative_samples = self.generate_negative_samples(all_tokens, all_labels)
            data.update(negative_samples)

        return data

    def generate_negative_samples(self, all_tokens, all_labels):
        new_tokens, new_labels = [], []

        for tokens, labels in zip(all_tokens, all_labels):
            # If all codebooks are not -100, we find where it starts
            start = torch.where(labels[1:].sum(0) != -100 * (labels.size(0) - 1))[0][0]
            assert (labels[1:, start:] != -100).all()  # This shouldn't happen

            mode = random.choice(["repeat", "lost", "noise"])
            begin = random.randint(start, labels.size(1) - 1)
            end = random.randint(begin, labels.size(1) - 1)

            if mode == "repeat":
                tokens = torch.cat(
                    [
                        tokens[:, :begin],
                        tokens[:, begin:end],
                        tokens[:, begin:end],
                        tokens[:, end:],
                    ],
                    dim=1,
                )
                labels = torch.cat(
                    [
                        labels[:, :begin],
                        labels[:, begin:end],
                        labels[:, begin:end],
                        labels[:, end:],
                    ],
                    dim=1,
                )
            elif mode == "lost":
                tokens = torch.cat([tokens[:, :begin], tokens[:, end:]], dim=1)
                labels = torch.cat([labels[:, :begin], labels[:, end:]], dim=1)
            elif mode == "noise":
                middle_tokens, middle_labels = (
                    tokens[:, begin:end],
                    labels[:, begin:end],
                )
                random_order0 = torch.randperm(middle_tokens.size(1))
                random_order1 = torch.randperm(middle_tokens.size(1))
                middle_tokens = middle_tokens[:, random_order0]
                middle_labels = middle_labels[:, random_order1]
                tokens = torch.cat(
                    [tokens[:, :begin], middle_tokens, tokens[:, end:]], dim=1
                )
                labels = torch.cat(
                    [labels[:, :begin], middle_labels, labels[:, end:]], dim=1
                )

            new_tokens.append(tokens)
            new_labels.append(labels)

        tokens = torch.cat(new_tokens, dim=1)
        labels = torch.cat(new_labels, dim=1)

        # Verify that the length is correct
        assert tokens.size(1) == labels.size(1), f"{tokens.size(1)} != {labels.size(1)}"

        return {"negative_tokens": tokens, "negative_labels": labels}

    def pack_sentences(
        self,
        sentences: list[str],
        semantics=list,
        speaker: Optional[str] = None,
        add_bos: bool = True,
    ):
        if speaker is not None:
            sentences = [f"[SPK: {speaker}]"] + sentences

        final_text = "<|im_start|>user<|im_sep|>" + " ".join(sentences) + "<|im_end|>"
        final_text = final_text + "<|im_start|>assistant<|im_sep|>"

        encoded = self.tokenizer.encode(
            final_text,
            add_special_tokens=False,
            truncation=False,
            max_length=10**6,
        )
        semantic_length = sum([len(i[0].values) for i in semantics])
        prompt_length = len(encoded)
        num_codebooks = (
            len(semantics[0]) if self.num_codebooks is None else self.num_codebooks
        )

        bos_bias = 1 if add_bos else 0

        # Pack the tokens and semantics (add <s> and </s> to semantic tokens)
        tokens = (
            encoded
            + [self.semantic_token_id] * semantic_length
            + self.tokenizer.convert_tokens_to_ids(
                ["<|im_end|>", "<|end_of_sequence|>"]
            )
        )

        if add_bos:
            tokens = [self.tokenizer.bos_token_id] + tokens

        # Codebook bos/padding: 0, eos: 1
        codes = [
            [CODEBOOK_PAD_TOKEN_ID] * (prompt_length + bos_bias)
            for _ in range(num_codebooks)
        ]
        for segment in semantics:
            for book_idx, book in zip(range(num_codebooks), segment):
                for j in book.values:
                    codes[book_idx].append(int(j) + 2)

        for book in codes:
            book.extend([CODEBOOK_EOS_TOKEN_ID] * 2)

        tokens = [tokens] + codes

        tokens = torch.tensor(tokens, dtype=torch.long)
        labels = tokens.clone()

        # Mask out the <s> tokens for semantic, predict semantic tokens only
        # Since we don't mask out the input tokens, the language modeling still works
        labels[1:, : (prompt_length + bos_bias)] = -100

        tokens = tokens[:, :-1]
        labels = labels[:, 1:]

        # Verify the padding is correct, and the last token is eos
        assert add_bos is False or tokens[0, 0] == self.tokenizer.bos_token_id
        assert (tokens[1:, : prompt_length + bos_bias] == CODEBOOK_PAD_TOKEN_ID).all()
        assert labels[0, -1] == self.tokenizer.eos_token_id
        assert (labels[1:, -2:] == CODEBOOK_EOS_TOKEN_ID).all()

        return tokens, labels


@dataclass
class TextDataCollator:
    tokenizer: AutoTokenizer
    max_length: int = 1024

    def __call__(self, examples):
        if "negative_tokens" in examples:
            positive_examples = []
            negative_examples = []

            for i in examples:
                positive_examples.append(
                    {
                        "tokens": i["tokens"],
                        "labels": i["labels"],
                    }
                )
                negative_examples.append(
                    {
                        "tokens": i["negative_tokens"],
                        "labels": i["negative_labels"],
                    }
                )

            examples = positive_examples + negative_examples

        return self.batchify(examples)

    def batchify(self, examples, tokens_key="tokens", labels_key="labels"):
        tokens, attention_masks, labels = [], [], []

        # Calculate the max length
        max_tokens_length = 0
        for example in examples:
            max_tokens_length = max(max_tokens_length, example[tokens_key].size(1))
        max_tokens_length = min(max_tokens_length, self.max_length)

        for example in examples:
            _tokens = example[tokens_key][:, :max_tokens_length]
            _labels = example[labels_key][:, :max_tokens_length]
            _attention_mask = torch.ones((max_tokens_length,), dtype=torch.bool)
            tokens_length = _tokens.size(1)
            _attention_mask[:tokens_length] = False

            assert tokens_length == _labels.size(
                1
            ), f"{tokens_length} != {_labels.size(1)}"

            if tokens_length < max_tokens_length:
                _tokens = F.pad(
                    _tokens,
                    (0, max_tokens_length - tokens_length),
                    value=self.tokenizer.eos_token_id,
                )
                _tokens[1:, tokens_length:] = CODEBOOK_PAD_TOKEN_ID
                _labels = F.pad(
                    _labels, (0, max_tokens_length - _labels.size(1)), value=-100
                )

            tokens.append(_tokens)
            attention_masks.append(_attention_mask)
            labels.append(_labels)

        tokens = torch.stack(tokens, dim=0)
        attention_masks = torch.stack(attention_masks, dim=0)
        labels = torch.stack(labels, dim=0)

        return {
            "inputs": tokens,
            "attention_masks": attention_masks,
            "labels": labels,
        }


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
    from tqdm import tqdm

    ds = AutoAugTextDataset(
        ["data/protos"],
        tokenizer=AutoTokenizer.from_pretrained("fishaudio/fish-speech-1"),
        use_speaker=False,
        interactive_prob=1.0,
        use_negative_samples=False,
    )

    # ds = AutoAugTextDataset(
    #     tokenizer=AutoTokenizer.from_pretrained("fishaudio/speech-lm-v1"),
    #     use_speaker=True,
    #     interactive_prob=1.0,
    #     use_data_server=False,
    #     proto_files=["data/wenet-speech.protos"],
    # )

    dm = TextDataModule(
        train_dataset=ds,
        val_dataset=ds,
        tokenizer=ds.tokenizer,
        batch_size=2,
        max_length=1024,
        num_workers=0,
    )

    for batch in tqdm(dm.train_dataloader()):
        print(batch)
        break
