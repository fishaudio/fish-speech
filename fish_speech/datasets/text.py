import gzip
import io
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
import torch.nn.functional as F
import zstandard as zstd
from datasets.download.streaming_download_manager import xopen
from huggingface_hub import HfApi
from lightning import LightningDataModule
from torch.distributed import get_rank, get_world_size, is_initialized
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import AutoTokenizer

from fish_speech.datasets.prompts import asr_instructions, tts_instructions
from fish_speech.datasets.protos.text_data_pb2 import SampledData
from fish_speech.datasets.protos.text_data_stream import read_pb_stream
from fish_speech.text.clean import clean_text
from fish_speech.utils import RankedLogger
from fish_speech.utils.braceexpand import braceexpand

log = RankedLogger(__name__, rank_zero_only=True)

CODEBOOK_PAD_TOKEN_ID = 0
CODEBOOK_EOS_TOKEN_ID = 1
SKIP_TEXT_STRING = "<|skip_text|>"
DCTX = zstd.ZstdDecompressor(max_window_size=2**31)


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


class TextPretrainDataset(IterableDataset):
    def __init__(
        self,
        source: str,
        seed: int = 42,
        max_length: int = 1024,
        tokenizer: AutoTokenizer = None,
        num_codebooks: int = 2,
    ):
        super().__init__()

        self.source = Path(source)
        self.seed = seed
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.num_codebooks = num_codebooks

        if self.source.is_file():
            with open(self.source, "r") as f:
                files = f.read().strip().split("\n")
            self.root = self.source.parent
        else:
            files = [
                str(i.relative_to(self.source)) for i in self.source.rglob("*.jsonl")
            ]
            self.root = self.source

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

    def read_jsonl(self, filename: str):
        with open(self.root / filename, "rb") as f:
            if filename.endswith(".zst"):
                stream_reader = DCTX.stream_reader(f)
            elif filename.endswith(".gz"):
                stream_reader = gzip.open(f, "rb")
            elif filename.endswith(".jsonl"):
                stream_reader = f
            else:
                raise ValueError(f"Unknown file type: {filename}")

            stream = io.TextIOWrapper(stream_reader, encoding="utf-8")

            # Parse jsonl
            for line in stream:
                line = json.loads(line)
                yield line

    def parse_data(self, filename: str):
        for line in self.read_jsonl(filename):
            # encode
            tokens = self.tokenizer.encode(
                line["text"],
                add_special_tokens=False,
                truncation=False,
                max_length=10**6,
            )

            tokens = (
                [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
            )

            if len(tokens) > self.max_length:
                tokens = tokens[: self.max_length]

            tokens = self.pad_codebooks(tokens)
            labels = tokens.clone()
            tokens = tokens[:, :-1]
            labels = labels[:, 1:]
            labels[1:] = -100  # no loss on codebook

            yield {"tokens": tokens, "labels": labels}

    def pad_codebooks(self, tokens):
        placeholder_multi_codebook = (
            torch.zeros((self.num_codebooks, len(tokens)), dtype=torch.long)
            + CODEBOOK_PAD_TOKEN_ID
        )
        return torch.concat(
            [
                torch.tensor([tokens], dtype=torch.long),
                placeholder_multi_codebook,
            ],
            dim=0,
        )


class TextInstructionDataset(TextPretrainDataset):
    def parse_data(self, filename: str):
        for line in self.read_jsonl(filename):
            all_tokens, all_labels = [], []
            for idx, conversation in enumerate(line["conversations"]):
                mapped_speaker = {
                    "human": "user",
                    "gpt": "assistant",
                    "system": "system",
                }[conversation["from"]]
                packed = f"<|im_start|>{mapped_speaker}<|im_sep|>{conversation['value']}<|im_end|>{self.tokenizer.eos_token}"
                tokens = self.tokenizer.encode(
                    packed,
                    add_special_tokens=False,
                    truncation=False,
                    max_length=10**6,
                )

                if idx == 0:
                    tokens = [self.tokenizer.bos_token_id] + tokens

                all_tokens.extend(tokens[:-1])
                all_labels.extend(tokens[1:])

                if len(all_tokens) > self.max_length:
                    break

            tokens = self.pad_codebooks(all_tokens)
            labels = self.pad_codebooks(all_labels)

            if len(tokens) > self.max_length:
                tokens = tokens[: self.max_length]
                labels = labels[: self.max_length]

            yield {"tokens": tokens, "labels": labels}


class AutoTextSemanticInstructionDataset(IterableDataset):
    def __init__(
        self,
        proto_files: list[str],
        seed: int = 42,
        max_length: int = 1024,
        tokenizer: AutoTokenizer = None,
        causual: Union[bool, float] = True,
        num_codebooks: Optional[int] = None,
        skip_text_prob: float = 0.0,
        asr_prob: float = 0.0,
    ):
        """
        Args:
            proto_files: proto buf files if using local data
            seed: random seed
            max_length: max length of the text
            tokenizer: tokenizer
            causual: use causual sampling when using local data, disable will lead to random sampling
            num_codebooks: number of codebooks, if None, it will be automatically detected
            skip_text_prob: probability to skip the text (audio only), this only applies to interactive mode
            asr_prob: probability to use ASR
        """

        super().__init__()

        assert 0 <= skip_text_prob <= 1, "skip_text_prob must be in [0, 1]"
        assert 0 <= asr_prob <= 1, "asr_prob must be in [0, 1]"

        self.seed = seed
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.proto_files = proto_files
        self.causual = causual
        self.num_codebooks = num_codebooks
        self.skip_text_prob = skip_text_prob
        self.asr_prob = asr_prob

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

        causual = self.causual
        if isinstance(self.causual, float):
            causual = random.random() < self.causual

        if causual:
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
        response = self.sample_data()
        if len(response.samples) == 0:
            # Invalid group
            return None

        samples = list(response.samples)
        idx = 0
        remaining_tokens = self.max_length

        all_tokens, all_labels = [], []
        while remaining_tokens > 0 and len(samples) > 0:
            sentence = samples.pop(0)

            text = random.choice(sentence.texts)
            text, length = self.tokenize_sentence(text)
            remaining_tokens -= length + len(sentence.semantics[0].values)

            # For interactive mode, we only apply speaker for the first sentence
            # [INST] [SPK: speaker] text [/INST] ... [INST] text [/INST]

            if random.random() < self.asr_prob:
                tokens, labels = self.pack_sentences_asr(
                    sentences=[text],
                    semantics=[sentence.semantics],
                    add_bos=idx == 0,
                )
            else:
                tokens, labels = self.pack_sentences_tts(
                    sentences=[text],
                    semantics=[sentence.semantics],
                    add_bos=idx == 0,
                    skip_text=random.random() < self.skip_text_prob,
                )

            all_tokens.append(tokens)
            all_labels.append(labels)

            idx += 1

        tokens = torch.cat(all_tokens, dim=1)
        labels = torch.cat(all_labels, dim=1)

        # Verify that the length is correct
        assert tokens.size(1) == labels.size(1), f"{tokens.size(1)} != {labels.size(1)}"

        # Verify bos token
        assert tokens[0, 0] == self.tokenizer.bos_token_id

        return {"tokens": tokens, "labels": labels}

    def pack_sentences_tts(
        self,
        sentences: list[str],
        semantics: list,
        add_bos: bool = True,
        skip_text: bool = False,
    ):
        cated_sentences = " ".join(sentences)
        if skip_text:
            cated_sentences = SKIP_TEXT_STRING

        cated_sentences = random.choice(tts_instructions) + cated_sentences

        final_text = "<|im_start|>user<|im_sep|>" + cated_sentences + "<|im_end|>"
        final_text = final_text + f"<|im_start|>assistant<|im_sep|>"

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
            + self.tokenizer.convert_tokens_to_ids(["<|im_end|>"])
            + [self.tokenizer.eos_token_id]
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

        if skip_text:
            # If text is not provided, the sentence is used for condition only, all labels are -100
            torch.fill_(labels, -100)
            return tokens, labels

        # Mask out instruction tokens
        labels[:, : (prompt_length + bos_bias)] = -100

        tokens = tokens[:, :-1]
        labels = labels[:, 1:]

        # Verify the padding is correct, and the last token is eos
        assert add_bos is False or tokens[0, 0] == self.tokenizer.bos_token_id
        assert (tokens[1:, : prompt_length + bos_bias] == CODEBOOK_PAD_TOKEN_ID).all()
        assert labels[0, -1] == self.tokenizer.eos_token_id
        assert (labels[1:, -2:] == CODEBOOK_EOS_TOKEN_ID).all()

        return tokens, labels

    def pack_sentences_asr(
        self,
        sentences: list[str],
        semantics: list,
        speaker: Optional[str] = None,
        add_bos: bool = True,
    ):
        if speaker is None:
            speaker = "assistant"

        cated_sentences = " ".join(sentences)
        prompt = random.choice(asr_instructions)

        final_text = "<|im_start|>user<|im_sep|>" + prompt

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
            + self.tokenizer.convert_tokens_to_ids(["<|im_end|>"])
        )

        if add_bos:
            tokens = [self.tokenizer.bos_token_id] + tokens

        response_header = self.tokenizer.encode(
            "<|im_start|>assistant<|im_sep|>",
            add_special_tokens=False,
            truncation=False,
            max_length=10**6,
        )
        encoded_result = self.tokenizer.encode(
            f"{cated_sentences}<|im_end|>",
            add_special_tokens=False,
            truncation=False,
            max_length=10**6,
        )

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
            book.extend(
                [CODEBOOK_PAD_TOKEN_ID]
                * (len(response_header) + len(encoded_result) + 2)
            )

        instruction_length = len(tokens)
        tokens = [
            tokens + response_header + encoded_result + [self.tokenizer.eos_token_id]
        ] + codes
        tokens = torch.tensor(tokens, dtype=torch.long)
        labels = tokens.clone()

        # Mask out instruction tokens
        labels[:, : instruction_length + len(response_header)] = -100
        tokens = tokens[:, :-1]
        labels = labels[:, 1:]

        # Verify the padding is correct, and the last token is eos
        assert add_bos is False or tokens[0, 0] == self.tokenizer.bos_token_id
        assert (tokens[1:, : prompt_length + bos_bias] == CODEBOOK_PAD_TOKEN_ID).all()
        assert (tokens[1:, instruction_length:] == CODEBOOK_PAD_TOKEN_ID).all()
        assert labels[0, -1] == self.tokenizer.eos_token_id

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
        train_dataset: Union[
            AutoTextSemanticInstructionDataset,
            TextPretrainDataset,
            TextInstructionDataset,
            InterleaveDataset,
        ],
        val_dataset: Union[
            AutoTextSemanticInstructionDataset,
            TextPretrainDataset,
            TextInstructionDataset,
            InterleaveDataset,
        ],
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
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=TextDataCollator(self.tokenizer, self.max_length),
            num_workers=self.num_workers,
            persistent_workers=True,
        )


if __name__ == "__main__":
    from tqdm import tqdm

    # ds = AutoTextSemanticInstructionDataset(
    #     ["data/protos/eleven_labs"],
    #     tokenizer=AutoTokenizer.from_pretrained("checkpoints/fish-speech-agent-1"),
    #     skip_text_prob=0.5,
    #     asr_prob=1.0
    # )

    ds = TextInstructionDataset(
        source="data/openhermes2_5",
        tokenizer=AutoTokenizer.from_pretrained("checkpoints/fish-speech-agent-1"),
    )

    for i in ds:
        # print(ds.tokenizer.decode(i["tokens"][0], skip_special_tokens=False))
        # i["labels"][0][i["labels"][0] == -100] = 0
        # print(ds.tokenizer.decode(i["labels"][0], skip_special_tokens=False))

        length = i["tokens"].size(1)
        for j in range(length):
            print(
                ds.tokenizer.decode(i["tokens"][0, j]),
                i["tokens"][:, j],
                i["labels"][:, j],
            )
            input()
        break
