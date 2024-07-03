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
from datasets.download.streaming_download_manager import xopen
from huggingface_hub import HfApi
from lightning import LightningDataModule
from torch.distributed import get_rank, get_world_size, is_initialized
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import AutoTokenizer

from fish_speech.conversation import CODEBOOK_PAD_TOKEN_ID
from fish_speech.datasets.protos.text_data_pb2 import SampledData
from fish_speech.datasets.protos.text_data_stream import read_pb_stream
from fish_speech.text.clean import clean_text
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


class AutoTextSemanticInstructionDataset(IterableDataset):
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
        use_speaker: bool | float = True,
        causal: bool = True,
        num_codebooks: Optional[int] = None,
        skip_text_prob: float = 0.0,
    ):
        """
        Args:
            proto_files: proto buf files if using local data
            seed: random seed
            interactive_prob: probability to use interactive mode
            max_length: max length of the text
            tokenizer: tokenizer
            use_speaker: include speaker information in the prompt
            causal: use causal sampling when using local data, disable will lead to random sampling
            num_codebooks: number of codebooks, if None, it will be automatically detected
            skip_text_prob: probability to skip the text (audio only), this only applies to interactive mode
        """

        super().__init__()

        assert 0 <= interactive_prob <= 1, "interactive_prob must be in [0, 1]"

        self.seed = seed
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.interactive_prob = interactive_prob
        self.use_speaker = use_speaker
        self.proto_files = proto_files
        self.causal = causal
        self.num_codebooks = num_codebooks
        self.skip_text_prob = skip_text_prob

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

        if self.causal:
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

        # Use speaker
        if isinstance(self.use_speaker, float):
            use_speaker = random.random() < self.use_speaker
        else:
            use_speaker = self.use_speaker

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
                    speaker=response.name if use_speaker else None,
                    skip_text=random.random() < self.skip_text_prob,
                )

                all_tokens.append(tokens)
                all_labels.append(labels)

            idx += 1

        if use_interactive is False:
            tokens, labels = self.pack_sentences(
                final_text,
                semantics=final_semantic,
                speaker=response.name if use_speaker else None,
            )
            all_tokens.append(tokens)
            all_labels.append(labels)

        tokens = torch.cat(all_tokens, dim=1)
        labels = torch.cat(all_labels, dim=1)

        # Verify that the length is correct
        assert tokens.size(1) == labels.size(1), f"{tokens.size(1)} != {labels.size(1)}"

        data = {"tokens": tokens, "labels": labels}

        return data

    def pack_sentences(
        self,
        sentences: list[str],
        semantics: list,
        speaker: Optional[str] = None,
        skip_text: bool = False,
    ):
        if speaker is None:
            speaker = "assistant"

        cated_sentences = " ".join(sentences)
        if skip_text:
            cated_sentences = "<|skip_text|>"

        final_text = "<|im_start|>user\n" + cated_sentences + "<|im_end|>"
        final_text = final_text + f"<|im_start|>{speaker}\n"

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

        # Pack the tokens and semantics (add <s> and </s> to semantic tokens)
        tokens = (
            encoded
            + [self.semantic_token_id] * semantic_length
            + self.tokenizer.convert_tokens_to_ids(["<|im_end|>"])
        )

        # Codebook bos/padding: 0, eos: 1
        codes = [[CODEBOOK_PAD_TOKEN_ID] * prompt_length for _ in range(num_codebooks)]
        for segment in semantics:
            for book_idx, book in zip(range(num_codebooks), segment):
                for j in book.values:
                    codes[book_idx].append(int(j) + 1)

        for book in codes:
            book.extend([CODEBOOK_PAD_TOKEN_ID] * 1)

        tokens = [tokens] + codes

        tokens = torch.tensor(tokens, dtype=torch.long)
        labels = tokens.clone()

        if skip_text:
            # If text is not provided, the sentence is used for condition only, all labels are -100
            torch.fill_(labels, -100)
            return tokens, labels

        # Mask out the <s> tokens for semantic, predict semantic tokens only
        # Since we don't mask out the input tokens, the language modeling still works
        labels[1:, :prompt_length] = -100

        tokens = tokens[:, :-1]
        labels = labels[:, 1:]

        # Verify the padding is correct, and the last token is eos
        assert (tokens[1:, :prompt_length] == CODEBOOK_PAD_TOKEN_ID).all()
        assert (labels[1:, -1:] == CODEBOOK_PAD_TOKEN_ID).all()

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


class SemanticDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: Union[AutoTextSemanticInstructionDataset, InterleaveDataset],
        val_dataset: Union[AutoTextSemanticInstructionDataset, InterleaveDataset],
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

    ds = AutoTextSemanticInstructionDataset(
        ["data/protos"],
        tokenizer=AutoTokenizer.from_pretrained("fishaudio/fish-speech-1"),
        use_speaker=False,
        interactive_prob=1.0,
        skip_text_prob=0.5,
    )

    for i in ds:
        print(ds.tokenizer.decode(i["tokens"][0], skip_special_tokens=False))
        # i["labels"][0][i["labels"][0] == -100] = 0
        # print(ds.tokenizer.decode(i["labels"][0], skip_special_tokens=False))
        break
