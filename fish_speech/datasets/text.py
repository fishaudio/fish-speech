import random
from dataclasses import dataclass
from itertools import chain
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

from fish_speech.datasets.protos.text_data_pb2 import SampleDataRequest, SampledData
from fish_speech.datasets.protos.text_data_pb2_grpc import DataServiceStub
from fish_speech.datasets.protos.text_data_stream import read_pb_stream
from fish_speech.text.parser import clean_text
from fish_speech.text.symbols import pad as pad_symbol
from fish_speech.text.symbols import pu_symbols
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

            # 30% modeling phones
            if random.random() < 0.3:
                text = " ".join(
                    [
                        (f"<p:{i}>" if i not in pu_symbols and i != pad_symbol else i)
                        for i in text
                    ]
                )

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
    3. Mix text and phones

    For interactive mode, we use the following format (multiple sequences):
    <s> [INST] [SPK: speaker] text [/INST] ... [INST] text [/INST] </s>

    For non-interactive mode, we use the following format (one long sequence):
    <s> [INST] text [/INST] ... </s>
    """

    def __init__(
        self,
        server: str = "localhost:50051",
        seed: int = 42,
        phones_prob: float = 0.3,
        repetition_prob: float = 0.0,
        interactive_prob: float = 0.5,
        max_length: int = 1024,
        tokenizer: AutoTokenizer = None,
        use_speaker: bool = True,
        use_data_server: bool = True,
        proto_files: str = "data",
        causual: bool = True,
        mix_text_phone_prob: float = 0.5,
        use_negative_samples: bool = False,
        num_codebooks: Optional[int] = None,
        use_delay_pattern: bool = True,
    ):
        """
        Args:
            server: gRPC server address
            seed: random seed
            phones_prob: probability to use phones
            repetition_prob: probability to repeat the same sentence
            interactive_prob: probability to use interactive mode
            max_length: max length of the text
            tokenizer: tokenizer
            use_speaker: include speaker information in the prompt
            use_data_server: use data server or local data
            proto_files: proto buf files if using local data
            causual: use causual sampling when using local data, disable will lead to random sampling
            mix_text_phone_prob: probability to mix text and phones, if this is 0, then it will be pure text or pure phones
            use_negative_samples: generate negative samples
            num_codebooks: number of codebooks, if None, it will be automatically detected
            use_delay_pattern: use delay pattern for codebooks
        """

        super().__init__()

        assert 0 <= phones_prob <= 1, "phones_prob must be in [0, 1]"
        assert 0 <= repetition_prob <= 1, "repetition_prob must be in [0, 1]"
        assert 0 <= interactive_prob <= 1, "interactive_prob must be in [0, 1]"
        assert 0 <= mix_text_phone_prob <= 1, "mix_text_phone_prob must be in [0, 1]"

        self.seed = seed
        self.phones_prob = phones_prob
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.repetition_prob = repetition_prob
        self.interactive_prob = interactive_prob
        self.use_speaker = use_speaker
        self.use_data_server = use_data_server
        self.proto_files = proto_files
        self.causual = causual
        self.mix_text_phone_prob = mix_text_phone_prob
        self.use_negative_samples = use_negative_samples
        self.num_codebooks = num_codebooks
        self.use_delay_pattern = use_delay_pattern

        if use_data_server is True:
            self.channel = grpc.insecure_channel(server)
            self.stub = DataServiceStub(self.channel)
        else:
            self.init_mock_data_server()

    def init_mock_data_server(self):
        self.groups = []
        count = 0
        for filename in self.proto_files:
            with open(filename, "rb") as f:
                for text_data in read_pb_stream(f):
                    self.groups.append(text_data)
                    count += 1

                    if count % 1000 == 0:
                        log.info(f"Read {count} groups of data")

        log.info(f"Read total {count} groups of data")

        # Shuffle the lines
        Random(self.seed).shuffle(self.groups)

    def __iter__(self):
        while True:
            yield self.augment()

    def tokenize_sentence(self, sentence: str, phones: list[str], mode: str = "sample"):
        if (
            mode == "sample" and (random.random() < self.phones_prob)
        ) or mode == "phones":
            sentence = " ".join(
                [
                    (f"<p:{i}>" if i not in pu_symbols and i != pad_symbol else i)
                    for i in phones
                ]
            )
        else:
            sentence = clean_text(sentence)

        tokens = self.tokenizer.encode(
            f"{sentence}",
            max_length=10**6,
            add_special_tokens=False,
            truncation=False,
        )
        return sentence, len(tokens)

    def sample_data(self):
        # Shuffle unique lines, estimate that each sample is at least 20 tokens
        num_samples = self.max_length // 20

        if self.use_data_server:
            request = SampleDataRequest(num_samples=num_samples)
            return self.stub.SampleData(request)

        # choice group based on their number of samples
        group = random.choices(
            self.groups, weights=[len(i.sentences) for i in self.groups], k=1
        )[0]

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
        # 50% to pure text or pure phones
        mode = "sample"
        if random.random() > self.mix_text_phone_prob:
            mode = random.choices(
                ["text", "phones"],
                weights=[1 - self.phones_prob, self.phones_prob],
                k=1,
            )[0]

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

        final_text, final_semantic = [], []
        response = self.sample_data()
        if len(response.samples) == 0:
            # Invalid group
            return None

        samples = list(response.samples)
        idx = 0
        use_interactive = random.random() < self.interactive_prob

        all_tokens, all_labels = [], []
        while remaining_tokens > 0 and len(samples) > 0:
            if random.random() < self.repetition_prob:
                # Repeat the same sentence
                sentence = samples[-1]
            else:
                sentence = samples.pop()

            text, length = self.tokenize_sentence(
                sentence.text, sentence.phones, mode=mode
            )
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
                speaker=None if self.use_speaker else response.name,
                add_bos=True,
            )
            all_tokens.append(tokens)
            all_labels.append(labels)

        tokens = torch.cat(all_tokens, dim=1)
        labels = torch.cat(all_labels, dim=1)

        # Verify that the length is correct
        assert tokens.size(1) == labels.size(1), f"{tokens.size(1)} != {labels.size(1)}"

        # Verify only one <s> token
        assert (tokens[:, 0] == self.tokenizer.bos_token_id).sum() == 1

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

        final_text = "[INST] " + " ".join(sentences) + " [/INST]"
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
        pad_token_length = semantic_length + (
            num_codebooks - 1 if self.use_delay_pattern else 0
        )
        tokens = (
            encoded
            + [self.tokenizer.pad_token_id] * pad_token_length
            + [self.tokenizer.eos_token_id]
        )

        if add_bos:
            tokens = [self.tokenizer.bos_token_id] + tokens

        # Codebook bos/padding: 0, eos: 1
        # Implement delay pattern
        codes = [
            [CODEBOOK_PAD_TOKEN_ID]
            * (prompt_length + bos_bias + (i if self.use_delay_pattern else 0))
            for i in range(num_codebooks)
        ]
        for segment in semantics:
            for book_idx, book in zip(range(num_codebooks), segment):
                for j in book.values:
                    codes[book_idx].append(int(j) + 2)

        for idx, book in enumerate(codes):
            if self.use_delay_pattern:
                book.extend([CODEBOOK_PAD_TOKEN_ID] * (len(codes) - idx - 1))
            book.append(CODEBOOK_EOS_TOKEN_ID)

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
        assert (labels[1:, -1] == CODEBOOK_EOS_TOKEN_ID).all()

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
        for example in examples:
            _tokens = example[tokens_key][:, : self.max_length]
            _labels = example[labels_key][:, : self.max_length]
            _attention_mask = torch.ones((self.max_length,), dtype=torch.bool)
            tokens_length = _tokens.size(1)
            _attention_mask[:tokens_length] = False

            assert tokens_length == _labels.size(
                1
            ), f"{tokens_length} != {_labels.size(1)}"

            if tokens_length < self.max_length:
                _tokens = F.pad(
                    _tokens,
                    (0, self.max_length - tokens_length),
                    value=self.tokenizer.eos_token_id,
                )
                _tokens[1:, tokens_length:] = CODEBOOK_EOS_TOKEN_ID
                _labels = F.pad(
                    _labels, (0, self.max_length - _labels.size(1)), value=-100
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
        tokenizer=AutoTokenizer.from_pretrained("fishaudio/speech-lm-v1"),
        use_speaker=True,
        interactive_prob=1.0,
        phones_prob=1.0,
        use_negative_samples=False,
        num_codebooks=4,
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
