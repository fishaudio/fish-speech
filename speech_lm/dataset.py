import random
from functools import partial

from datasets import IterableDataset, interleave_datasets, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed import get_rank, get_world_size, is_initialized


def encode(examples, tokenizer, max_length=512):
    # Random choice a 512 token window for each example
    texts = []
    for text in examples["text"]:
        if len(text) <= max_length:
            texts.append(text)
        else:
            start = random.randint(0, len(text) - max_length)
            texts.append(text[start : start + max_length])

    data = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    data["labels"] = data["input_ids"].clone()
    data["labels"][data["attention_mask"] == 0] = -100
    print(data["input_ids"].shape)
    return data


def build_dataset(tokenizer, max_length=512):
    en_dataset = load_dataset("uonlp/CulturaX", "en", split="train", streaming=True)
    ja_dataset = load_dataset("uonlp/CulturaX", "ja", split="train", streaming=True)
    zh_dataset = load_dataset("uonlp/CulturaX", "zh", split="train", streaming=True)

    multilingual_dataset: IterableDataset = interleave_datasets(
        [en_dataset, ja_dataset, zh_dataset], probabilities=[0.4, 0.3, 0.3], seed=42
    )

    # DDP
    if is_initialized():
        multilingual_dataset = split_dataset_by_node(
            multilingual_dataset,
            rank=get_rank(),
            world_size=get_world_size(),
        )

    multilingual_dataset = multilingual_dataset.shuffle(seed=42, buffer_size=10000)

    multilingual_dataset = multilingual_dataset.map(
        partial(encode, tokenizer=tokenizer, max_length=max_length),
        batched=True,
        remove_columns=multilingual_dataset.column_names,
    )

    return multilingual_dataset


if __name__ == "__main__":
    dataset = build_dataset()
    print(list(dataset.take(16)))
