import random
from transformers import AutoTokenizer
from datasets import load_dataset, interleave_datasets, IterableDataset
from functools import lru_cache
from torch.utils.data import DataLoader
from datasets.distributed import split_dataset_by_node
from torch.distributed import get_rank, get_world_size, is_initialized


@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained("fishaudio/speech-lm-300m", revision="init")

def encode(examples):
    # Random choice a 512 token window for each example
    texts = []
    for text in examples["text"]:
        if len(text) <= 512:
            texts.append(text)
        else:
            start = random.randint(0, len(text) - 512)
            texts.append(text[start : start + 512])
    
    data = get_tokenizer()(
        texts, 
        truncation=True, 
        padding="max_length",
        max_length=512,
    )
    data["labels"] = data["input_ids"].copy()
    data["labels"][data["attention_mask"] == 0] = -100

    return data


def build_dataset():
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
            num_replicas=get_world_size(),
        )

    multilingual_dataset = multilingual_dataset.shuffle(seed=42, buffer_size=10000)

    multilingual_dataset = multilingual_dataset.map(
        encode, batched=True, remove_columns=multilingual_dataset.column_names
    )

    return multilingual_dataset


if __name__ == "__main__":
    dataset = build_dataset()
    print(list(dataset.take(16)))
