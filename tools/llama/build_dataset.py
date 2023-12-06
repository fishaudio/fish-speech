import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union

import numpy as np
from loguru import logger
from tqdm import tqdm

from fish_speech.text import g2p
from fish_speech.utils.file import AUDIO_EXTENSIONS, list_files

# Define datasets
DATASETS = [
    ("data/StarRail/Chinese", "StarRail", ["ZH", "EN"], ".lab", 1),
    ("data/StarRail/English", "StarRail", ["EN"], ".lab", 1),
    ("data/StarRail/Japanese", "StarRail", ["JP", "EN"], ".lab", 1),
    ("data/Genshin/Chinese", "Genshin", ["ZH", "EN"], ".lab", 1),
    ("data/Genshin/English", "Genshin", ["EN"], ".lab", 1),
    ("data/Genshin/Japanese", "Genshin", ["JP", "EN"], ".lab", 1),
    ("data/LibriTTS_R", "LibriTTS_R", ["EN"], ".normalized.txt", 2),
    ("data/WenetSpeech", "WenetSpeech", ["ZH", "EN"], ".txt", 1),
]


@dataclass
class Sentence:
    text: str
    phones: list[str]
    # Support multiple codebooks
    semantics: Union[list[int], list[list[int]]]


@dataclass
class PackedSentences:
    source: str
    name: str
    languages: list[str]
    sentences: list[Sentence]


dataset_fp = open("data/quantized-dataset-1205.json", "w")

for root, source, languages, extension, parent_level in DATASETS:
    # Load the files
    exts = extension.split(".")
    files = list_files(root, AUDIO_EXTENSIONS, recursive=True)
    logger.info(f"Found {len(files)} files in {root}")

    grouped_files = defaultdict(list)
    for file in files:
        if parent_level == 1:
            p = file.parent.name
        elif parent_level == 2:
            p = file.parent.parent.name
        else:
            raise ValueError(f"Invalid parent level {parent_level}")

        grouped_files[p].append(file)

    for name, subset in tqdm(grouped_files.items()):
        # Parse the files
        sentences = []
        for file in subset:
            np_file = file.with_suffix(".npy")
            txt_file = file.with_suffix(extension)
            if np_file.exists() is False or txt_file.exists() is False:
                continue

            with open(txt_file, "r") as f:
                text = f.read().strip()

            # Simple cleaning: replace { xxx } and < xxx > with space
            text = re.sub(r"\{.*?\}", " ", text)
            text = re.sub(r"<.*?>", " ", text)
            text = re.sub(r"\s+", " ", text)

            try:
                phones = [v for _, v in g2p(text, order=languages)]
                semantics = np.load(np_file)
            except Exception as e:
                logger.error(f"Failed to parse {file}: {e}")
                continue

            if isinstance(semantics, np.ndarray):
                semantics = semantics.tolist()

            sentences.append(
                Sentence(
                    text=text,
                    phones=phones,
                    semantics=semantics,
                )
            )

        # Pack the sentences
        packed_sentences = PackedSentences(
            source=source,
            name=name,
            languages=languages,
            sentences=sentences,
        )

        dataset_fp.write(
            json.dumps(asdict(packed_sentences), ensure_ascii=False) + "\n"
        )


dataset_fp.close()
