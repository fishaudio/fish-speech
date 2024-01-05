import re
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import click
import numpy as np
import yaml
from loguru import logger
from tqdm import tqdm

from fish_speech.datasets.protos.text_data_pb2 import Semantics, Sentence, TextData
from fish_speech.datasets.protos.text_data_stream import pack_pb_stream
from fish_speech.text import g2p
from fish_speech.utils.file import AUDIO_EXTENSIONS, list_files, load_filelist


def task_generator_yaml(config):
    with open(config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for row in config["datasets"]:
        root, source, languages, extension, parent_level = (
            row["root"],
            row["source"],
            row["languages"],
            row["extension"],
            row["group_parent_level"],
        )

        # Load the files
        files = list_files(root, AUDIO_EXTENSIONS, recursive=True, sort=True)

        grouped_files = defaultdict(list)
        for file in files:
            if parent_level == 1:
                p = file.parent.name
            elif parent_level == 2:
                p = file.parent.parent.name
            else:
                raise ValueError(f"Invalid parent level {parent_level}")
            grouped_files[p].append(file)

        logger.info(f"Found {len(grouped_files)} groups in {root}")
        for name, subset in grouped_files.items():
            yield name, subset, source, languages, extension


def task_generator_filelist(filelist):
    grouped_files = defaultdict(list)
    for filename, speaker, languages, text in load_filelist(filelist):
        grouped_files[speaker].append((Path(filename), text, languages))

    logger.info(f"Found {len(grouped_files)} groups in {filelist}")
    for speaker, values in grouped_files.items():
        yield speaker, values, "filelist", languages, None


def run_task(task):
    name, subset, source, languages, extension = task

    # Parse the files
    sentences = []
    for file in subset:
        if isinstance(file, tuple):
            file, text, languages = file
        else:
            text = None

        np_file = file.with_suffix(".npy")
        if np_file.exists() is False:
            logger.warning(f"Can't find {np_file}")
            continue

        if text is None:
            txt_file = file.with_suffix(extension)

            if txt_file.exists() is False:
                logger.warning(f"Can't find {txt_file}")
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
                semantics=[Semantics(values=s) for s in semantics],
            )
        )

    # Pack the sentences
    return pack_pb_stream(
        TextData(
            source=source,
            name=name,
            languages=languages,
            sentences=sentences,
        )
    )


@click.command()
@click.option(
    "--config", type=click.Path(), default="fish_speech/configs/data/finetune.yaml"
)
@click.option("--output", type=click.Path(), default="data/quantized-dataset-ft.protos")
@click.option("--filelist", type=click.Path(), default=None)
@click.option("--num-workers", type=int, default=16)
def main(config, output, filelist, num_workers):
    dataset_fp = open(output, "wb")
    generator_fn = (
        task_generator_yaml(config)
        if filelist is None
        else task_generator_filelist(filelist)
    )

    with Pool(num_workers) as p:
        for result in tqdm(p.imap_unordered(run_task, generator_fn)):
            dataset_fp.write(result)

    dataset_fp.close()


if __name__ == "__main__":
    main()
