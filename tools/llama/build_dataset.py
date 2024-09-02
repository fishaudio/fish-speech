import itertools
import os
import re
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import click
import numpy as np
from loguru import logger
from tqdm import tqdm

from fish_speech.datasets.protos.text_data_pb2 import Semantics, Sentence, TextData
from fish_speech.datasets.protos.text_data_stream import pack_pb_stream
from tools.file import load_filelist

# To avoid CPU overload
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def task_generator_folder(root: Path, text_extension: str):
    files = list(tqdm(Path(root).rglob("*.npy"), desc=f"Loading {root}"))
    files = sorted(files)

    grouped_files = defaultdict(list)
    for file in tqdm(files, desc=f"Grouping {root}"):
        p = str(file.parent)
        speaker = file.parent.name

        try:
            if isinstance(text_extension, str):
                texts = [file.with_suffix(text_extension).read_text(encoding="utf-8")]
            else:
                texts = [
                    file.with_suffix(ext).read_text(encoding="utf-8")
                    for ext in text_extension
                ]
        except Exception as e:
            logger.error(f"Failed to read text {file}: {e}")
            continue

        grouped_files[p].append((speaker, file, texts))

    logger.info(
        f"Found {len(grouped_files)} groups in {root}, {list(grouped_files.keys())[:5]}..."
    )

    for i in grouped_files.values():
        subset = [(f, t) for _, f, t in i]
        yield i[0][0], subset, "folder"


def task_generator_filelist(filelist):
    grouped_files = defaultdict(list)
    for filename, speaker, _, text in load_filelist(filelist):
        grouped_files[speaker].append((Path(filename), [text]))

    logger.info(f"Found {len(grouped_files)} groups in {filelist}")
    for speaker, values in grouped_files.items():
        yield speaker, values, "filelist"


def run_task(task):
    name, subset, source = task

    # Parse the files
    sentences = []
    for file, texts in subset:
        np_file = file.with_suffix(".npy")
        if np_file.exists() is False:
            logger.warning(f"Can't find {np_file}")
            continue

        new_texts = []

        for text in texts:
            # Simple cleaning: replace { xxx } and < xxx > with space
            text = re.sub(r"\{.*?\}", " ", text)
            text = re.sub(r"<.*?>", " ", text)
            text = re.sub(r"\s+", " ", text)
            new_texts.append(text)

        try:
            semantics = np.load(np_file)
        except Exception as e:
            logger.error(f"Failed to parse {file}: {e}")
            continue

        if isinstance(semantics, np.ndarray):
            semantics = semantics.tolist()

        sentences.append(
            Sentence(
                texts=new_texts,
                semantics=[Semantics(values=s) for s in semantics],
            )
        )

    # Pack the sentences
    return pack_pb_stream(
        TextData(
            source=source,
            name=name,
            sentences=sentences,
        )
    )


@click.command()
@click.option(
    "--input",
    type=click.Path(path_type=Path),
    required=True,
    help="A folder containing the dataset or a filelist",
    multiple=True,
)
@click.option(
    "--output", type=click.Path(path_type=Path), default="data/quantized-dataset-ft"
)
@click.option("--num-workers", type=int, default=16)
@click.option("--text-extension", type=str, default=[".txt"], multiple=True)
@click.option(
    "--shard-size", type=int, default=10, help="The maximum size of each shard in mb"
)
def main(input, output, num_workers, text_extension, shard_size):
    generator_fns = []

    for f in input:
        assert f.exists(), f"{f} not found"

        if f.is_dir():
            generator_fn = task_generator_folder(f, text_extension)
        else:
            generator_fn = task_generator_filelist(f)

        generator_fns.append(generator_fn)

    generator_fn = itertools.chain(*generator_fns)
    output.mkdir(parents=True, exist_ok=True)

    dataset_fp = None
    tar_idx = 0
    written_size = 0

    with Pool(num_workers) as p:
        for result in tqdm(p.imap_unordered(run_task, generator_fn)):
            if dataset_fp is None:
                dataset_fp = open(Path(output) / f"{tar_idx:08d}.protos", "wb")

            dataset_fp.write(result)
            written_size += len(result)

            if written_size > shard_size * 1024 * 1024:
                logger.info(f"Finished writing {tar_idx} shards to {output}")
                dataset_fp.close()
                dataset_fp = None
                written_size = 0
                tar_idx += 1

    if dataset_fp is not None:
        dataset_fp.close()

    logger.info(f"Finished writing {tar_idx + 1} shards to {output}")


if __name__ == "__main__":
    main()
