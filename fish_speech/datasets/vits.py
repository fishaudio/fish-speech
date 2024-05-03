from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
import torch.distributed as dist
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from fish_speech.utils import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)


class VITSDataset(Dataset):
    def __init__(
        self,
        filelist: str,
        tokenizer: AutoTokenizer,
        sample_rate: int = 44100,
        hop_length: int = 512,
        min_duration: float = 1.5,
        max_duration: float = 30.0,
        suffix: str = ".lab",
    ):
        super().__init__()

        filelist = Path(filelist)
        root = filelist.parent

        self.files = []
        for line in filelist.read_text(encoding="utf-8").splitlines():
            path = root / line
            self.files.append(path)

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.tokenizer = tokenizer
        self.suffix = suffix

    def __len__(self):
        return len(self.files)

    def get_item(self, idx):
        audio_file = self.files[idx]
        text_file = audio_file.with_suffix(self.suffix)

        if text_file.exists() is False or audio_file.exists() is False:
            return None

        audio, _ = librosa.load(audio_file, sr=self.sample_rate, mono=True)
        duration = len(audio) / self.sample_rate

        if (
            len(audio) == 0
            or duration < self.min_duration
            or duration > self.max_duration
        ):
            return None

        max_value = np.abs(audio).max()
        if max_value > 1.0:
            audio = audio / max_value

        text = text_file.read_text(encoding="utf-8")
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.squeeze(0)

        return {
            "audio": torch.from_numpy(audio),
            "text": input_ids,
        }

    def __getitem__(self, idx):
        try:
            return self.get_item(idx)
        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(f"Error loading {self.files[idx]}: {e}")
            return None


@dataclass
class VITSCollator:
    tokenizer: AutoTokenizer

    def __call__(self, batch):
        batch = [x for x in batch if x is not None]

        audio_lengths = torch.tensor([len(x["audio"]) for x in batch])
        audio_maxlen = audio_lengths.max()

        text_lengths = torch.tensor([len(x["text"]) for x in batch])
        text_maxlen = text_lengths.max()

        # Rounds up to nearest multiple of 2 (audio_lengths)
        audios = []
        texts = []
        for x in batch:
            audios.append(
                torch.nn.functional.pad(x["audio"], (0, audio_maxlen - len(x["audio"])))
            )

            texts.append(
                torch.nn.functional.pad(
                    x["text"],
                    (0, text_maxlen - len(x["text"])),
                    value=self.tokenizer.eos_token_id,
                )
            )

        return {
            "audios": torch.stack(audios),
            "audio_lengths": audio_lengths,
            "texts": torch.stack(texts),
            "text_lengths": text_lengths,
        }


class VITSDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: VITSDataset,
        val_dataset: VITSDataset,
        tokenizer: AutoTokenizer,
        batch_size: int = 32,
        num_workers: int = 4,
        val_batch_size: Optional[int] = None,
    ):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=VITSCollator(self.tokenizer),
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            collate_fn=VITSCollator(self.tokenizer),
            num_workers=self.num_workers,
            persistent_workers=True,
        )


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("fishaudio/fish-speech-1")
    dataset = VITSDataset(
        "data/source/Genshin/filelist.train.txt", tokenizer=tokenizer, suffix=".lab"
    )
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=False, collate_fn=VITSCollator(tokenizer)
    )

    for batch in dataloader:
        print(batch["audios"].shape)
        print(batch["audio_lengths"])
        print(batch["texts"].shape)
        print(batch["text_lengths"])
        break
