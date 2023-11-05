from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from fish_speech.utils import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)


class VQGANDataset(Dataset):
    def __init__(
        self,
        filelist: str,
        sample_rate: int = 32000,
        hop_length: int = 640,
        slice_frames: Optional[int] = None,
    ):
        super().__init__()

        filelist = Path(filelist)
        root = filelist.parent

        self.files = [
            root / line.strip()
            for line in filelist.read_text().splitlines()
            if ("Genshin" in line or "StarRail" in line)
        ]
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.slice_frames = slice_frames

    def __len__(self):
        return len(self.files)

    def get_item(self, idx):
        file = self.files[idx]

        audio, _ = librosa.load(file, sr=self.sample_rate, mono=True)
        features = np.load(file.with_suffix(".npy"))  # (T, 1024)

        # Slice audio and features
        if self.slice_frames is not None and features.shape[0] > self.slice_frames:
            start = np.random.randint(0, features.shape[0] - self.slice_frames)
            features = features[start : start + self.slice_frames]
            audio = audio[
                start * self.hop_length : (start + self.slice_frames) * self.hop_length
            ]

        if features.shape[0] % 2 != 0:
            features = features[:-1]

        if len(audio) > len(features) * self.hop_length:
            audio = audio[: features.shape[0] * self.hop_length]

        if len(audio) < len(features) * self.hop_length:
            audio = np.pad(
                audio,
                (0, len(features) * self.hop_length - len(audio)),
                mode="constant",
                constant_values=0,
            )

        return {
            "audio": torch.from_numpy(audio),
            "features": torch.from_numpy(features),
        }

    def __getitem__(self, idx):
        try:
            return self.get_item(idx)
        except Exception as e:
            logger.error(f"Error loading {self.files[idx]}: {e}")
            return None


@dataclass
class VQGANCollator:
    def __call__(self, batch):
        batch = [x for x in batch if x is not None]

        audio_lengths = torch.tensor([len(x["audio"]) for x in batch])
        feature_lengths = torch.tensor([len(x["features"]) for x in batch])

        audio_maxlen = audio_lengths.max()
        feature_maxlen = feature_lengths.max()

        audios, features = [], []
        for x in batch:
            audios.append(
                torch.nn.functional.pad(x["audio"], (0, audio_maxlen - len(x["audio"])))
            )
            features.append(
                torch.nn.functional.pad(
                    x["features"], (0, 0, 0, feature_maxlen - len(x["features"]))
                )
            )

        return {
            "audios": torch.stack(audios),
            "features": torch.stack(features),
            "audio_lengths": audio_lengths,
            "feature_lengths": feature_lengths,
        }


class VQGANDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: VQGANDataset,
        val_dataset: VQGANDataset,
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

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=VQGANCollator(),
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=VQGANCollator(),
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    dataset = VQGANDataset("data/LibriTTS_R/vq_train_filelist.txt")
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=False, collate_fn=VQGANCollator()
    )

    for batch in dataloader:
        print(batch["audios"].shape)
        print(batch["features"].shape)
        print(batch["audio_lengths"])
        print(batch["feature_lengths"])
        break
