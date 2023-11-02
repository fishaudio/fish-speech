from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset


class VQGANDataset(Dataset):
    def __init__(
        self,
        filelist: str,
        sample_rate: int = 32000,
    ):
        super().__init__()

        filelist = Path(filelist)
        root = filelist.parent

        self.files = [root / line.strip() for line in filelist.read_text().splitlines()]
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        audio, _ = librosa.load(file, sr=self.sample_rate, mono=True)
        features = np.load(file.with_suffix(".npy"))  # (T, 1024)

        return {
            "audio": torch.from_numpy(audio),
            "features": torch.from_numpy(features),
        }


@dataclass
class VQGANCollator:
    hop_length: int = 640

    def __call__(self, batch):
        audio_lengths = torch.tensor([len(x["audio"]) for x in batch])
        feature_lengths = torch.tensor([len(x["features"]) for x in batch])

        audio_maxlen = audio_lengths.max()
        feature_maxlen = feature_lengths.max()

        if audio_maxlen % self.hop_length != 0:
            audio_maxlen += self.hop_length - (audio_maxlen % self.hop_length)

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
        hop_length: int = 640,
        num_workers: int = 4,
    ):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.hop_length = hop_length
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=VQGANCollator(self.hop_length),
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=VQGANCollator(self.hop_length),
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    from torch.utils.data import DataLoader

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
