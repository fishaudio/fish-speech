from pathlib import Path

import click
import numpy as np
import torch
from einops import rearrange, repeat
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from vector_quantize_pytorch.vector_quantize_pytorch import (
    batched_bincount,
    batched_sample_vectors,
    cdist,
)


class KMeansDataset(Dataset):
    def __init__(self, filelist):
        root = Path(filelist).parent

        with open(filelist) as f:
            self.files = f.read().splitlines()

        self.files = [root / file.strip() for file in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        feature = np.load(file.with_suffix(".npy"))
        return torch.from_numpy(feature).float()

    @staticmethod
    def collate_fn(features):
        features = torch.concat(features, dim=0)
        return features


@click.command()
@click.option(
    "--filelist",
    type=click.Path(exists=True, path_type=Path),
    default="data/test.filelist",
)
@click.option("--output", type=click.Path(path_type=Path), default="kmeans.pt")
@click.option("--num-clusters", type=int, default=2048)
@click.option("--epochs", type=int, default=10)
def main(filelist: Path, output: Path, num_clusters: int, epochs: int):
    loader = DataLoader(
        KMeansDataset(filelist),
        batch_size=1024,
        shuffle=True,
        num_workers=2,
        collate_fn=KMeansDataset.collate_fn,
    )

    means = None
    for _ in tqdm(range(epochs), desc="Epochs", position=0):
        total_bins = torch.zeros(1, num_clusters, dtype=torch.int64, device="cuda")

        for samples in tqdm(loader, desc="Batches", position=1):
            samples = samples.cuda()[None]
            num_codebooks, dim, dtype = (
                samples.shape[0],
                samples.shape[-1],
                samples.dtype,
            )

            if means is None:
                means = batched_sample_vectors(samples, num_clusters)

            dists = -cdist(samples, means)

            buckets = torch.argmax(dists, dim=-1)
            bins = batched_bincount(buckets, minlength=num_clusters)

            zero_mask = bins == 0
            bins_min_clamped = bins.masked_fill(zero_mask, 1)

            new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype=dtype)

            new_means.scatter_add_(1, repeat(buckets, "h n -> h n d", d=dim), samples)
            new_means = new_means / rearrange(bins_min_clamped, "... -> ... 1")

            means = torch.where(rearrange(zero_mask, "... -> ... 1"), means, new_means)

            total_bins += bins

    torch.save(
        {
            "centroids": means,
            "bins": bins,
        },
        output,
    )
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
