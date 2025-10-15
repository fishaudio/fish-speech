import bisect
import random
from typing import Iterable

from torch.utils.data import Dataset, IterableDataset


class ConcatRepeatDataset(Dataset):
    datasets: list[Dataset]
    cumulative_sizes: list[int]
    repeats: list[int]

    @staticmethod
    def cumsum(sequence, repeats):
        r, s = [], 0
        for dataset, repeat in zip(sequence, repeats):
            l = len(dataset) * repeat
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset], repeats: list[int]):
        super().__init__()

        self.datasets = list(datasets)
        self.repeats = repeats

        assert len(self.datasets) > 0, "datasets should not be an empty iterable"
        assert len(self.datasets) == len(
            repeats
        ), "datasets and repeats should have the same length"

        for d in self.datasets:
            assert not isinstance(
                d, IterableDataset
            ), "ConcatRepeatDataset does not support IterableDataset"

        self.cumulative_sizes = self.cumsum(self.datasets, self.repeats)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)

        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        dataset = self.datasets[dataset_idx]

        return dataset[sample_idx % len(dataset)]
