import itertools
from logging import Logger
from pathlib import Path
from typing import Literal, Callable, Any

import h5py
import numpy as np
from torch.utils.data.sampler import Sampler

from ..basedataset import BaseDataset
from utils import get_path

class ACDCDataset(BaseDataset):
    SAMPLES_DIR = "data"
    TRAIN_SPLIT_FILE = "train_slices.list"
    VALID_SPLIT_FILE = "val.list"

    @staticmethod
    def find_samples(
        data_path: Path | str, require_label: bool = True
    ) -> list[dict]:
        data_path = get_path(data_path)
        samples_dir = data_path / ACDCDataset.SAMPLES_DIR
        samples_list = []
        for sample in samples_dir.glob(".h5"):
            if not sample.is_file():
                continue
            h5f = h5py.File(sample, "r")

            if "image" not in h5f:
                continue

            labeled = "label" in h5f

            if require_label and not labeled:
                continue

            samples_list.append(
                {
                    "id": sample.stem,
                    "path": sample.resolve(),
                    "labeled": labeled,
                }
            )

        return samples_list

    def __init__(
        self,
        data_path: Path | str,
        split: Literal["train", "valid"] = "train",
        num: int | None = None,
        normalize: Callable | None = None,
        transform: Callable | None = None,
        logger: Logger | None = None,
    ):
        self.data_path = get_path(data_path)
        self.split = split
        self.num = num
        self.normalize = normalize
        self.transform = transform
        self.logger = logger

        self.samples_list = []

        self._register_samples()

    def _register_samples(self):
        if self.split == "train":
            with open(self.data_path / "train_slices.list", "r") as f:
                self.samples_list = f.readlines()
            self.samples_list = [
                item.replace("\n", "") for item in self.samples_list
            ]

        elif self.split == "valid":
            with open(self.data_path / "val.list", "r") as f:
                self.samples_list = f.readlines()
            self.samples_list = [
                item.replace("\n", "") for item in self.samples_list
            ]

        if self.num is not None and self.split == "train":
            self.samples_list = self.samples_list[: self.num]

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, index):
        return self.get_sample(index)

    def get_sample(self, index: int, normalize: bool = True) -> Any:
        case = self.samples_list[index]

        if self.split == "train":
            h5f = h5py.File(
                self.data_path / "data/slices/{}.h5".format(case), "r"
            )
        else:
            h5f = h5py.File(self.data_path / "data/{}.h5".format(case), "r")

        if "image" in h5f:
            image = h5f["image"][:]
        else:
            raise RuntimeError(f"Case {case}.h5 does not have image field")
        if "label" in h5f:
            label = h5f["label"][:]
        else:
            raise RuntimeError(f"Case {case}.h5 does not have label field")

        if self.transform:
            image, label = self.transform(image, label)

        if self.normalize and normalize:
            image, label = self.normalize(image, label)

        sample = {"image": image, "label": label}
        sample["case_name"] = self.samples_list[index].strip("\n")

        return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(
        self,
        primary_indices,
        secondary_indices,
        batch_size,
        secondary_batch_size,
    ):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
