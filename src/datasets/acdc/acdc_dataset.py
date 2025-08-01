import itertools
from logging import Logger
from pathlib import Path
from typing import Literal, Callable, Any

import h5py
import numpy as np
import torch
from torch.utils.data.sampler import Sampler
import torchvision.transforms.functional as F

import pandas as pd

from ..basedataset import BaseDataset
from utils import get_path
from transforms.joint_transform import JointResize


class ACDCDataset(BaseDataset):
    CLASSES = {0: "bg", 1: "RV", 2: "Myo", 3: "LV"}
    RAW_DIR = "ACDC_raw"

    PROCESSED_DIR = "ACDC"
    SAMPLES_DIR = f"{PROCESSED_DIR}/data"
    TRAIN_SPLIT_FILE = f"{PROCESSED_DIR}/train_slices.list"
    VALID_SPLIT_FILE = f"{PROCESSED_DIR}/val.list"
    TEST_SPLIT_FILE = f"{PROCESSED_DIR}/test.list"
    NUM_CLASSES = 3
    Z_SPACING = 1
    RAW_SPACING = [10.0, 1.4843800067901611, 1.4843800067901611]

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
        split: Literal["train", "valid", "test"] = "train",
        num: int | None = None,
        normalize: Callable | None = None,
        transform: Callable | None = None,
        logger: Logger | None = None,
        image_channels: int = 3,
        image_size: int | tuple[int, int] | None = None,
    ):
        self.data_path = get_path(data_path)
        self.split = split
        self.num = num
        self.normalize = normalize
        self.transform = transform
        self.logger = logger
        self.image_channels = image_channels
        self.image_size = image_size

        self.final_transform = self._get_final_transform()

        self.samples_list = []

        self._register_samples()

    def _get_final_transform(self):
        if self.image_size is None:
            return None
        else:
            return JointResize(self.image_size)

    def _register_samples(self):
        if self.split == "train":
            with open(self.data_path / ACDCDataset.TRAIN_SPLIT_FILE, "r") as f:
                self.samples_list = f.readlines()
            self.samples_list = [
                item.replace("\n", "") for item in self.samples_list
            ]

        elif self.split == "valid":
            with open(self.data_path / ACDCDataset.VALID_SPLIT_FILE, "r") as f:
                self.samples_list = f.readlines()
            self.samples_list = [
                item.replace("\n", "") for item in self.samples_list
            ]
        elif self.split == "test":
            with open(self.data_path / ACDCDataset.TEST_SPLIT_FILE, "r") as f:
                self.samples_list = f.readlines()
            self.samples_list = [
                item.replace("\n", "") for item in self.samples_list
            ]

        raw_spacing_path = (
            self.data_path / ACDCDataset.PROCESSED_DIR / "raw_spacing.csv"
        )
        if raw_spacing_path.is_file():
            self.raw_spacing = pd.read_csv(raw_spacing_path, index_col=0)
        else:
            self.raw_spacing = None

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
                self.data_path / f"{ACDCDataset.SAMPLES_DIR}/slices/{case}.h5",
                "r",
            )
        else:
            h5f = h5py.File(
                self.data_path / f"{ACDCDataset.SAMPLES_DIR}/{case}.h5", "r"
            )

        if "image" in h5f:
            image_ds = h5f["image"]
            assert isinstance(image_ds, h5py.Dataset)
            image = image_ds[:]
        else:
            raise RuntimeError(f"Case {case}.h5 does not have image field")
        if "label" in h5f:
            label_ds = h5f["label"]
            assert isinstance(label_ds, h5py.Dataset)
            label = label_ds[:]
        else:
            raise RuntimeError(f"Case {case}.h5 does not have label field")

        image = torch.from_numpy(image).unsqueeze(0).float()
        label = torch.from_numpy(label).unsqueeze(0).long()

        if self.split == "train":
            image = image.repeat(self.image_channels, 1, 1)
        else:
            image = image.repeat(self.image_channels, 1, 1, 1)

        data: dict = {"image": image, "label": label}

        if self.transform:
            data = self.transform(data)

        if self.final_transform:
            data = self.final_transform(data)

        if self.normalize and normalize:
            data = self.normalize(data)

        data["label"] = data["label"].squeeze(0)

        data["case_name"] = self.samples_list[index].strip("\n")

        patient_frame_id = "_".join(self.samples_list[index].split("_")[:2])

        data["spacing"] = self._get_spacing(patient_frame_id)

        return data

    def _get_spacing(self, patient_frame_id: str):
        if self.raw_spacing is None:
            return None

        if self.split == "train":
            return self.raw_spacing.loc[patient_frame_id][:2].values
        else:
            return self.raw_spacing.loc[patient_frame_id].values



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
