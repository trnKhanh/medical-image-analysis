import itertools
import json
from logging import Logger
from pathlib import Path
from typing import Literal, Callable, Any

import h5py
import numpy as np
import torch
from torch.utils.data.sampler import Sampler
import torchvision.transforms.functional as F
from PIL import Image

import pandas as pd

from ..basedataset import BaseDataset
from utils import get_path
from transforms.joint_transform import JointResize


class TN3KDataset(BaseDataset):
    CLASSES = {0: "bg", 1: "thyroid"}
    PROCESSED_DIR = "ACDC"

    TEST_IMAGES_DIR = "test-image"
    TEST_LABELS_DIR = "test-mask"

    TRAINVAL_IMAGES_DIR = "trainval-image"
    TRAINVAL_LABELS_DIR = "trainval-mask"
    TRAINVAL_SPLIT_FORMAT = "tn3k-trainval-fold{}.json"

    NUM_CLASSES = 1

    @staticmethod
    def find_samples(
        data_path: Path | str, require_label: bool = True
    ) -> list[dict]:
        data_path = get_path(data_path)
        images_dir = data_path / TN3KDataset.TRAINVAL_IMAGES_DIR
        labels_dir = data_path / TN3KDataset.TRAINVAL_LABELS_DIR
        samples_list = []
        for image_path in images_dir.glob("*.jpg"):
            if not image_path.is_file():
                continue

            sample_id = image_path.stem

            label_path = labels_dir / image_path.name

            labeled = label_path.is_file()

            if require_label and not labeled:
                continue

            samples_list.append(
                {
                    "id": sample_id,
                    "image_path": image_path.resolve(),
                    "label_path": label_path.resolve(),
                    "labeled": labeled,
                }
            )

        return samples_list

    def __init__(
        self,
        data_path: Path | str,
        split: Literal["train", "valid", "test"] = "train",
        fold: int = 0,
        normalize: Callable | None = None,
        transform: Callable | None = None,
        logger: Logger | None = None,
        image_channels: int = 3,
        image_size: int | tuple[int, int] | None = None,
    ):
        self.data_path = get_path(data_path)
        self.split = split
        self.fold = fold
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
        split_file = TN3KDataset.TRAINVAL_SPLIT_FORMAT.format(self.fold)
        with open(split_file, "r") as f:
            split_dict = json.load(f)

        if self.split == "train":
            self.samples_list = [f"{sample_id:04}" for sample_id in split_dict["train"]]
        elif self.split == "valid":
            self.samples_list = [f"{sample_id:04}" for sample_id in split_dict["val"]]
        elif self.split == "test":
            self.samples_list = []
            test_images_dir = self.data_path / TN3KDataset.TEST_IMAGES_DIR
            test_labels_dir = self.data_path / TN3KDataset.TEST_LABELS_DIR
            for image_path in test_images_dir.glob("*.jpg"):
                if not image_path.is_file():
                    continue
                
                label_path = test_images_dir / image_path.name
                if not label_path.is_file():
                    continue

                self.samples_list.append(image_path.stem)

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, index):
        return self.get_sample(index)

    def get_sample(self, index: int, normalize: bool = True) -> Any:
        case = self.samples_list[index]

        if self.split != "test":
            image_path = self.data_path / f"{TN3KDataset.TRAINVAL_IMAGES_DIR}/{case}.jpg"
            label_path = self.data_path / f"{TN3KDataset.TRAINVAL_LABELS_DIR}/{case}.jpg"
        else:
            image_path = self.data_path / f"{TN3KDataset.TEST_IMAGES_DIR}/{case}.jpg"
            label_path = self.data_path / f"{TN3KDataset.TEST_LABELS_DIR}/{case}.jpg"

        image_pil = Image.open(image_path)
        label_pil = Image.open(label_path)

        image = F.pil_to_tensor(image_pil)
        label = F.pil_to_tensor(label_pil)

        data: dict = {"image": image, "label": label}

        if self.transform:
            data = self.transform(data)

        if self.final_transform:
            data = self.final_transform(data)

        if self.normalize and normalize:
            data = self.normalize(data)

        data["label"] = data["label"].squeeze(0)

        data["case_name"] = case

        return data
