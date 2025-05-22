from typing import Callable, Literal, Any
from pathlib import Path
from logging import Logger
import json

import torch
import torchvision.transforms.functional as F
from PIL import Image

from datasets.basedataset import BaseDataset
from transforms.joint_transform import JointResize
from utils import get_path


class FUGCDataset(BaseDataset):
    CLASSES = {0: "bg", 1: "anterior lip", 2: "posterior lip"}

    TRAIN_DIR = "train"
    VALID_DIR = "val"
    TEST_DIR = "test"

    IMAGES_DIR = "images"
    LABELS_DIR = "labels"

    NUM_CLASSES = 2

    @staticmethod
    def find_samples(
        data_path: Path | str, require_label: bool = True
    ) -> list[dict]:
        data_path = get_path(data_path)
        images_dir = data_path / FUGCDataset.TRAIN_DIR / FUGCDataset.IMAGES_DIR
        labels_dir = data_path / FUGCDataset.TRAIN_DIR / FUGCDataset.LABELS_DIR
        samples_list = []
        for image_path in images_dir.glob("*.png"):
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
        if self.split == "train":
            images_path = (
                self.data_path / FUGCDataset.TRAIN_DIR / FUGCDataset.IMAGES_DIR
            )
        elif self.split == "valid":
            images_path = (
                self.data_path / FUGCDataset.VALID_DIR / FUGCDataset.IMAGES_DIR
            )
        elif self.split == "test":
            images_path = (
                self.data_path / FUGCDataset.TEST_DIR / FUGCDataset.IMAGES_DIR
            )
        else:
            raise ValueError(f"FUGCDataset does not have {self.split} split")

        self.samples_list = [
            sample_path.stem for sample_path in images_path.glob("*.png")
        ]

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, index):
        return self.get_sample(index)

    def get_sample(self, index: int, normalize: bool = True) -> Any:
        case = self.samples_list[index]
        
        if self.split == "train":
            split_dir = FUGCDataset.TRAIN_DIR
        elif self.split == "valid":
            split_dir = FUGCDataset.VALID_DIR
        else:
            split_dir = FUGCDataset.TEST_DIR

        image_path = (
            self.data_path
            / split_dir
            / FUGCDataset.IMAGES_DIR
            / f"{case}.png"
        )
        label_path = (
            self.data_path
            / split_dir
            / FUGCDataset.LABELS_DIR
            / f"{case}.png"
        )

        image_pil = Image.open(image_path).convert("L")
        label_pil = Image.open(label_path)

        image = F.to_tensor(image_pil)
        label = F.pil_to_tensor(label_pil)
        label = label.long()

        image = image.repeat(self.image_channels // image.shape[0], 1, 1)

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
