from copy import deepcopy
from typing import Callable, Dict, List, Literal
from pathlib import Path
from logging import Logger
import json

import torch
from datasets.basedataset import BaseDataset
import torchvision.transforms.functional as F
from PIL import Image

from utils import get_path


class FUGCDataset(BaseDataset):
    images_dir = "images"
    labels_dir = "labels"

    @staticmethod
    def get_samples(data_path: Path | str) -> List[str]:
        data_path = get_path(data_path)
        images_dir_path = data_path / FUGCDataset.images_dir
        labels_dir_path = data_path / FUGCDataset.labels_dir

        samples = []
        for image_path in images_dir_path.glob("*.png"):
            image_id = image_path.stem

            try:
                next(labels_dir_path.glob(f"{image_id}.*"))
            except StopIteration:
                continue

            samples.append(image_id)

        return samples

    def __init__(
        self,
        data_path: Path | str,
        transform: Callable | None = None,
        normalize: Callable | None = None,
        split: Literal["train", "valid"] = "train",
        split_dict: Dict | Path | str | None = None,
        logger: Logger | None = None,
        oversample: int = 1
    ):
        self.data_path = get_path(data_path)
        self.transform = transform
        self.normalize = normalize
        self.split = split
        self.oversample = oversample

        if isinstance(split_dict, (str, Path)):
            split_path = get_path(split_dict)
            try:
                with open(split_path, "r") as f:
                    split_dict = dict(json.load(f))
            except Exception as e:
                if self.logger:
                    self.logger.warn(
                        f'Cannot read split_dict from "{split_path}" due to error={e}'
                    )
                split_dict = None

        if split_dict:
            if split not in split_dict:
                if self.logger:
                    self.logger.warn(
                        f'Invalid split_dict: "{split}" key not found.'
                    )
                split_dict = None

        self.split_dict = split_dict
        self.logger = logger

        self._register_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.get_sample(index)

    def get_sample(self, index, normalize=True):
        sample = self.samples[index]
        image_path = sample["image"]
        label_path = sample["label"]

        image = Image.open(image_path).convert("RGB")
        label = F.pil_to_tensor(Image.open(label_path)).to(dtype=torch.int32)

        if self.transform:
            image, label = self.transform(image, label)

        if self.normalize and normalize:
            image, label = self.normalize(image, label)

        if not isinstance(image, torch.Tensor):
            image = F.to_tensor(image)

        return image, label

    def get_image_path(self, index):
        return self.samples[index]["image"]

    def _register_samples(self):
        images_dir_path = self.data_path / FUGCDataset.images_dir
        labels_dir_path = self.data_path / FUGCDataset.labels_dir

        self.samples: List[Dict[str, Path]] = []
        if self.split_dict:
            for image_id in self.split_dict[self.split]:
                self.samples.append({
                    "image": images_dir_path / f"{image_id}.png",
                    "label": labels_dir_path / f"{image_id}.png",
                })
        else:
            for image_path in images_dir_path.glob("*.png"):
                image_id = image_path.stem

                if self.split_dict and image_id not in self.split_dict[self.split]:
                    continue

                try:
                    label_path = next(labels_dir_path.glob(f"{image_id}.*"))
                except StopIteration:
                    if self.logger:
                        self.logger.warn(
                            f"Image {image_path.name} does not have the corresponding label file"
                        )
                    continue

                self.samples.append({"image": image_path, "label": label_path})

        oversampled_samples = []
        for _ in range(self.oversample):
            oversampled_samples.extend(deepcopy(self.samples))
        self.samples = oversampled_samples
