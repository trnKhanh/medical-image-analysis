import json
from logging import Logger
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
import torchvision.transforms.functional as F

from datasets import BaseDataset
from datasets.utils.exceptions import SplitDictKeyException
from datasets.utils.logging import (log_no_split_dict, log_not_found_label,
                                    log_not_found_split_dict,
                                    log_not_found_split_dict_key)
from utils import get_path
from utils.images import read_nrrd


class LA2018Dataset(BaseDataset):
    IMAGE_FILE = "lgemri.nrrd"
    LABEL_ENDO_FILE = "laendo.nrrd"
    LABEL_WALL_FILE = "lawall.nrrd"

    @staticmethod
    def find_samples(
        data_path: Path | str, require_label: bool = True
    ) -> list[dict]:
        data_path = get_path(data_path)
        samples = []
        for patient in data_path.glob("*"):
            if not patient.is_dir():
                continue
            if not (patient / LA2018Dataset.IMAGE_FILE).is_file():
                continue
            labeled = True
            labeled &= (patient / LA2018Dataset.LABEL_ENDO_FILE).is_file()
            labeled &= (patient / LA2018Dataset.LABEL_WALL_FILE).is_file()

            if require_label and not labeled:
                continue

            samples.append(
                {
                    "id": patient.stem,
                    "path": patient.resolve(),
                    "labeled": labeled,
                }
            )

        return samples

    def __init__(
        self,
        data_path: Path | str,
        require_label: bool = True,
        transform: Callable | None = None,
        normalize: Callable | None = None,
        sample_ids: list[str] | None = None,
        logger: Logger | None = None,
    ):
        self.data_path = data_path
        self.require_label = require_label
        self.transform = transform
        self.normalize = normalize
        self.logger = logger
        self.sample_ids = sample_ids

        self._register_samples()

    def _register_samples(self):
        samples = LA2018Dataset.find_samples(self.data_path, self.require_label)
        registed_samples = []

        for sample in samples:
            if self.sample_ids and (sample[id] not in self.sample_ids):
                continue
            image_path = get_path(sample["path"]) / LA2018Dataset.IMAGE_FILE
            label_endo_path = (
                get_path(sample["path"]) / LA2018Dataset.LABEL_ENDO_FILE
            )
            label_wall_path = (
                get_path(sample["path"]) / LA2018Dataset.LABEL_WALL_FILE
            )
            registed_samples.append({"image": image_path})
            if label_endo_path.is_file():
                registed_samples[-1]["label_endo"] = label_endo_path
            if label_wall_path.is_file():
                registed_samples[-1]["label_wall"] = label_wall_path

        self.samples = registed_samples

    def __getitem__(self, index):
        return self.get_sample(index)

    def get_sample(self, index: int, normalize: bool = True):
        sample = self.samples[index]

        image = read_nrrd(sample["image"])

        try:
            label_endo = read_nrrd(sample["label_endo"])
            label_wall = read_nrrd(sample["label_wall"])
            label = np.zeros_like(image)
            label[label_endo > 0] = 1
            label[label_wall > 0] = 2
        except Exception as e:
            if self.require_label:
                raise e
            else:
                label = np.ones_like(image) * -1

        if self.transform:
            image, label = self.transform(image, label)

        if self.normalize and normalize:
            image, label = self.normalize(image, label)

        if not isinstance(image, torch.Tensor):
            image = F.to_tensor(image)

        if not isinstance(label, torch.Tensor):
            label = F.to_tensor(label)

        return image, label
