from pathlib import Path
from typing import Any, Callable

import torch
import torchvision.transforms.functional as F
from PIL import Image

from .basedataset import BaseDataset
from transforms.joint_transform import JointResize


class ExtendableDataset(BaseDataset):
    @staticmethod
    def find_samples(data_path: Path | str, require_label: bool = True) -> list[dict]:
        raise RuntimeError("ExtendableDataset does not have find_samples function")

    def __init__(self, dataset: BaseDataset, image_idx: list | None = None):
        self.dataset = dataset
        self.case_name_to_idx = {}

        for id in range(len(self.dataset)):
            sample = self.dataset[id]

            self.case_name_to_idx[sample["case_name"]] = id

        if image_idx is None:
            image_idx = list(self.case_name_to_idx.keys())

        self.image_idx = image_idx

    def __len__(self):
        return len(self.image_idx)

    def __getitem__(self, index):
        return self.get_sample(index)

    def get_sample(self, index: int, normalize: bool = True):
        case_name = self.image_idx[index]

        return self.dataset.get_sample(self.case_name_to_idx[case_name], normalize)


class ImageDataset(BaseDataset):
    @staticmethod
    def find_samples(data_path: Path | str, require_label: bool = True) -> list[dict]:
        raise RuntimeError("ImageDataset does not have find_samples function")

    def __init__(
        self,
        samples_list: list[Path | str],
        normalize: Callable | None = None,
        transform: Callable | None = None,
        image_channels: int = 3,
        image_size: int | tuple[int, int] | None = None,
    ):
        self.samples_list = samples_list
        self.normalize = normalize
        self.transform = transform
        self.image_channels = image_channels
        self.image_size = image_size

        self.final_transform = self._get_final_transform()

    def _get_final_transform(self):
        if self.image_size is None:
            return None
        else:
            return JointResize(self.image_size)

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, index):
        return self.get_sample(index)

    def get_sample(self, index: int, normalize: bool = True) -> Any:
        image_path = self.samples_list[index]

        image_pil = Image.open(image_path).convert("L")

        image = F.to_tensor(image_pil)
        label = torch.zeros((1, image.shape[-2], image.shape[-1]))
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

        data["case_name"] = image_path

        return data
