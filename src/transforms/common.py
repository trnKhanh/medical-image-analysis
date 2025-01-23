from abc import ABC, abstractmethod
import json
from typing import Callable

import torch
import torchvision.transforms.functional as F

import numpy as np
from PIL import Image

class BaseTransform(ABC):
    @abstractmethod
    def __call__(
        self, data: torch.Tensor | Image.Image, seg: torch.Tensor | Image.Image
    )-> tuple[torch.Tensor | Image.Image, torch.Tensor | Image.Image]:
        pass

    @abstractmethod
    def get_params_dict(self) -> dict:
        pass


class RandomTransform(BaseTransform):
    def __init__(self, transform: BaseTransform, p):
        self.p = np.clip(p, 0.0, 1.0)
        self.transform = transform

    def __call__(
        self, data: torch.Tensor | Image.Image, seg: torch.Tensor | Image.Image
    ):
        if torch.rand(1).item() < self.p:
            return self.transform(data, seg)

        return data, seg

    def get_params_dict(self):
        params_dict = {
            RandomTransform.__name__: {
                "p": self.p,
                "transform": self.transform.get_params_dict()
            }
        }
        return params_dict


class ComposeTransform(BaseTransform):
    def __init__(self, transforms: list[BaseTransform]):
        self.transforms = transforms

    def __call__(
        self, data: torch.Tensor | Image.Image, seg: torch.Tensor | Image.Image
    ):
        for t in self.transforms:
            data, seg = t(data, seg)
        return data, seg

    def get_params_dict(self):
        params_dict = {
            ComposeTransform.__name__: {
                "transforms": [t.get_params_dict() for t in self.transforms]
            }
        }
        return params_dict


def image_to_tensor(data: torch.Tensor | Image.Image | np.ndarray):
    if isinstance(data, (Image.Image, np.ndarray)):
        return F.to_tensor(data)

    return data
