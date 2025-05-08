from abc import ABC, abstractmethod
import json
from typing import Callable

import torch
import torchvision.transforms.functional as F

import numpy as np
from PIL import Image


class BaseTransform(ABC):
    @abstractmethod
    def __call__(self, data: dict) -> dict:
        pass

    @abstractmethod
    def get_params_dict(self) -> dict:
        pass


class RandomTransform(BaseTransform):
    def __init__(self, transform: BaseTransform, p):
        self.p = np.clip(p, 0.0, 1.0)
        self.transform = transform

    def __call__(self, data: dict) -> dict:
        if torch.rand(1).item() < self.p:
            return self.transform(data)

        return data

    def get_params_dict(self):
        params_dict = {
            RandomTransform.__name__: {
                "p": self.p,
                "transform": self.transform.get_params_dict(),
            }
        }
        return params_dict


class RandomChoiceTransform(BaseTransform):
    def __init__(
        self, transforms: list[BaseTransform], weight: list | None = None
    ):
        self.weight = (
            torch.Tensor(weight) if weight else torch.ones(len(transforms))
        )

        self.transforms = transforms

    def __call__(self, data: dict) -> dict:
        index = int(torch.multinomial(self.weight, 1).item())
        return self.transforms[index](data)

    def get_params_dict(self):
        params_dict = {
            RandomChoiceTransform.__name__: {
                "weights": self.weight.tolist(),
                "transforms": [t.get_params_dict() for t in self.transforms],
            }
        }
        return params_dict


class ComposeTransform(BaseTransform):
    def __init__(self, transforms: list[BaseTransform]):
        self.transforms = transforms

    def __call__(self, data: dict) -> dict:
        for t in self.transforms:
            data = t(data)
        return data

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
