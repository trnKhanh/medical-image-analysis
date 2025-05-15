__all__ = ["ZScoreNormalize", "MinMaxNormalize"]

from abc import ABC, abstractmethod
from typing import Any

import torch

from .common import image_to_tensor


class BaseNormalize(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        raise NotImplementedError


class ZScoreNormalize(BaseNormalize):
    def __init__(self, target_dtype: torch.dtype = torch.float32):
        self.target_dtype = target_dtype

    def __call__(self, data: dict) -> dict:
        image = image_to_tensor(data["image"])
        label = image_to_tensor(data["label"])

        image = image.to(dtype=self.target_dtype)
        mean = image.mean()
        std = image.std()

        image = (image - mean) / std.clip(1e-8)

        data["image"] = image
        data["label"] = label

        return data


class MinMaxNormalize(BaseNormalize):
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0, target_dtype: torch.dtype = torch.float32):
        self.target_dtype = target_dtype
        self.min = min_val
        self.max = max_val

    def __call__(self, image: Any) -> Any:
        image = image_to_tensor(image)
        if self.min is None:
            self.min = image.min()
            self.max = image.max()
        image = (image - self.min) / (self.max - self.min)
        image = image.to(dtype=self.target_dtype)

        return image
