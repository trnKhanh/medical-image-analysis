from typing import Callable

import torch
import torchvision.transforms.functional as F

import numpy as np
from PIL import Image

class RandomTransform:
    def __init__(self, transform: Callable, p):
        self.p = np.clip(p, 0.0, 1.0)
        self.transform = transform

    def __call__(
        self, data: torch.Tensor | Image.Image, seg: torch.Tensor | Image.Image
    ):
        if torch.rand(1).item() < self.p:
            return self.transform(data, seg)

        return data, seg

class ComposeTransform:
    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms

    def __call__(
        self, data: torch.Tensor | Image.Image, seg: torch.Tensor | Image.Image
    ):
        for t in self.transforms:
            data, seg = t(data, seg)
        return data, seg

def image_to_tensor(data: torch.Tensor | Image.Image | np.ndarray):
    if isinstance(data, (Image.Image, np.ndarray)):
        return F.to_tensor(data)

    return data

    

