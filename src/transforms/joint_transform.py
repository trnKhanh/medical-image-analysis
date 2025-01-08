from typing import Tuple, List, Sequence

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

from PIL import Image

from .common import image_to_tensor


class MirrorTransform:
    def __init__(self, allowed_axes: Tuple[int, ...]):
        self.allowed_axes = allowed_axes

    def __call__(
        self, data: torch.Tensor | Image.Image, seg: torch.Tensor | Image.Image
    ):
        data = image_to_tensor(data)
        seg = image_to_tensor(seg)

        if len(self.allowed_axes) == 0:
            return data, seg

        axes = [i + 1 for i in self.allowed_axes]
        data = torch.flip(data, axes)
        seg = torch.flip(data, axes)

        return data, seg


class RandomRotation:
    def __init__(self, degrees: float | Sequence[float]):
        if not isinstance(degrees, Sequence):
            degrees = [-degrees, degrees]

        self.degrees = list(degrees)

    def __call__(
        self, data: torch.Tensor | Image.Image, seg: torch.Tensor | Image.Image
    ):
        data = image_to_tensor(data)
        seg = image_to_tensor(seg)

        angle = T.RandomRotation.get_params(self.degrees)

        data = F.rotate(data, angle)
        seg = F.rotate(seg, angle)

        return data, seg


class RandomCrop2D:
    def __init__(self, crop: int | Tuple[int, int]):
        if not isinstance(crop, (List, Tuple)):
            crop = (crop, crop)
        self.crop = crop

    def __call__(
        self, data: torch.Tensor | Image.Image, seg: torch.Tensor | Image.Image
    ):
        data = image_to_tensor(data)
        seg = image_to_tensor(seg)

        i, j, h, w = T.RandomCrop.get_params(data, self.crop)
        data = F.crop(data, i, j, h, w)
        seg = F.crop(seg, i, j, h, w)

        return data, seg


class RandomAffine:
    def __init__(
        self,
        degrees: float | Sequence[float] = 0.0,
        translate: Tuple[float, float] | None = None,
        scale: Tuple[float, float] | None = None,
        shear: float | Sequence[float] | None = None,
    ):
        if not isinstance(degrees, Sequence):
            degrees = [-degrees, degrees]
        self.degrees = list(degrees)

        self.translate = list(translate) if translate else None
        self.scale = list(scale) if scale else None

        if shear:
            if not isinstance(shear, Sequence):
                shear = [-shear, shear]
            self.shear = list(shear)
        else:
            self.shear = None

    def __call__(
        self, data: torch.Tensor | Image.Image, seg: torch.Tensor | Image.Image
    ):
        data = image_to_tensor(data)
        seg = image_to_tensor(seg)

        _, h, w = data.shape

        degree, translate, scale, shear = T.RandomAffine.get_params(
            self.degrees, self.translate, self.scale, self.shear, [h, w]
        )
        data = F.affine(data, degree, list(translate), scale, list(shear))
        seg = F.affine(seg, degree, list(translate), scale, list(shear))

        return data, seg
