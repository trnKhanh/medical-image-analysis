from typing import Tuple, List, Sequence

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

from PIL import Image

from .common import image_to_tensor, BaseTransform

class JointResize(BaseTransform):
    def __init__(self, image_size: Tuple[int, int] | int):
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        if len(image_size) < 2:
            image_size = image_size * 2
        self.image_size = list(image_size)

    def __call__(
        self, data: torch.Tensor | Image.Image, seg: torch.Tensor | Image.Image
    ):
        data = image_to_tensor(data)
        seg = image_to_tensor(seg)

        data = F.resize(data, self.image_size, F.InterpolationMode.BILINEAR)
        seg = F.resize(seg, self.image_size, F.InterpolationMode.NEAREST)

        return data, seg

    def get_params_dict(self):
        params_dict = {
            JointResize.__name__: {
                "image_size": self.image_size,
            }
        }
        return params_dict

class MirrorTransform(BaseTransform):
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

    def get_params_dict(self):
        params_dict = {
            MirrorTransform.__name__: {
                "allowed_axes": self.allowed_axes,
            }
        }
        return params_dict


class RandomRotation(BaseTransform):
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

    def get_params_dict(self):
        params_dict = {
            RandomRotation.__name__: {
                "degrees": self.degrees,
            }
        }
        return params_dict


class RandomCrop2D(BaseTransform):
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

    def get_params_dict(self):
        params_dict = {
            RandomCrop2D.__name__: {
                "crop": self.crop,
            }
        }
        return params_dict


class RandomAffine(BaseTransform):
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

    def get_params_dict(self):
        params_dict = {
            RandomAffine.__name__: {
                "degrees": self.degrees,
                "translate": self.translate,
                "scale": self.scale,
                "shear": self.shear
            }
        }
        return params_dict
