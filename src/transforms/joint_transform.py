from typing import Tuple, List, Sequence

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

from PIL import Image

from .common import image_to_tensor, BaseTransform
from utils import zoom_image


class JointResize(BaseTransform):
    def __init__(
        self, image_size: Tuple[int, int] | int, use_torchvision: bool = True
    ):
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        if len(image_size) < 2:
            image_size = image_size * 2
        self.image_size = list(image_size)
        self.use_torchvision = use_torchvision

    def __call__(self, data: dict) -> dict:
        image = image_to_tensor(data["image"])
        label = image_to_tensor(data["label"])

        if self.use_torchvision:
            image = F.resize(
                image, self.image_size, F.InterpolationMode.BILINEAR
            )
            label = F.resize(
                label, self.image_size, F.InterpolationMode.NEAREST
            )
        else:
            image = zoom_image(image, self.image_size, order=3)
            label = zoom_image(label, self.image_size, order=0)

        data["image"] = image
        data["label"] = label

        return data

    def get_params_dict(self):
        params_dict = {
            JointResize.__name__: {
                "image_size": self.image_size,
            }
        }
        return params_dict


class RandomRotation90(BaseTransform):
    def __init__(self, axes: Tuple[int, int] = (-2, -1)):
        assert axes[0] != axes[1]

        self.axes = axes

    def __call__(self, data: dict) -> dict:
        image = image_to_tensor(data["image"])
        label = image_to_tensor(data["label"])

        k = int(torch.randint(0, 4, (1,)).item())
        image = torch.rot90(image, k, self.axes)
        label = torch.rot90(label, k, self.axes)

        data["image"] = image
        data["label"] = label

        return data

    def get_params_dict(self):
        params_dict = {
            RandomRotation90.__name__: {
                "axes": self.axes,
            }
        }
        return params_dict


class MirrorTransform(BaseTransform):
    def __init__(self, axes: int | Tuple[int, ...]):
        if not isinstance(axes, Sequence):
            axes = tuple([axes])
        self.axes = axes

    def __call__(self, data: dict) -> dict:
        image = image_to_tensor(data["image"])
        label = image_to_tensor(data["label"])

        if len(self.axes) == 0:
            data["image"] = image
            data["label"] = label

            return data

        image = torch.flip(image, self.axes)
        label = torch.flip(label, self.axes)

        data["image"] = image
        data["label"] = label

        return data

    def get_params_dict(self):
        params_dict = {
            MirrorTransform.__name__: {
                "allowed_axes": self.axes,
            }
        }
        return params_dict


class RandomRotation(BaseTransform):
    def __init__(self, degrees: float | Sequence[float]):
        if not isinstance(degrees, Sequence):
            degrees = [-degrees, degrees]

        self.degrees = list(degrees)

    def __call__(self, data: dict) -> dict:
        image = image_to_tensor(data["image"])
        label = image_to_tensor(data["label"])

        angle = T.RandomRotation.get_params(self.degrees)

        image = F.rotate(image, angle)
        label = F.rotate(label, angle)

        data["image"] = image
        data["label"] = label

        return data

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

    def __call__(self, data: dict) -> dict:
        image = image_to_tensor(data["image"])
        label = image_to_tensor(data["label"])

        i, j, h, w = T.RandomCrop.get_params(image, self.crop)
        image = F.crop(image, i, j, h, w)
        label = F.crop(label, i, j, h, w)

        data["image"] = image
        data["label"] = label

        return data

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

    def __call__(self, data: dict) -> dict:
        image = image_to_tensor(data["image"])
        label = image_to_tensor(data["label"])

        _, h, w = image.shape

        degree, translate, scale, shear = T.RandomAffine.get_params(
            self.degrees, self.translate, self.scale, self.shear, [h, w]
        )
        image = F.affine(image, degree, list(translate), scale, list(shear))
        label = F.affine(label, degree, list(translate), scale, list(shear))

        data["image"] = image
        data["label"] = label

        return data

    def get_params_dict(self):
        params_dict = {
            RandomAffine.__name__: {
                "degrees": self.degrees,
                "translate": self.translate,
                "scale": self.scale,
                "shear": self.shear,
            }
        }
        return params_dict
