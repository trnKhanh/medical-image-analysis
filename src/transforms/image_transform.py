import json
from typing import Sequence, Tuple
import math

import torch
from torch.nn.functional import interpolate
from torchvision import transforms as T
import torchvision.transforms.functional as F

from PIL import Image

from .common import image_to_tensor, BaseTransform


class RandomGamma(BaseTransform):
    def __init__(
        self,
        gamma: float | Sequence[float],
    ):
        if not isinstance(gamma, Sequence):
            gamma = [gamma, gamma]

        self.gamma = list(gamma)

    def __call__(
        self, data: torch.Tensor | Image.Image, seg: torch.Tensor | Image.Image
    ):
        data = image_to_tensor(data)
        seg = image_to_tensor(seg)

        gamma = torch.rand(1) * (self.gamma[1] - self.gamma[0]) + self.gamma[0]

        data = torch.pow(data, gamma)

        return data, seg

    def get_params_dict(self):
        params_dict = {
            RandomGamma.__name__: {
                "gamma": self.gamma,
            }
        }
        return params_dict


class RandomContrast(BaseTransform):
    def __init__(
        self,
        contrast: float | Tuple[float, float],
    ):
        if not isinstance(contrast, Sequence):
            contrast = (max(1.0 - contrast, 0.0), 1.0 + contrast)

        self.contrast = contrast
        self.fn = T.ColorJitter(contrast=self.contrast)

    def __call__(
        self, data: torch.Tensor | Image.Image, seg: torch.Tensor | Image.Image
    ):
        data = image_to_tensor(data)
        seg = image_to_tensor(seg)

        data = self.fn(data)

        return data, seg

    def get_params_dict(self):
        params_dict = {
            RandomContrast.__name__: {
                "contrast": self.contrast,
            }
        }
        return params_dict


class RandomBrightness(BaseTransform):
    def __init__(
        self,
        brightness: float | Tuple[float, float],
    ):
        if not isinstance(brightness, Sequence):
            brightness = (max(1.0 - brightness, 0.0), 1.0 + brightness)

        self.brightness = brightness
        self.fn = T.ColorJitter(contrast=self.brightness)

    def __call__(
        self, data: torch.Tensor | Image.Image, seg: torch.Tensor | Image.Image
    ):
        data = image_to_tensor(data)
        seg = image_to_tensor(seg)

        data = self.fn(data)

        return data, seg

    def get_params_dict(self):
        params_dict = {
            RandomBrightness.__name__: {
                "brightness": self.brightness,
            }
        }
        return params_dict


class RandomGaussianNoise(BaseTransform):
    def __init__(
        self,
        sigma: float | Sequence[float],
    ):
        if not isinstance(sigma, Sequence):
            sigma = [sigma, sigma]

        self.sigma = list(sigma)

    def __call__(
        self, data: torch.Tensor | Image.Image, seg: torch.Tensor | Image.Image
    ):
        data = image_to_tensor(data)
        seg = image_to_tensor(seg)

        sigma = (
            torch.rand(1).item() * (self.sigma[1] - self.sigma[0])
            + self.sigma[0]
        )
        noise = torch.normal(0, sigma, size=data.shape)

        data = torch.clip(data + noise, 0, 1)

        return data, seg

    def get_params_dict(self):
        params_dict = {
            RandomGaussianNoise.__name__: {
                "sigma": self.sigma,
            }
        }
        return params_dict


class RandomGaussianBlur(BaseTransform):
    def __init__(
        self,
        sigma: float | Sequence[float],
    ):
        if not isinstance(sigma, Sequence):
            sigma = [sigma, sigma]

        self.sigma = list(sigma)

    def __call__(
        self, data: torch.Tensor | Image.Image, seg: torch.Tensor | Image.Image
    ):
        data = image_to_tensor(data)
        seg = image_to_tensor(seg)

        sigma = (
            torch.rand(1).item() * (self.sigma[1] - self.sigma[0])
            + self.sigma[0]
        )

        kernel_size = self._get_kernel_size(sigma)

        data = F.gaussian_blur(
            data,
            [kernel_size] * (len(data.shape) - 1),
            [sigma] * (len(data.shape) - 1),
        )

        return data, seg

    def _get_kernel_size(self, sigma: float, truncate: float = 4.0):
        return self._round_to_odd(sigma * truncate + 0.5)

    def _round_to_odd(self, x: float):
        c = math.ceil(x)
        if c % 2:
            return c
        else:
            return c - 1

    def get_params_dict(self):
        params_dict = {
            RandomGaussianBlur.__name__: {
                "sigma": self.sigma,
            }
        }
        return params_dict


class SimulateLowRes(BaseTransform):
    def __init__(
        self,
        scale: float | Sequence[float],
    ):
        if not isinstance(scale, Sequence):
            scale = [scale, scale]
        self.scale = list(scale)

        self.upmodes = {1: "linear", 2: "bilinear", 3: "trilinear"}

    def __call__(
        self, data: torch.Tensor | Image.Image, seg: torch.Tensor | Image.Image
    ):
        data = image_to_tensor(data)
        seg = image_to_tensor(seg)

        scales = (
            torch.rand(len(data.shape) - 1) * (self.scale[1] - self.scale[0])
            + self.scale[0]
        ).tolist()

        origin_sizes = data.shape[1:]
        low_res_sizes = [int(s * i) for s, i in zip(scales, data.shape[1:])]
        low_res_data = interpolate(
            data[None], low_res_sizes, mode="nearest-exact"
        )
        data = interpolate(
            low_res_data, origin_sizes, mode=self.upmodes[data.ndim - 1]
        )[0]

        return data, seg

    def get_params_dict(self):
        params_dict = {
            SimulateLowRes.__name__: {
                "scale": self.scale,
            }
        }
        return params_dict
