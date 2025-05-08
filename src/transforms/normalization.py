import torch
import torchvision.transforms.functional as F

from PIL import Image

from .common import image_to_tensor


class ZScoreNormalize:
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

