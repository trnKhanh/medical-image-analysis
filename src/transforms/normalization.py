import torch
import torchvision.transforms.functional as F

from PIL import Image

from .common import image_to_tensor


class ZScoreNormalize:
    def __init__(self, target_dtype: torch.dtype = torch.float32):
        self.target_dtype = target_dtype

    def __call__(
        self, data: torch.Tensor | Image.Image, seg: torch.Tensor | Image.Image
    ):
        data = image_to_tensor(data)
        seg = image_to_tensor(seg)

        data = data.to(dtype=self.target_dtype)
        mean = data.mean()
        std = data.std()

        data = (data - mean) / std.clip(1e-8)

        return data, seg
