from pathlib import Path

import torch
import numpy as np
import SimpleITK as sitk

from scipy.ndimage import zoom


def read_nrrd(image_path: str | Path) -> np.ndarray:
    data = sitk.ReadImage(image_path)
    data = sitk.Cast(sitk.RescaleIntensity(data), sitk.sitkUInt8)
    data = sitk.GetArrayFromImage(data)

    return data

def zoom_image(
    input: torch.Tensor | np.ndarray,
    new_size: list[int] | tuple[int, int],
    order: int = 3,
) -> torch.Tensor | np.ndarray:
    if isinstance(input, torch.Tensor):
        input_np = input.cpu().numpy()
    else:
        input_np = input.copy()

    if input.ndim == 4:
        B, C, H, W = input_np.shape
        resize_input = np.zeros(
            (B, C, new_size[0], new_size[1]), dtype=input_np.dtype
        )
        for b in range(B):
            for c in range(C):
                resize_input[b, c] = torch.from_numpy(
                    zoom(
                        input_np[b, c],
                        (new_size[0] / H, new_size[1] / W),
                        order=order,
                    )
                )

    elif input.ndim == 3:
        C, H, W = input_np.shape
        resize_input = np.zeros(
            (C, new_size[0], new_size[1]), dtype=input_np.dtype
        )
        for c in range(C):
            resize_input[c] = torch.from_numpy(
                zoom(
                    input_np[c], (new_size[0] / H, new_size[1] / W), order=order
                )
            )
    elif input.ndim == 2:
        H, W = input_np.shape
        resize_input = zoom(input_np, (new_size[0] / H, new_size[1] / W), order=order)
    else:
        raise RuntimeError(f"input.ndim={input.ndim} is not supported")

    if isinstance(input, torch.Tensor):
        return torch.from_numpy(resize_input).to(input.device, dtype=input.dtype)
    else:
        return resize_input

