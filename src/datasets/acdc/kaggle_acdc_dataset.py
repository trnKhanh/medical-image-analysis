import os
from typing import Any, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F


class KaggleACDCDataset:
    """Dataset handler for the Kaggle ACDC dataset."""
    @staticmethod
    def load(directory: str, is_training: bool = True, target_size: tuple[int, int]=(256, 256), max_samples: Optional[int] = None) -> dict:
        """
        Loads the dataset from the given directory.

        :param directory: The directory containing the dataset.
        :param is_training: Whether to load the dataset for training.
        :param target_size: The target size for resizing the images and masks.
        :param max_samples: The maximum number of samples to load.

        :return: A tuple containing the images and masks (if is_training is True), or just the images (if is_training is False).
        """
        images, masks = [], []
        sample_count = 0

        for filename in os.listdir(directory):
            if max_samples and sample_count >= max_samples:
                break
            if filename.endswith(".h5"):
                try:
                    with h5py.File(os.path.join(directory, filename), "r") as f:
                        image = f['image'][:]
                        if is_training:
                            mask = f['label'][:]
                        if image.ndim == 3:  # (slices, H, W)
                            for slice_idx in range(image.shape[0]):
                                slice_img = image[slice_idx]  # (H, W)
                                slice_img = np.expand_dims(slice_img, axis=-1)  # (H, W, 1)
                                slice_img = resize_tensor(slice_img, target_size, mode='bilinear')
                                images.append(slice_img)

                                if is_training:
                                    slice_mask = mask[slice_idx]
                                    slice_mask = np.expand_dims(slice_mask, axis=-1)
                                    slice_mask = resize_tensor(slice_mask, target_size, mode='nearest')
                                    masks.append(np.squeeze(slice_mask, axis=-1))

                        elif image.ndim == 2:  # (H, W)
                            image = np.expand_dims(image, axis=-1)  # (H, W, 1)
                            image = resize_tensor(image, target_size, mode='bilinear')
                            images.append(image)

                            if is_training:
                                mask = np.expand_dims(mask, axis=-1)
                                mask = resize_tensor(mask, target_size, mode='nearest')
                                masks.append(np.squeeze(mask, axis=-1))
                        else:
                            print(f"Unexpected shape in {filename}: {image.shape}")
                            continue
                        sample_count += 1
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    continue

        images = np.array(images, dtype=np.float32)
        masks = np.array(masks, dtype=np.uint8) if is_training else None
        return {"images": images, "masks": masks} if is_training else {"images": images}

def resize_tensor(tensor, target_size, mode) -> Any:
    # tensor shape: (H, W, 1)
    tensor = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0).float() # (1, 1, H, W)
    resized = F.interpolate(tensor, target_size, mode=mode, align_corners=False if mode == "bilinear" else None)
    resized = resized.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, 1)
    return resized
