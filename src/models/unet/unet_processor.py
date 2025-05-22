import os

import cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import numpy as np


class UnetProcessor:
    def __init__(
        self,
        image_size: list[int] | tuple[int, int] | None = None,
        dilate_size: int = 5,
        erode_size: int = 5,
        smooth_kernel: int = 7,
    ):
        """
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.
        """
        self.dilate_size = dilate_size
        self.erode_size = erode_size
        self.smooth_kernel = smooth_kernel

        self.mean = None
        self.std = None
        if image_size is not None:
            image_size = list(image_size)
        if image_size and len(image_size) < 2:
            image_size *= 2
        self.image_size = image_size

    def preprocess(self, X: torch.Tensor):
        image = X
        if self.image_size and (
            self.image_size[0] != image.shape[-2]
            or self.image_size[1] != image.shape[-1]
        ):
            image = F.resize(
                image, self.image_size, F.InterpolationMode.BILINEAR
            )
        if image.ndim == 3:
            image = image.unsqueeze(0)

        return image

    def postprocess(
        self, P: torch.Tensor, ori_shape: list[int], do_denoise: bool = False
    ):
        masks = P.clone()
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)

        if self.image_size and (
            ori_shape[0] != masks.shape[-2]
            or ori_shape[1] != masks.shape[-1]
        ):
            masks = F.resize(masks, ori_shape, F.InterpolationMode.NEAREST)


        if do_denoise:
            processed_masks = torch.zeros_like(masks)
            for i in range(masks.shape[0]):
                processed_masks[i] = self.denoise_one_mask(masks[i])
        else:
            processed_masks = masks

        return processed_masks.to(P.device, dtype=P.dtype)

    def denoise_one_mask(self, P: torch.Tensor):
        mask = P.detach().numpy()
        mask_list = []

        pad_size = max(self.dilate_size, self.erode_size)

        # Denoise the object mask (i.e. including all the classes)
        object_mask = np.zeros_like(mask, dtype=np.uint8)
        object_mask[mask > 0] = 255
        object_mask = self.pad_mask(object_mask, pad_size)

        denoised_object_mask = self.remove_cc(self.fill_hole(object_mask))
        denoised_object_mask = self.remove_pad(denoised_object_mask, pad_size)
        object_mask = denoised_object_mask.copy()

        final_object_mask = self.smoothen_boundary(object_mask)
        mask_list.append(final_object_mask == 0)

        # Denoise individual class except for the last class
        num_classes = 2
        for c in range(1, num_classes):
            class_mask = np.zeros_like(mask, dtype=np.uint8)
            class_mask[mask == c] = 255
            class_mask = self.pad_mask(class_mask, pad_size)

            denoised_class_mask = self.remove_cc(self.fill_hole(class_mask))
            denoised_class_mask = self.remove_pad(denoised_class_mask, pad_size)
            class_mask = denoised_class_mask.copy()

            final_class_mask = self.smoothen_boundary(class_mask)
            mask_list.append(final_class_mask > 0)

        # First we fill the whole mask with the last class
        mask = np.ones_like(mask) * num_classes
        c = num_classes - 1
        # Then, we iteratively fill the mask with the other class
        # This ensure the denoise_object_mask is the same after filling
        for class_mask in mask_list[::-1]:
            mask[class_mask] = c
            c -= 1

        return torch.from_numpy(mask)

    def fill_hole(self, mask):
        dilated_mask = self.dilate(mask, self.dilate_size)
        eroded_mask = self.erode(dilated_mask, self.erode_size)

        return eroded_mask

    def remove_cc(self, mask):
        eroded_mask = self.erode(mask, self.erode_size)
        dilated_mask = self.dilate(eroded_mask, self.dilate_size)

        return dilated_mask

    def pad_mask(self, mask, pad_size):
        padded_mask = cv2.copyMakeBorder(
            mask,
            pad_size,
            pad_size,
            pad_size,
            pad_size,
            cv2.BORDER_CONSTANT,
            None,
            [0],
        )
        return padded_mask

    def remove_pad(self, mask, pad_size):
        top = pad_size
        bot = mask.shape[0] - pad_size
        left = pad_size
        right = mask.shape[1] - pad_size
        return mask[top:bot, left:right]

    def dilate(self, mask, dilate_size):
        kernel_size = (dilate_size * 2 + 1, dilate_size * 2 + 1)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        res = cv2.dilate(mask, element)
        return res

    def erode(self, mask, erode_size):
        kernel_size = (erode_size * 2 + 1, erode_size * 2 + 1)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        res = cv2.erode(mask, element)
        return res

    def smoothen_boundary(self, mask):
        mask = cv2.GaussianBlur(
            mask, (self.smooth_kernel, self.smooth_kernel), 0
        )
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask
