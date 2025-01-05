"""Data constructor handler"""

__all__ = ["BaseHandler"]

import os
import cv2
import glob

from pathlib import Path

class BaseHandler:
    """Base class for data constructors"""
    def __init__(self) -> None:
        """
        Initialize the handler.

        :return:
        """

        labeled_data_folder = os.path.join("dataset", "train", "labeled_data")
        self.raw_images_path = os.path.join(labeled_data_folder, "images")
        self.mask_images_path = os.path.join(labeled_data_folder, "labels")
        self.target_images_path = os.path.join(labeled_data_folder, "targets")
        os.makedirs(self.target_images_path, exist_ok=True)

    def __call__(self, *args, **kwargs) -> None:
        """
        Execute the handler.
        """
        raw_image_paths = glob.glob(os.path.join(self.raw_images_path, "*.png"))

        for raw_image_path in raw_image_paths:
            filename = os.path.basename(raw_image_path)
            raw_image = cv2.imread(raw_image_path, cv2.IMREAD_GRAYSCALE)
            mask_image_path = os.path.join(self.mask_images_path, filename)
            mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
            target_image_path = os.path.join(self.target_images_path, filename)
            target_image = raw_image * mask_image
            cv2.imwrite(target_image_path, target_image)
        