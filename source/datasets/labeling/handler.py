"""Labeling handler."""

__all__ = ["BaseHandler"]

import os
import glob

class BaseHandler:
    """Base class for labeling handlers."""
    def __init__(self):
        """Initialize the handler."""
        self.source_data_folder = os.path.join("dataset", "train", "unlabeled_data")
        self.target_data_folder = os.path.join("dataset", "train", "targets")
        os.makedirs(self.target_data_folder, exist_ok=True)

    def __call__(self, *args, **kwargs):
        """Execute the handler."""
        image_paths = glob.glob(os.path.join(self.source_data_folder, "*.png"))
        for image_path in image_paths:
            filename = os.path.basename(image_path)
            target_image_path = os.path.join(self.target_data_folder, filename)
            os.rename(image_path, target_image_path)
