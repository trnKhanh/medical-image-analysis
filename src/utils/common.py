import os
import zipfile
from datetime import datetime
from pathlib import Path
import numpy as np

def get_path(path: Path | str) -> Path:
    if isinstance(path, str):
        path = Path(path)
    return path


class dummy_context(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def get_current_time_str():
    return datetime.now().strftime("%d%m%Y_%H%M%S")

def draw_mask(image, mask, opacity=0.2):
    image = np.array(image)
    mask = np.array(mask)

    if image.ndim == 2:
        image = np.expand_dims(image, -1)

    class_colors = {
        1: np.array([255, 0, 0], dtype=np.uint8),
        2: np.array([0, 255, 0], dtype=np.uint8),
        3: np.array([0, 0, 255], dtype=np.uint8),
        4: np.array([128, 0, 255], dtype=np.uint8),
    }
    visualized_image = np.copy(image)
    if image.shape[-1] == 1:
        visualized_image = visualized_image.repeat(3, -1)

    for class_id in class_colors.keys():
        class_mask = mask == class_id
        visualized_image[class_mask] = opacity * class_colors[class_id] + (1 - opacity) * visualized_image[class_mask]
        
    return visualized_image


def zip_folder(folder_path: str | Path, zip_filename: str = None) -> Path:
    """
    Simple function to zip a folder and return the zip file path.

    Args:
        folder_path: Path to the folder to zip
        zip_filename: Optional custom name for a zip file (without .zip extension)

    Returns:
        Path to the created zip file
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder {folder_path} does not exist")

    if not folder_path.is_dir():
        raise ValueError(f"{folder_path} is not a directory")

    if zip_filename is None:
        zip_filename = folder_path.name

    zip_path = folder_path.parent / f"{zip_filename}.zip"

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(folder_path)
                zipf.write(file_path, arcname)

    return zip_path