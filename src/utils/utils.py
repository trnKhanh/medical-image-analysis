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

def draw_mask(image, mask):
    class_colors = {
        1: np.array([255, 0, 0], dtype=np.uint8),
        2: np.array([0, 255, 0], dtype=np.uint8),
    }
    visualized_image = np.copy(image)
    for class_id in [1, 2]:
        class_mask = mask == class_id
        visualized_image[class_mask] = 0.1 * class_colors[class_id] + 0.9 * visualized_image[class_mask]
        
    return visualized_image
