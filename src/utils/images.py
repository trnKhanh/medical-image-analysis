import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image

class_color_map = {
    1: "#ff0000",
    2: "#00ff00",
}

def read_nrrd(image_path: str | Path) -> np.ndarray:
    data = sitk.ReadImage(image_path)
    data = sitk.Cast(sitk.RescaleIntensity(data), sitk.sitkUInt8) 
    data = sitk.GetArrayFromImage(data)

    return data

def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))

def image_to_base64(image_pil):
    buffered = BytesIO()
    image_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def hex_to_rgb(h) -> list:
    h = h[1:]
    return [int(h[i : i + 2], 16) for i in range(0, 6, 2)]
