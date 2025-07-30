import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image
from fastapi import HTTPException
from starlette.status import HTTP_400_BAD_REQUEST

class_color_map = {
    1: "#ff0000",
    2: "#00ff00",
}

def read_nrrd(image_path: str | Path) -> np.ndarray:
    data = sitk.ReadImage(image_path)
    data = sitk.Cast(sitk.RescaleIntensity(data), sitk.sitkUInt8) 
    data = sitk.GetArrayFromImage(data)

    return data


def base64_to_image(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    try:
        # Handle both cases: with and without data URL prefix
        if isinstance(base64_string, str):
            if base64_string.startswith('data:image'):
                # Remove data URL prefix if present
                base64_string = base64_string.split(',')[1]

            base64_string = base64_string.strip()

            image_data = base64.b64decode(base64_string)

            # Check if we have actual image data
            if len(image_data) < 10:
                raise ValueError(f"Decoded data too small ({len(image_data)} bytes), probably not a valid image")

            byte_stream = BytesIO(image_data)
            image = Image.open(byte_stream)

            return image
        else:
            raise ValueError(f"Expected string, got {type(base64_string)}")
    except Exception as e:
        print(f"DEBUG: Error in base64_to_image: {str(e)}")
        print(f"DEBUG: Input type: {type(base64_string)}")
        if isinstance(base64_string, str):
            print(f"DEBUG: Input length: {len(base64_string)}")
            print(f"DEBUG: Input starts with: {base64_string[:100]}")
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"Invalid image data: {str(e)}")


def image_to_base64(image) -> str:
    """Convert PIL Image or numpy array to base64 string"""
    try:
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                if image.shape[2] == 4:
                    pil_image = Image.fromarray(image, 'RGBA')
                else:
                    pil_image = Image.fromarray(image, 'RGB')
            else:
                pil_image = Image.fromarray(image, 'L')
        else:
            pil_image = image

        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    except Exception as e:
        print(f"DEBUG: Error in image_to_base64: {str(e)}")
        print(f"DEBUG: Input type: {type(image)}")
        if isinstance(image, np.ndarray):
            print(f"DEBUG: Array shape: {image.shape}, dtype: {image.dtype}")
        raise e

def hex_to_rgb(h) -> list:
    h = h[1:]
    return [int(h[i : i + 2], 16) for i in range(0, 6, 2)]
