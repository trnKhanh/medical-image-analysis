from pathlib import Path

import numpy as np
import SimpleITK as sitk


def read_nrrd(image_path: str | Path) -> np.ndarray:
    data = sitk.ReadImage(image_path)
    data = sitk.Cast(sitk.RescaleIntensity(data), sitk.sitkUInt8) 
    data = sitk.GetArrayFromImage(data)

    return data

