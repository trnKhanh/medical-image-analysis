import base64
from typing import List, Optional
from fastapi import UploadFile

from pydantic import BaseModel, Field, validator


class ActiveLearningConfigRequest(BaseModel):
    budget: int = Field(default=10, ge=1, le=100, description="Number of samples to select")
    model: str = Field(default="BiomedCLIP", description="Foundation model name")
    device: str = Field(default="cpu", description="Device to use (cpu/cuda)")
    batch_size: int = Field(default=4, ge=1, le=32, description="Batch size for processing")
    loaded_feature_weight: float = Field(default=1.0, ge=0.0, le=10.0, description="Weight for loaded features")
    sharp_factor: float = Field(default=1.0, ge=0.0, le=10.0, description="Sharpening factor")
    loaded_feature_only: bool = Field(default=False, description="Use only loaded features")
    model_ckpt: str = Field(default="init_model.pth", description="Specialist model checkpoint path")

    @validator('device')
    def validate_device(cls, v):
        if v not in ['cpu', 'cuda']:
            raise ValueError('Device must be either "cpu" or "cuda"')
        return v

class ImageUploadRequest(BaseModel):
    type: str = Field(..., description="Image type (train/pool)")
    images: List[UploadFile] = Field(..., description="List of uploaded images")

class DatasetRequest(BaseModel):
    train_images: List[str] = Field(default=[], description="List of training image paths")
    pool_images: List[str] = Field(default=[], description="List of pool image paths")

class ActiveSelectionRequest(BaseModel):
    train_set: List[str] = Field(..., description="Training set image paths")
    pool_set: List[str] = Field(..., description="Pool set image paths")
    config: ActiveLearningConfigRequest = Field(default_factory=ActiveLearningConfigRequest)

class AnnotationRequest(BaseModel):
    image_path: str = Field(..., description="Path to the image being annotated")
    mask_data: str = Field(..., description="Base64 encoded mask data")
    case_name: Optional[str] = Field(None, description="Case name")

class ModelCheckpointRequest(BaseModel):
    name: str = Field(..., description="Model checkpoint name")
    description: Optional[str] = Field(None, description="Model description")
    file_content: str = Field(..., description="Base64 encoded model file content")

class PredictionRequest(BaseModel):
    image_path: str = Field(..., description="Path to image for prediction")
    model_ckpt: Optional[str] = Field(None, description="Specific model checkpoint to use")

class DatasetExportRequest(BaseModel):
    dataset_name: str = Field(..., description="Name for the exported dataset")
