from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    success: bool = Field(default=True, description="Operation success status")
    message: str = Field(default="Operation completed successfully", description="Response message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class ErrorResponse(BaseResponse):
    success: bool = Field(default=False)
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")


class ImageInfo(BaseModel):
    filename: str = Field(..., description="Image filename")
    path: str = Field(..., description="Image file path")
    size: tuple = Field(..., description="Image dimensions (width, height)")
    case_name: Optional[str] = Field(None, description="Case name")
    created_at: datetime = Field(default_factory=datetime.now, description="Upload timestamp")


class ImageUploadResponse(BaseResponse):
    success: bool = True
    message: str
    uploaded_images: Optional[List[ImageInfo]] = None
    failed_uploads: Optional[List[Dict[str, str]]] = None

class DatasetInfo(BaseModel):
    name: str = Field(..., description="Dataset name")
    train_count: int = Field(..., description="Number of training images")
    pool_count: int = Field(..., description="Number of pool images")
    total_count: int = Field(..., description="Total number of images")
    created_at: datetime = Field(default_factory=datetime.now, description="Dataset creation timestamp")


class DatasetResponse(BaseResponse):
    dataset_info: DatasetInfo = Field(..., description="Dataset information")


class ActiveSelectionResponse(BaseResponse):
    selected_images: List[str] = Field(..., description="Selected image paths for annotation")
    selection_method: str = Field(..., description="Selection method used")
    budget: int = Field(..., description="Budget used for selection")
    total_pool_size: int = Field(..., description="Total pool size before selection")


class   ModelCheckpointInfo(BaseModel):
    name: str = Field(..., description="Model checkpoint name")
    size: int = Field(..., description="File size in bytes")
    description: Optional[str] = Field(None, description="Model description")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


class ModelCheckpointResponse(BaseResponse):
    model_info: ModelCheckpointInfo = Field(..., description="Model checkpoint information")


class ModelCheckpointListResponse(BaseResponse):
    models: List[ModelCheckpointInfo] = Field(..., description="List of available model checkpoints")
    total_count: int = Field(..., description="Total number of models")


class PredictionResponse(BaseResponse):
    image_path: str = Field(..., description="Path to the predicted image")
    prediction_mask: str = Field(..., description="Base64 encoded prediction mask")
    prediction_visual: str = Field(..., description="Base64 encoded visualization")
    model_used: str = Field(..., description="Model checkpoint used for prediction")
    confidence_scores: Optional[Dict[str, float]] = Field(None, description="Confidence scores by class")


class AnnotationInfo(BaseModel):
    image_path: str = Field(..., description="Path to annotated image")
    mask_path: str = Field(..., description="Path to annotation mask")
    case_name: Optional[str] = Field(None, description="Case name")
    annotated_at: datetime = Field(default_factory=datetime.now, description="Annotation timestamp")


class AnnotationResponse(BaseResponse):
    annotation_info: AnnotationInfo = Field(..., description="Annotation information")


class DatasetExportResponse(BaseResponse):
    export_path: str = Field(..., description="Path to exported dataset")
    export_format: str = Field(..., description="Export format used")
    file_size: int = Field(..., description="Export file size in bytes")
    included_items: Dict[str, int] = Field(..., description="Count of included items")


class FoundationModelInfo(BaseModel):
    name: str = Field(..., description="Foundation model name")
    loaded: bool = Field(..., description="Is model loaded")
    device: str = Field(..., description="Device model is loaded on")


class SpecialistModelInfo(BaseModel):
    current_checkpoint: str = Field(..., description="Current specialist model checkpoint")
    loaded: bool = Field(..., description="Is model loaded")
    device: str = Field(..., description="Device model is loaded on")


class ActiveLearningConfigResponse(BaseResponse):
    config: Dict[str, Any] = Field(..., description="Current active learning configuration")


class HealthResponse(BaseResponse):
    status: str = Field(..., description="Health status")
    foundation_model_loaded: bool = Field(..., description="Foundation model status")
    specialist_model_loaded: bool = Field(..., description="Specialist model status")
    uptime: str = Field(..., description="Server uptime")
    version: str = Field(..., description="API version")
