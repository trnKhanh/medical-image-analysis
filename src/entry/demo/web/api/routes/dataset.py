from typing import List


from fastapi import APIRouter, File, Form, UploadFile
from entry.demo.web.models.requests import (AnnotationRequest, ImageUploadRequest)
from entry.demo.web.models.responses import (AnnotationResponse,
                                             ImageUploadResponse)
from entry.demo.web.services.dataset import dataset_service

router = APIRouter()


@router.post("/upload/images", response_model=ImageUploadResponse)
async def upload_images(
    files: List[UploadFile] = File(...),
    type: str = Form(...)
):
    """Upload multiple images to the dataset."""
    request = ImageUploadRequest(type=type, images=files)
    return await dataset_service.upload_images(request)


@router.get("/download")
async def create_dataset():
    """Create a dataset with train and pool splits."""
    result = await dataset_service.export_dataset()
    return await dataset_service.create_streaming_response(result)


@router.post("/annotations", response_model=AnnotationResponse)
async def save_annotation(request: AnnotationRequest):
    """Save an annotation for an image."""
    return await dataset_service.save_annotation(request)
