from typing import List
import base64

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from entry.demo.web.config import settings
from entry.demo.web.models.requests import (AnnotationRequest,
                                            DatasetExportRequest,
                                            DatasetRequest, ImageUploadRequest)
from entry.demo.web.models.responses import (AnnotationResponse,
                                             DatasetExportResponse,
                                             DatasetResponse, ImageInfo,
                                             ImageUploadResponse)
from entry.demo.web.services.dataset import dataset_service

router = APIRouter()


@router.post("/upload/train-images", response_model=ImageUploadResponse)
async def upload_image(request: ImageUploadRequest):
    """Upload a single image to the dataset."""
    return await dataset_service.upload_image(request)


@router.post("/images/upload-file", response_model=ImageUploadResponse)
async def upload_image_file(
        file: UploadFile = File(...),
        case_name: str = Form(None)
):
    """Upload a single image file to the dataset."""
    try:
        content = await file.read()

        file_content_b64 = base64.b64encode(content).decode()

        request = ImageUploadRequest(
            filename=file.filename,
            content=file_content_b64,
            case_name=case_name
        )

        return await dataset_service.upload_image(request)
    except Exception as e:
        return ImageUploadResponse(
            success=False,
            message=f"Failed to upload file: {str(e)}"
        )


@router.post("/images/batch", response_model=List[ImageUploadResponse])
async def upload_images_batch(requests: List[ImageUploadRequest]):
    """Upload multiple images in batch."""
    return await dataset_service.upload_images_batch(requests)


@router.get("/images", response_model=List[ImageInfo])
async def get_image_list():
    """Get list of all uploaded images."""
    return await dataset_service.get_image_list()


@router.post("/datasets/{dataset_name}", response_model=DatasetResponse)
async def create_dataset(dataset_name: str, request: DatasetRequest):
    """Create a dataset with train and pool splits."""
    return await dataset_service.create_dataset(request, dataset_name)


@router.get("/datasets/{dataset_name}", response_model=DatasetResponse)
async def get_dataset_info(dataset_name: str):
    """Get information about a specific dataset."""
    return await dataset_service.get_dataset_info(dataset_name)


@router.get("/datasets")
async def list_datasets():
    """List all available datasets."""
    return {
        "datasets": await dataset_service.list_datasets(),
        "message": "Datasets retrieved successfully"
    }


@router.post("/annotations", response_model=AnnotationResponse)
async def save_annotation(request: AnnotationRequest):
    """Save an annotation for an image."""
    return await dataset_service.save_annotation(request)


@router.post("/export", response_model=DatasetExportResponse)
async def export_dataset(request: DatasetExportRequest):
    """Export dataset with annotations to a downloadable format."""
    return await dataset_service.export_dataset(request)


@router.get("/export/{export_filename}")
async def download_exported_dataset(export_filename: str):
    """Download an exported dataset file."""
    export_path = settings.DATASETS_DIR / "exports" / export_filename

    if not export_path.exists():
        raise HTTPException(status_code=404, detail="Export file not found")

    return FileResponse(
        path=str(export_path),
        filename=export_filename,
        media_type='application/octet-stream'
    )


@router.post("/clear-data")
async def clear_dataset_data():
    """Clear all in-memory dataset data."""
    dataset_service.clear_data()
    return {"message": "Dataset data cleared successfully"}


@router.get("/stats")
async def get_dataset_stats():
    """Get current dataset statistics."""
    images = await dataset_service.get_image_list()
    datasets = await dataset_service.list_datasets()

    return {
        "total_images": len(images),
        "total_datasets": len(datasets),
        "total_annotations": len(dataset_service.current_annotations),
        "message": "Dataset statistics retrieved successfully"
    }
