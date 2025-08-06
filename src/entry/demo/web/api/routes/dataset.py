from typing import List

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from entry.demo.web.middleware.workspace import get_workspace, validate_disk_space
from entry.demo.web.models.requests import (AnnotationRequest,
                                            ImageUploadRequest)
from entry.demo.web.models.responses import (AnnotationResponse,
                                             ImageUploadResponse)
from entry.demo.web.services.dataset import (create_streaming_response,
                                             dataset_service)

router = APIRouter()


@router.post("/upload/images", response_model=ImageUploadResponse)
async def upload_images(
    workspace_id: str = Depends(get_workspace),
    is_enough_space: bool = Depends(validate_disk_space),
    files: List[UploadFile] = File(...),
    type: str = Form(...)
):
    """Upload multiple images to the dataset."""
    if not is_enough_space:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="Not enough disk space to upload images."
        )
    with dataset_service.workspace(workspace_id) as ws:
        request = ImageUploadRequest(type=type, images=files)
        return await ws.upload_images(request)


@router.get("/download")
async def download_dataset(workspace_id: str = Depends(get_workspace)):
    """Create a dataset with train and pool splits."""
    with dataset_service.workspace(workspace_id) as ws:
        result = await ws.export_dataset()
        return await create_streaming_response(result)


@router.post("/annotations", response_model=AnnotationResponse)
async def save_annotation(
    request: AnnotationRequest,
    workspace_id: str = Depends(get_workspace),
):
    """Save an annotation for an image."""
    with dataset_service.workspace(workspace_id) as ws:
        return await ws.save_annotation(request)

@router.get("/disk-info")
def get_disk_info(workspace_id: str = Depends(get_workspace)):
    """Get disk usage information."""
    with dataset_service.workspace(workspace_id) as ws:
        return ws.get_workspace_disk_state()

@router.get("/state")
def get_status(workspace_id: str = Depends(get_workspace)):
    """Get status."""
    with dataset_service.workspace(workspace_id) as ws:
        return ws.get_dataset_state()
