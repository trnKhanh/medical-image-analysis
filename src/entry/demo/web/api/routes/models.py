import base64

from fastapi import APIRouter, File, Form, UploadFile

from entry.demo.web.models.requests import ModelCheckpointRequest
from entry.demo.web.models.responses import (ModelCheckpointListResponse,
                                             ModelCheckpointResponse)
from entry.demo.web.services.model import model_service

router = APIRouter()


@router.get("/checkpoints", response_model=ModelCheckpointListResponse)
async def list_model_checkpoints():
    """List all available model checkpoints."""
    return await model_service.list_model_checkpoints()

@router.post("/checkpoints", response_model=ModelCheckpointResponse)
async def upload_model_checkpoint(request: ModelCheckpointRequest):
    """Upload a new model checkpoint."""
    return await model_service.upload_model_checkpoint(request)


@router.post("/checkpoints/upload-file", response_model=ModelCheckpointResponse)
async def upload_model_checkpoint_file(
        file: UploadFile = File(...),
        name: str = Form(...),
        description: str = Form("")
):
    """Upload a new model checkpoint from file."""
    try:
        content = await file.read()

        file_content_b64 = base64.b64encode(content).decode()

        request = ModelCheckpointRequest(
            name=name,
            description=description,
            file_content=file_content_b64
        )

        return await model_service.upload_model_checkpoint(request)
    except Exception as e:
        return ModelCheckpointResponse(
            success=False,
            message=f"Failed to upload file: {str(e)}"
        )


@router.delete("/checkpoints/{model_name}", response_model=ModelCheckpointResponse)
async def delete_model_checkpoint(model_name: str):
    """Delete a model checkpoint."""
    return await model_service.delete_model_checkpoint(model_name)


@router.post("/checkpoints/{model_name}/validate", response_model=ModelCheckpointResponse)
async def validate_model_checkpoint(model_name: str):
    """Validate a model checkpoint by attempting to load it."""
    return await model_service.validate_model_checkpoint(model_name)
