from entry.demo.web.models.requests import ImageUploadRequest
from entry.demo.web.models.responses import (DatasetExportResponse,
                                             ImageUploadResponse)
from entry.demo.web.services.interfaces.dataset import DatasetProtocol


class DatasetContext:
    """Context wrapper for clean workspace-specific API."""

    def __init__(self, service: DatasetProtocol, workspace_id: str):
        self._service = service
        self._workspace_id = workspace_id

    def save_annotated_image(self, annotated_image) -> str:
        return self._service.save_annotated_image(self._workspace_id, annotated_image)

    async def upload_images(self, request: ImageUploadRequest) -> ImageUploadResponse:
        return await self._service.upload_images(self._workspace_id, request)

    async def export_dataset(self, use_memory: bool = False) -> DatasetExportResponse:
        return await self._service.export_dataset(self._workspace_id, use_memory)

    def get_dataset_state(self) -> dict:
        return self._service.get_dataset_state(self._workspace_id)

    def get_size(self) -> int:
        return self._service.get_workspace_disk_state(self._workspace_id)

    def clear(self):
        return self._service.clear(self._workspace_id)

    def get_workspace_disk_state(self):
        return self._service.get_workspace_disk_state(self._workspace_id)

    @property
    def workspace_id(self) -> str:
        return self._workspace_id
