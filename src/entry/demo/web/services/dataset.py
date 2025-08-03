import io
import os
import shutil
import threading
import zipfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator

from fastapi import HTTPException, UploadFile
from PIL import Image
from starlette.responses import StreamingResponse
from starlette.status import HTTP_404_NOT_FOUND, HTTP_500_INTERNAL_SERVER_ERROR

from entry.demo.web.config import settings
from entry.demo.web.models.requests import ImageUploadRequest
from entry.demo.web.models.responses import (DatasetExportResponse, ImageInfo,
                                             ImageUploadResponse)
from entry.demo.web.services.active_learning import active_learning_service
from entry.demo.web.services.context.dataset import DatasetContext


async def create_streaming_response(export_response: DatasetExportResponse) -> StreamingResponse:
    """Create a streaming response for large files."""
    if not export_response.export_path.exists():
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Export file not found")

    def file_generator():
        with open(export_response.export_path, "rb") as f:
            while chunk := f.read(8192):
                yield chunk

    return StreamingResponse(
        file_generator(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=annotated_dataset_{export_response.sample_count}_samples.zip"
        }
    )

class DatasetState:
    def __init__(self, workspace_id: str):
        self.workspace_id = workspace_id
        self.base_data_dir = settings.DATA_DIR
        self.data_dir = self.base_data_dir / "workspaces" / workspace_id
        self.annotations_dir = self.data_dir / "annotations"
        self.annotated_image_dir = self.annotations_dir / "images"
        self.annotated_label_dir = self.annotations_dir / "labels"
        self.train_images_dir = self.data_dir / "train"
        self.pool_images_dir = self.data_dir / "pool"
        self.data_archive_dir = self.data_dir / "archives"

        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure all necessary directories exist."""
        directories = [
            self.data_dir,
            self.annotations_dir,
            self.annotated_image_dir,
            self.annotated_label_dir,
            self.train_images_dir,
            self.pool_images_dir,
            self.data_archive_dir
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


class DatasetService:
    """Main dataset service managing multiple workspace directories."""

    def __init__(self):
        self._workspace_dirs: Dict[str, DatasetState] = {}
        self._lock = threading.RLock()

    def _get_workspace_dirs(self, workspace_id: str) -> DatasetState:
        """Get or create workspace directories."""
        with self._lock:
            if workspace_id not in self._workspace_dirs:
                self._workspace_dirs[workspace_id] = DatasetState(workspace_id)
                print(f"Created workspace directories for: {workspace_id}")
            return self._workspace_dirs[workspace_id]

    @staticmethod
    def validate_image_file(file: UploadFile) -> bool:
        """Validate if the uploaded file is an image"""
        if not file.content_type or not file.content_type.startswith('image/'):
            return False

        if file.filename:
            file_extension = Path(file.filename).suffix.lower()
            return file_extension in settings.ALLOWED_IMAGE_EXTENSIONS

        return False

    def save_annotated_image(self, workspace_id: str, annotated_image) -> str:
        """Save an annotated image to the workspace dataset."""
        dirs = self._get_workspace_dirs(workspace_id)

        image_path = dirs.annotated_image_dir / f"{annotated_image['case_name']}.png"
        label_path = dirs.annotated_label_dir / f"{annotated_image['case_name']}.png"

        image_pil = annotated_image["image"]
        label_pil = Image.fromarray(annotated_image["mask"])

        image_pil.save(image_path)
        label_pil.save(label_path)

        return str(image_path)

    async def upload_images(self, workspace_id: str, request: ImageUploadRequest) -> ImageUploadResponse:
        """Upload multiple images to the workspace dataset."""
        dirs = self._get_workspace_dirs(workspace_id)

        try:
            successful_uploads = []
            failed_uploads = []

            for uploaded_file in request.images:
                try:
                    image_content = await uploaded_file.read()

                    try:
                        image = Image.open(io.BytesIO(image_content))
                        image_size = image.size
                    except Exception as e:
                        failed_uploads.append({
                            "filename": uploaded_file.filename,
                            "error": f"Invalid image content: {str(e)}"
                        })
                        continue

                    file_extension = Path(uploaded_file.filename).suffix.lower()
                    if file_extension not in settings.ALLOWED_IMAGE_EXTENSIONS:
                        failed_uploads.append({
                            "filename": uploaded_file.filename,
                            "error": f"Unsupported image format: {file_extension}"
                        })
                        continue

                    if len(image_content) > settings.MAX_UPLOAD_SIZE:
                        failed_uploads.append({
                            "filename": uploaded_file.filename,
                            "error": "Image file too large"
                        })
                        continue

                    case_name = Path(uploaded_file.filename).stem
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    unique_filename = f"{case_name}_{timestamp}{file_extension}"

                    data_dir = dirs.train_images_dir
                    if request.type == "pool":
                        data_dir = dirs.pool_images_dir

                    image_path = data_dir / unique_filename

                    with open(image_path, 'wb') as f:
                        f.write(image_content)

                    if request.type == "pool":
                        pool_set = active_learning_service.get_pool_set(workspace_id)
                        pool_set.append(str(image_path))
                    else:
                        train_set = active_learning_service.get_train_set(workspace_id)
                        train_set.append(str(image_path))

                    image_info = ImageInfo(
                        filename=uploaded_file.filename,
                        path=str(image_path),
                        size=image_size,
                        case_name=case_name,
                        created_at=datetime.now()
                    )

                    successful_uploads.append(image_info)

                except Exception as e:
                    failed_uploads.append({
                        "filename": uploaded_file.filename,
                        "error": f"Failed to process image: {str(e)}"
                    })

            total_files = len(request.images)
            successful_count = len(successful_uploads)
            failed_count = len(failed_uploads)

            if successful_count == total_files:
                return ImageUploadResponse(
                    success=True,
                    message=f"All {successful_count} images uploaded successfully to workspace {workspace_id}",
                    uploaded_images=successful_uploads
                )
            elif successful_count > 0:
                return ImageUploadResponse(
                    success=True,
                    message=f"{successful_count} of {total_files} images uploaded successfully to workspace {workspace_id}",
                    uploaded_images=successful_uploads,
                    failed_uploads=failed_uploads
                )
            else:
                return ImageUploadResponse(
                    success=False,
                    message=f"No images were uploaded successfully to workspace {workspace_id}",
                    failed_uploads=failed_uploads
                )

        except Exception as e:
            return ImageUploadResponse(
                success=False,
                message=f"Failed to upload images to workspace {workspace_id}: {str(e)}"
            )

    async def export_dataset(self, workspace_id: str, use_memory: bool = False) -> DatasetExportResponse:
        """Export dataset with annotations to a downloadable format for specific workspace."""
        dirs = self._get_workspace_dirs(workspace_id)
        with active_learning_service.workspace(workspace_id) as ws:
            annotated_count = ws.get_state().annotated_count
            if not annotated_count:
                raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail=f"No annotated samples available in workspace {workspace_id}")

        try:
            dirs.annotations_dir.mkdir(exist_ok=True, parents=True)
            dirs.annotated_image_dir.mkdir(exist_ok=True, parents=True)
            dirs.annotated_label_dir.mkdir(exist_ok=True, parents=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            zip_file = dirs.data_archive_dir / f"dataset_{workspace_id}_{annotated_count}_samples_{timestamp}.zip"

            with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as archive:
                for root, dirs_list, files in os.walk(dirs.annotations_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, start=dirs.annotations_dir)
                        archive.write(file_path, arcname)

            file_size = zip_file.stat().st_size if zip_file.exists() else None

            return DatasetExportResponse(
                export_path=zip_file,
                file_size=file_size,
                sample_count=annotated_count,
                export_format="zip"
            )

        except Exception as e:
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Export failed for workspace {workspace_id}: {str(e)}"
            )

    def get_workspace_stats(self, workspace_id: str) -> dict:
        """Get statistics for workspace."""
        dirs = self._get_workspace_dirs(workspace_id)

        stats = {}
        directory_mapping = {
            'train_images': dirs.train_images_dir,
            'pool_images': dirs.pool_images_dir,
            'annotated_images': dirs.annotated_image_dir,
            'annotated_labels': dirs.annotated_label_dir,
            'archives': dirs.data_archive_dir
        }

        for name, dir_path in directory_mapping.items():
            if dir_path.exists():
                file_count = len([f for f in dir_path.iterdir() if f.is_file()])
                stats[f'{name}_count'] = file_count
            else:
                stats[f'{name}_count'] = 0

        return stats

    def list_images(self, workspace_id: str, image_type: str = "all") -> dict:
        """List images in workspace by type."""
        dirs = self._get_workspace_dirs(workspace_id)

        result = {}

        if image_type in ["all", "train"]:
            train_images = [f.name for f in dirs.train_images_dir.glob("*") if f.is_file()]
            result["train_images"] = train_images

        if image_type in ["all", "pool"]:
            pool_images = [f.name for f in dirs.pool_images_dir.glob("*") if f.is_file()]
            result["pool_images"] = pool_images

        if image_type in ["all", "annotated"]:
            annotated_images = [f.name for f in dirs.annotated_image_dir.glob("*") if f.is_file()]
            result["annotated_images"] = annotated_images

        return result

    def get_workspace_disk_state(self, workspace_id: str) -> dict:
        """Get disk usage for workspace."""
        dirs = self._get_workspace_dirs(workspace_id)

        if not dirs.data_dir.exists():
            return {'total_size': 0, 'file_count': 0}

        total_size = 0
        file_count = 0

        for file_path in dirs.data_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1

        return {
            'total_size': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'file_count': file_count,
            'max_size': (100 * 1024 * 1024),
            'max_size_mb': 100,
            'usage_percent': total_size / (100 * 1024 * 1024) if total_size < (100 * 1024 * 1024) else 100,
        }

    def clear(self, workspace_id: str):
        """Clear all images from the workspace dataset."""
        dirs = self._get_workspace_dirs(workspace_id)

        try:
            for image_path in dirs.train_images_dir.glob("*"):
                if image_path.is_file():
                    image_path.unlink()

            for image_path in dirs.pool_images_dir.glob("*"):
                if image_path.is_file():
                    image_path.unlink()

            if dirs.annotated_label_dir.exists():
                shutil.rmtree(dirs.annotated_label_dir, ignore_errors=True)
            if dirs.annotated_image_dir.exists():
                shutil.rmtree(dirs.annotated_image_dir, ignore_errors=True)

            for item in dirs.data_archive_dir.glob("*"):
                if item.is_file():
                    item.unlink()

            dirs._ensure_directories()

        except Exception as e:
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to clear dataset for workspace {workspace_id}: {str(e)}"
            )

    def delete_workspace(self, workspace_id: str):
        """Delete the entire workspace and its data."""
        with self._lock:
            if workspace_id in self._workspace_dirs:
                dirs = self._workspace_dirs[workspace_id]

                try:
                    # Remove the entire workspace directory
                    if dirs.data_dir.exists():
                        shutil.rmtree(dirs.data_dir)
                        print(f"Deleted workspace directory: {dirs.data_dir}")

                    # Remove from cache
                    del self._workspace_dirs[workspace_id]
                    print(f"Deleted workspace: {workspace_id}")

                except Exception as e:
                    raise HTTPException(
                        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to delete workspace {workspace_id}: {str(e)}"
                    )

    def get_all_workspace_stats(self) -> Dict[str, dict]:
        """Get stats for all workspaces."""
        with self._lock:
            stats = {}
            for workspace_id in self._workspace_dirs.keys():
                stats[workspace_id] = self.get_workspace_stats(workspace_id)
            return stats

    @contextmanager
    def workspace(self, workspace_id: str) -> Generator['DatasetContext', None, None]:
        """Context manager for workspace operations."""
        yield DatasetContext(self, workspace_id)

# Global service instance
dataset_service = DatasetService()
