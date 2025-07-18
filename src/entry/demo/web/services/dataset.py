import io
import zipfile
from datetime import datetime
from pathlib import Path

from fastapi import UploadFile, HTTPException
from PIL import Image
from starlette.responses import StreamingResponse

from datasets import ActiveDataset
from entry.demo.web.config import settings
from entry.demo.web.models.requests import (ImageUploadRequest)
from entry.demo.web.models.responses import (DatasetExportResponse,
                                             ImageInfo, ImageUploadResponse)
from entry.demo.web.services.active_learning import active_learning_service


class DatasetService:
    def __init__(self):
        self.data_dir = settings.DATA_DIR
        self.annotations_dir = self.data_dir / "annotations"
        self.train_images_dir = self.data_dir / "train"
        self.pool_images_dir = self.data_dir / "pool"

        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure all necessary directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.train_images_dir.mkdir(parents=True, exist_ok=True)
        self.pool_images_dir.mkdir(parents=True, exist_ok=True)

    def validate_image_file(file: UploadFile) -> bool:
        """Validate if the uploaded file is an image"""
        if not file.content_type or not file.content_type.startswith('image/'):
            return False

        if file.filename:
            file_extension = Path(file.filename).suffix.lower()
            return file_extension in settings.ALLOWED_IMAGE_EXTENSIONS

        return False

    async def upload_images(self, request: ImageUploadRequest) -> ImageUploadResponse:
        """Upload multiple images to the dataset."""
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

                    data_dir = self.train_images_dir
                    if request.type == "pool":
                        data_dir = self.pool_images_dir

                    image_path = data_dir / unique_filename

                    with open(image_path, 'wb') as f:
                        f.write(image_content)

                    if request.type == "pool":
                        active_learning_service.current_pool_set.append(str(image_path))
                    else:
                        active_learning_service.current_train_set.append(str(image_path))

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
                    message=f"All {successful_count} images uploaded successfully",
                    uploaded_images=successful_uploads
                )
            elif successful_count > 0:
                return ImageUploadResponse(
                    success=True,
                    message=f"{successful_count} of {total_files} images uploaded successfully",
                    uploaded_images=successful_uploads,
                    failed_uploads=failed_uploads
                )
            else:
                return ImageUploadResponse(
                    success=False,
                    message="No images were uploaded successfully",
                    failed_uploads=failed_uploads
                )

        except Exception as e:
            return ImageUploadResponse(
                success=False,
                message=f"Failed to upload images: {str(e)}"
            )

    async def export_dataset(self, use_memory: bool = False) -> DatasetExportResponse:
        """Export dataset with annotations to a downloadable format."""
        annotated_count = active_learning_service.get_state().annotated_count
        if not annotated_count:
            raise HTTPException(status_code=404, detail="No annotated samples available")
        annotated_set = active_learning_service.get_annotated_set()

        try:
            self.annotations_dir.mkdir(exist_ok=True, parents=True)
            images_dir = self.annotations_dir / "images"
            labels_dir = self.annotations_dir / "labels"
            images_dir.mkdir(exist_ok=True, parents=True)
            labels_dir.mkdir(exist_ok=True, parents=True)

            zip_file = self.annotations_dir / f"dataset_{annotated_count}_samples.zip"

            with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as archive:
                for sample in annotated_set:
                    case_name = Path(sample["path"]).stem
                    image_np = sample["image"]
                    label_np = sample["mask"]

                    image_pil = Image.fromarray(image_np)
                    label_pil = Image.fromarray(label_np)

                    image_path = images_dir / f"{case_name}.png"
                    label_path = labels_dir / f"{case_name}.png"

                    image_pil.save(image_path)
                    label_pil.save(label_path)

                    archive.write(image_path, arcname=f"images/{case_name}.png")
                    archive.write(label_path, arcname=f"labels/{case_name}.png")

            file_size = zip_file.stat().st_size if zip_file.exists() else None

            return DatasetExportResponse(
                export_path=zip_file,
                file_size=file_size,
                sample_count=annotated_count,
                export_format="zip"
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

    async def create_streaming_response(self, export_response: DatasetExportResponse) -> StreamingResponse:
        """Create a streaming response for large files."""
        if not export_response.export_path.exists():
            raise HTTPException(status_code=404, detail="Export file not found")

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

    def clear(self):
        """Clear all images from the dataset."""
        try:
            for image_path in self.train_images_dir.glob("*"):
                image_path.unlink()
            for image_path in self.pool_images_dir.glob("*"):
                image_path.unlink()
            for image_path in self.annotations_dir.glob("*"):
                image_path.unlink()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to clear dataset: {str(e)}")
        else:
            return {"message": "Dataset cleared successfully"}


# Global service instance
dataset_service = DatasetService()
