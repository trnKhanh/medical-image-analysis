
import base64
import io
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from fastapi import UploadFile
from PIL import Image

from entry.demo.web.config import settings
from entry.demo.web.models.requests import (AnnotationRequest,
                                            DatasetExportRequest,
                                            DatasetRequest, ImageUploadRequest)
from entry.demo.web.models.responses import (AnnotationInfo,
                                             AnnotationResponse,
                                             DatasetExportResponse,
                                             DatasetInfo, DatasetResponse,
                                             ImageInfo, ImageUploadResponse)


class DatasetService:
    def __init__(self):
        self.datasets_dir = settings.DATASETS_DIR
        self.annotations_dir = self.datasets_dir / "annotations"
        self.train_images_dir = self.datasets_dir / "train"
        self.pool_images_dir = self.datasets_dir / "pool"
        self.exports_dir = self.datasets_dir / "exports"

        # In-memory storage
        self.current_images: List[ImageInfo] = []
        self.current_datasets: Dict[str, DatasetInfo] = {}
        self.current_annotations: List[AnnotationInfo] = []

        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure all necessary directories exist."""
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.train_images_dir.mkdir(parents=True, exist_ok=True)
        self.pool_images_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    def validate_image_file(file: UploadFile) -> bool:
        """Validate if the uploaded file is an image"""
        if not file.content_type or not file.content_type.startswith('image/'):
            return False

        if file.filename:
            file_extension = Path(file.filename).suffix.lower()
            return file_extension in settings.ALLOWED_IMAGE_EXTENSIONS

        return False

    async def upload_image(self, request: ImageUploadRequest) -> ImageUploadResponse:
        """Upload a single image to the dataset."""
        try:
            image_content = base64.b64decode(request.content)

            # Validate image
            try:
                image = Image.open(io.BytesIO(image_content))
                image_size = image.size
            except Exception as e:
                return ImageUploadResponse(
                    success=False,
                    message=f"Invalid image content: {str(e)}"
                )

            file_extension = Path(request.filename).suffix.lower()
            if file_extension not in settings.ALLOWED_IMAGE_EXTENSIONS:
                return ImageUploadResponse(
                    success=False,
                    message=f"Unsupported image format: {file_extension}"
                )

            case_name = request.case_name or Path(request.filename).stem
            image_path = self.images_dir / f"{case_name}{file_extension}"

            # Check file size
            if len(image_content) > settings.MAX_UPLOAD_SIZE:
                return ImageUploadResponse(
                    success=False,
                    message="Image file too large"
                )

            # Save image file
            with open(image_path, 'wb') as f:
                f.write(image_content)

            # Create image info and store in memory
            image_info = ImageInfo(
                filename=request.filename,
                path=str(image_path),
                size=image_size,
                case_name=case_name,
                created_at=datetime.now()
            )

            # Add to in-memory storage
            self.current_images.append(image_info)

            return ImageUploadResponse(
                message="Image uploaded successfully",
                image_info=image_info
            )

        except Exception as e:
            return ImageUploadResponse(
                success=False,
                message=f"Failed to upload image: {str(e)}"
            )

    async def upload_images_batch(self, requests: List[ImageUploadRequest]) -> List[ImageUploadResponse]:
        """Upload multiple images in a batch."""
        responses = []
        for request in requests:
            response = await self.upload_image(request)
            responses.append(response)
        return responses

    async def create_dataset(self, request: DatasetRequest, dataset_name: str) -> DatasetResponse:
        """Create a dataset with train and pool splits."""
        try:
            # Validate image paths exist
            all_images = request.train_images + request.pool_images
            for image_path in all_images:
                if not Path(image_path).exists():
                    return DatasetResponse(
                        success=False,
                        message=f"Image not found: {image_path}"
                    )

            # Create dataset info and store in memory
            dataset_info = DatasetInfo(
                name=dataset_name,
                train_count=len(request.train_images),
                pool_count=len(request.pool_images),
                total_count=len(all_images),
                created_at=datetime.now()
            )

            # Store in memory
            self.current_datasets[dataset_name] = dataset_info

            return DatasetResponse(
                message="Dataset created successfully",
                dataset_info=dataset_info
            )

        except Exception as e:
            return DatasetResponse(
                success=False,
                message=f"Failed to create dataset: {str(e)}"
            )

    async def get_dataset_info(self, dataset_name: str) -> DatasetResponse:
        """Get information about a specific dataset."""
        try:
            if dataset_name not in self.current_datasets:
                return DatasetResponse(
                    success=False,
                    message=f"Dataset '{dataset_name}' not found"
                )

            dataset_info = self.current_datasets[dataset_name]

            return DatasetResponse(
                message="Dataset information retrieved successfully",
                dataset_info=dataset_info
            )

        except Exception as e:
            return DatasetResponse(
                success=False,
                message=f"Failed to get dataset info: {str(e)}"
            )

    async def list_datasets(self) -> List[DatasetInfo]:
        """List all available datasets."""
        try:
            datasets = list(self.current_datasets.values())
            # Sort by creation date (newest first)
            datasets.sort(key=lambda x: x.created_at, reverse=True)
            return datasets
        except Exception as e:
            return []

    async def save_annotation(self, request: AnnotationRequest) -> AnnotationResponse:
        """Save an annotation for an image."""
        try:
            # Decode mask data
            mask_content = base64.b64decode(request.mask_data)

            # Create paths
            case_name = request.case_name or Path(request.image_path).stem
            mask_path = self.annotations_dir / f"{case_name}_mask.png"

            # Save mask as binary file
            with open(mask_path, 'wb') as f:
                f.write(mask_content)

            # Create annotation info and store in memory
            annotation_info = AnnotationInfo(
                image_path=request.image_path,
                mask_path=str(mask_path),
                case_name=case_name,
                annotated_at=datetime.now()
            )

            # Add to in-memory storage
            self.current_annotations.append(annotation_info)

            return AnnotationResponse(
                message="Annotation saved successfully",
                annotation_info=annotation_info
            )

        except Exception as e:
            return AnnotationResponse(
                success=False,
                message=f"Failed to save annotation: {str(e)}"
            )

    async def export_dataset(self, request: DatasetExportRequest) -> DatasetExportResponse:
        """Export dataset with annotations to a downloadable format."""
        try:
            # Create export file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_filename = f"{request.dataset_name}_{timestamp}.{request.format}"
            export_path = self.exports_dir / export_filename

            included_items = {"images": 0, "annotations": 0}

            if request.format == "zip":
                with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Add dataset structure
                    zip_file.writestr(f"{request.dataset_name}/", "")
                    zip_file.writestr(f"{request.dataset_name}/images/", "")
                    if request.include_annotations:
                        zip_file.writestr(f"{request.dataset_name}/labels/", "")

                    # Add images and annotations from in-memory storage
                    for annotation in self.current_annotations:
                        image_path = Path(annotation.image_path)
                        if image_path.exists():
                            zip_file.write(
                                image_path,
                                f"{request.dataset_name}/images/{image_path.name}"
                            )
                            included_items["images"] += 1

                        if request.include_annotations:
                            mask_path = Path(annotation.mask_path)
                            if mask_path.exists():
                                zip_file.write(
                                    mask_path,
                                    f"{request.dataset_name}/labels/{mask_path.name}"
                                )
                                included_items["annotations"] += 1

            # Get file size
            file_size = export_path.stat().st_size if export_path.exists() else 0

            return DatasetExportResponse(
                message="Dataset exported successfully",
                export_path=str(export_path),
                export_format=request.format,
                file_size=file_size,
                included_items=included_items
            )

        except Exception as e:
            return DatasetExportResponse(
                success=False,
                message=f"Failed to export dataset: {str(e)}",
                export_path="",
                export_format="",
                file_size=0,
                included_items={}
            )

    async def get_image_list(self) -> List[ImageInfo]:
        """Get a list of all uploaded images."""
        try:
            # Return from in-memory storage
            images = self.current_images.copy()
            # Sort by upload date (newest first)
            images.sort(key=lambda x: x.created_at, reverse=True)
            return images
        except Exception as e:
            return []

    def clear_data(self):
        """Clear all in-memory data."""
        self.current_images.clear()
        self.current_datasets.clear()
        self.current_annotations.clear()


# Global service instance
dataset_service = DatasetService()
