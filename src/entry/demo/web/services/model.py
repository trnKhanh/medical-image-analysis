import base64
from datetime import datetime

from entry.demo.web.config import settings
from entry.demo.web.core.foundation_model import FoundationModelManager
from entry.demo.web.core.specialist_model import SpecialistModelManager
from entry.demo.web.models.requests import ModelCheckpointRequest
from entry.demo.web.models.responses import (ModelCheckpointInfo,
                                             ModelCheckpointListResponse,
                                             ModelCheckpointResponse)


class ModelService:
    def __init__(self):
        self.foundation_model_manager = FoundationModelManager()
        self.specialist_model_manager = SpecialistModelManager()
        self.models_dir = settings.MODELS_DIR
        self._ensure_models_directory()

    def _ensure_models_directory(self):
        """Ensure models directory exists."""
        self.models_dir.mkdir(parents=True, exist_ok=True)

    async def list_model_checkpoints(self) -> ModelCheckpointListResponse:
        """List all available model checkpoints."""
        try:
            model_infos = []

            for model_path in self.models_dir.glob("*.pth"):
                try:
                    stat = model_path.stat()

                    model_info = ModelCheckpointInfo(
                        name=model_path.name,
                        size=stat.st_size,
                        description=f"Model checkpoint: {model_path.name}",
                        created_at=datetime.fromtimestamp(stat.st_ctime),
                    )
                    model_infos.append(model_info)
                except Exception as e:
                    print(f"Error processing {model_path}: {e}")
                    continue

            model_infos.sort(key=lambda x: x.created_at, reverse=True)

            return ModelCheckpointListResponse(
                message="Model checkpoints retrieved successfully",
                models=model_infos,
                total_count=len(model_infos)
            )
        except Exception as e:
            return ModelCheckpointListResponse(
                success=False,
                message=f"Failed to list model checkpoints: {str(e)}",
                models=[],
                total_count=0
            )

    async def upload_model_checkpoint(self, request: ModelCheckpointRequest) -> ModelCheckpointResponse:
        """Upload a new model checkpoint."""
        try:
            model_content = base64.b64decode(request.file_content)

            model_filename = f"{request.name}.pth" if not request.name.endswith('.pth') else request.name
            model_path = self.models_dir / model_filename

            if model_path.exists():
                return ModelCheckpointResponse(
                    success=False,
                    message=f"Model checkpoint '{model_filename}' already exists"
                )

            # Save a model file
            with open(model_path, 'wb') as f:
                f.write(model_content)

            # Create model info
            model_info = ModelCheckpointInfo(
                name=model_filename,
                size=len(model_content),
                description=request.description or f"Uploaded model: {model_filename}",
                created_at=datetime.now(),
            )

            return ModelCheckpointResponse(
                message="Model checkpoint uploaded successfully",
                model_info=model_info
            )

        except Exception as e:
            return ModelCheckpointResponse(
                success=False,
                message=f"Failed to upload model checkpoint: {str(e)}"
            )

    async def delete_model_checkpoint(self, model_name: str) -> ModelCheckpointResponse:
        """Delete a model checkpoint."""
        try:
            model_path = self.models_dir / model_name

            if not model_path.exists():
                return ModelCheckpointResponse(
                    success=False,
                    message=f"Model checkpoint '{model_name}' not found"
                )

            # Check if it's the default model
            if str(model_path) == settings.DEFAULT_SPECIALIST_MODEL:
                return ModelCheckpointResponse(
                    success=False,
                    message="Cannot delete the default model checkpoint"
                )

            # Remove a file
            model_path.unlink()

            return ModelCheckpointResponse(
                message=f"Model checkpoint '{model_name}' deleted successfully"
            )

        except Exception as e:
            return ModelCheckpointResponse(
                success=False,
                message=f"Failed to delete model checkpoint: {str(e)}"
            )

    async def validate_model_checkpoint(self, model_name: str) -> ModelCheckpointResponse:
        """Validate a model checkpoint by attempting to load it."""
        try:
            model_path = self.models_dir / model_name

            if not model_path.exists():
                return ModelCheckpointResponse(
                    success=False,
                    message=f"Model checkpoint '{model_name}' not found"
                )

            validation_result = await self.specialist_model_manager.validate_model(str(model_path))

            if validation_result:
                return ModelCheckpointResponse(
                    message=f"Model checkpoint '{model_name}' is valid"
                )
            else:
                return ModelCheckpointResponse(
                    success=False,
                    message=f"Model checkpoint '{model_name}' is invalid or corrupted"
                )

        except Exception as e:
            return ModelCheckpointResponse(
                success=False,
                message=f"Failed to validate model checkpoint: {str(e)}"
            )


# Global service instance
model_service = ModelService()
