import asyncio
import base64
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from entry.demo.web.config import ActiveLearningConfig, settings
from entry.demo.web.core.foundation_model import FoundationModelManager
from entry.demo.web.core.specialist_model import SpecialistModelManager
from entry.demo.web.models.requests import (ActiveLearningConfigRequest,
                                            ActiveSelectionRequest,
                                            PredictionRequest)
from entry.demo.web.models.responses import (ActiveLearningConfigResponse,
                                             ActiveSelectionResponse,
                                             PredictionResponse)


class ActiveLearningService:
    def __init__(self):
        self.config = ActiveLearningConfig()
        self.feature_dict: Optional[Dict[str, torch.Tensor]] = None
        self.foundation_model_manager = FoundationModelManager()
        self.specialist_model_manager = SpecialistModelManager()
        self.executor = ThreadPoolExecutor(max_workers=4)

        # In-memory storage for the current session
        self.current_train_set: List[str] = []
        self.current_pool_set: List[str] = []
        self.selected_images: List[str] = []

    async def update_config(self, config_request: ActiveLearningConfigRequest) -> ActiveLearningConfigResponse:
        """Update active learning configuration."""
        try:
            self.config.budget = config_request.budget
            self.config.model = config_request.model
            self.config.device = torch.device(config_request.device)
            self.config.batch_size = config_request.batch_size
            self.config.loaded_feature_weight = config_request.loaded_feature_weight
            self.config.sharp_factor = config_request.sharp_factor
            self.config.loaded_feature_only = config_request.loaded_feature_only
            self.config.model_ckpt = config_request.model_ckpt

            # Reset feature dict when config changes
            self.feature_dict = None

            return ActiveLearningConfigResponse(
                message="Configuration updated successfully",
                config=config_request.dict()
            )
        except Exception as e:
            return ActiveLearningConfigResponse(
                success=False,
                message=f"Failed to update configuration: {str(e)}"
            )

    async def get_config(self) -> ActiveLearningConfigResponse:
        """Get the current active learning configuration."""
        config_dict = {
            "budget": self.config.budget,
            "model": self.config.model,
            "device": str(self.config.device),
            "batch_size": self.config.batch_size,
            "loaded_feature_weight": self.config.loaded_feature_weight,
            "sharp_factor": self.config.sharp_factor,
            "loaded_feature_only": self.config.loaded_feature_only,
            "model_ckpt": self.config.model_ckpt
        }

        return ActiveLearningConfigResponse(
            message="Configuration retrieved successfully",
            config=config_dict
        )

    async def compute_features_simple(self, image_paths: List[str]) -> Dict[str, List[float]]:
        """Compute simple features for images (simplified version without an actual model)."""

        def _compute_features_sync():
            feature_dict = {}

            for image_path in image_paths:
                try:
                    # Simple feature extraction (for demo purposes)
                    # In real implementation, this would use the foundation model
                    image = Image.open(image_path).convert("RGB")

                    # Simple features: mean RGB values and image size
                    image_array = np.array(image)
                    mean_rgb = image_array.mean(axis=(0, 1)).tolist()
                    size_features = [image.width, image.height]

                    # Combine features
                    simple_features = mean_rgb + size_features

                    case_name = Path(image_path).stem
                    feature_dict[case_name] = simple_features

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue

            return feature_dict

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _compute_features_sync)

    async def select_next_batch_simple(self, request: ActiveSelectionRequest) -> ActiveSelectionResponse:
        """Perform simple active selection (without complex models)."""
        try:
            # Update configuration if provided
            if request.config:
                await self.update_config(request.config)

            # Store current sets
            self.current_train_set = request.train_set.copy()
            self.current_pool_set = request.pool_set.copy()

            # Simple selection strategy: random sampling with some intelligence
            import random

            def _select_sync():
                available_pool = [img for img in request.pool_set if img not in request.train_set]

                if len(available_pool) == 0:
                    return []

                # Simple selection: random choice from a pool
                num_to_select = min(self.config.budget, len(available_pool))
                selected = random.sample(available_pool, num_to_select)

                return selected

            loop = asyncio.get_event_loop()
            selected_images = await loop.run_in_executor(self.executor, _select_sync)

            # Store selected images
            self.selected_images = selected_images

            return ActiveSelectionResponse(
                message="Active selection completed successfully",
                selected_images=selected_images,
                selection_method="Random",
                budget=self.config.budget,
                total_pool_size=len(request.pool_set)
            )

        except Exception as e:
            return ActiveSelectionResponse(
                success=False,
                message=f"Active selection failed: {str(e)}",
                selected_images=[],
                selection_method="",
                budget=0,
                total_pool_size=0
            )

    async def predict_pseudo_label_simple(self, request: PredictionRequest) -> PredictionResponse:
        """Generate simple pseudo label for an image."""
        try:
            def _predict_sync():
                # Simple prediction: generate random mask
                image = Image.open(request.image_path).convert("L")
                width, height = image.size

                # Create simple pseudo mask (for demo)
                mask = np.random.randint(0, 3, (height, width), dtype=np.uint8)

                # Convert mask to image
                mask_image = Image.fromarray(mask * 85)  # Scale to visible range

                # Convert to base64
                buffer = io.BytesIO()
                mask_image.save(buffer, format='PNG')
                mask_b64 = base64.b64encode(buffer.getvalue()).decode()

                # Create visualization (simple overlay)
                visual_image = image.convert("RGB")
                visual_buffer = io.BytesIO()
                visual_image.save(visual_buffer, format='PNG')
                visual_b64 = base64.b64encode(visual_buffer.getvalue()).decode()

                return mask_b64, visual_b64

            loop = asyncio.get_event_loop()
            mask_b64, visual_b64 = await loop.run_in_executor(self.executor, _predict_sync)

            return PredictionResponse(
                message="Prediction generated successfully",
                image_path=request.image_path,
                prediction_mask=mask_b64,
                prediction_visual=visual_b64,
                model_used=request.model_ckpt or "simple_model"
            )

        except Exception as e:
            return PredictionResponse(
                success=False,
                message=f"Prediction failed: {str(e)}",
                image_path="",
                prediction_mask="",
                prediction_visual="",
                model_used=""
            )

    def reset_feature_cache(self):
        """Reset the feature cache."""
        self.feature_dict = None

    def get_current_session_data(self) -> Dict[str, Any]:
        """Get current session data."""
        return {
            "train_set": self.current_train_set,
            "pool_set": self.current_pool_set,
            "selected_images": self.selected_images,
            "config": {
                "budget": self.config.budget,
                "model": self.config.model,
                "device": str(self.config.device),
                "batch_size": self.config.batch_size
            }
        }

    def clear_session_data(self):
        """Clear current session data."""
        self.current_train_set.clear()
        self.current_pool_set.clear()
        self.selected_images.clear()
        self.feature_dict = None


# Global service instance
active_learning_service = ActiveLearningService()
