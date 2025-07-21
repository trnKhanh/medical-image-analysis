
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Final
from logging import getLogger

import torch
from torch.utils.data import ConcatDataset, DataLoader
import torchvision.transforms.functional as F

from activelearning import KMeanSelector
from datasets import ActiveDataset, ExtendableDataset, ImageDataset
from entry.demo.web.config import ActiveLearningConfig, settings
from entry.demo.web.core.foundation_model import FoundationModelManager
from entry.demo.web.core.specialist_model import SpecialistModelManager
from entry.demo.web.models.requests import (ActiveLearningConfigRequest)
from entry.demo.web.models.responses import (ActiveLearningConfigResponse, ActiveLearningStateResponse)

Logger: Final = getLogger(__name__)

class ActiveLearningService:
    def __init__(self):
        self.config = ActiveLearningConfig()
        self.feature_dict: Optional[Dict[str, torch.Tensor]] = None
        self.foundation_model_manager = FoundationModelManager()
        self.specialist_model_manager = SpecialistModelManager()
        self.executor = ThreadPoolExecutor(max_workers=4)

        self.current_train_set: List[str] = []
        self.current_pool_set: List[str] = []
        self.annotated_set: List[str] = []
        self.selected_images = []
        self.selected_set = []

        self.foundation_model = FoundationModelManager()
        self.specialist_model = SpecialistModelManager()

    async def get_feature_dict(self, batch_size, device, active_dataset: ActiveDataset):
        dataset = ConcatDataset([active_dataset.get_train_dataset(), active_dataset.get_pool_dataset()])
        dataloader = DataLoader(dataset, batch_size=batch_size)
        model, preprocess, _ = await self.foundation_model.initialize(device)

        feature_dict = {}
        if model is None:
            print("model is none")
            return feature_dict
        if preprocess is None:
            print("preprocess is none")
        for sampled_batch in dataloader:
            image_batch = sampled_batch["image"]
            image_list = []
            for image in image_batch:
                image_pil = F.to_pil_image(image).convert("RGB")
                image_list.append(preprocess(image_pil))
            image_batch = torch.stack(image_list, dim=0)
            image_batch = image_batch.to(device)

            with torch.no_grad():
                feature_batch = model.encode_image(image_batch)

            for i in range(len(feature_batch)):
                case_name = sampled_batch["case_name"][i]
                feature_dict[case_name] = feature_batch[i]

        return feature_dict

    def get_state(self) -> ActiveLearningStateResponse:
        return ActiveLearningStateResponse(
            train_count=len(self.current_train_set),
            pool_count=len(self.current_pool_set),
            annotated_count=len(self.annotated_set)
        )

    def get_annotated_set(self) -> List[str]:
        return self.annotated_set

    def get_train_set(self) -> List[str]:
        return self.current_train_set

    def get_pool_set(self) -> List[str]:
        return self.current_pool_set

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

            self.feature_dict = None

            return ActiveLearningConfigResponse(
                message="Configuration updated successfully",
                config=config_request.model_dump()
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

    async def active_select(self, train_set, pool_set, budget, model_ckpt, batch_size, device,
                      loaded_feature_weight, sharp_factor, loaded_feature_only):
        train_dataset = ExtendableDataset(ImageDataset(train_set, image_channels=1, image_size=settings.IMAGE_SIZE))
        pool_dataset = ExtendableDataset(ImageDataset(pool_set, image_channels=1, image_size=settings.IMAGE_SIZE))

        Logger.info(f"train_dataset: {len(train_dataset)}")
        Logger.info(f"pool_dataset: {len(pool_dataset)}")
        active_dataset = ActiveDataset(train_dataset, pool_dataset)

        if self.feature_dict is None:
            self.feature_dict = await self.get_feature_dict(batch_size, device, active_dataset)

        active_selector = KMeanSelector(
            batch_size=4,
            num_workers=1,
            pin_memory=True,
            metric="l2",
            feature_dict=self.feature_dict,
            loaded_feature_weight=loaded_feature_weight,
            sharp_factor=sharp_factor,
            loaded_feature_only=loaded_feature_only,
        )
        await self.specialist_model.load_model(str(settings.MODELS_DIR) + "/" + model_ckpt)
        return active_selector.select_next_batch(active_dataset, budget, self.specialist_model.model, device)

    def predict_pseudo_label(self, image_pil):
        image = F.to_tensor(image_pil)
        image = image.unsqueeze(0)
        _, _, H, W = image.shape
        image = self.specialist_model.processor.preprocess(image)
        with torch.no_grad():
            pred = self.specialist_model.model(image)
            pseudo_label = pred.argmax(1)
        pseudo_label = self.specialist_model.processor.postprocess(pseudo_label, [H, W])
        return pseudo_label[0]

    def reset_feature_cache(self):
        """Reset the feature cache."""
        self.feature_dict = None

    def clear(self):
        """Clear current session data."""
        self.current_train_set.clear()
        self.current_pool_set.clear()
        self.selected_images.clear()
        self.feature_dict = None


# Global service instance
active_learning_service = ActiveLearningService()

