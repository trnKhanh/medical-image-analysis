import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from logging import getLogger
from typing import Any, Coroutine, Dict, Final, Generator, List, Optional

import torch
import torchvision.transforms.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from activelearning import KMeanSelector
from datasets import ActiveDataset, ExtendableDataset, ImageDataset
from entry.demo.web.config import ActiveLearningConfig, settings
from entry.demo.web.core.foundation_model import FoundationModelManager
from entry.demo.web.core.specialist_model import SpecialistModelManager
from entry.demo.web.models.requests import ActiveLearningConfigRequest
from entry.demo.web.models.responses import (ActiveLearningConfigResponse,
                                             ActiveLearningStateResponse)
from entry.demo.web.services.context.active_learning import \
    ActiveLearningContext

Logger: Final = getLogger(__name__)

class ActiveLearningState:
    def __init__(self, workspace_id: str):
        self.workspace_id = workspace_id
        self.config = ActiveLearningConfig()
        self.feature_dict: Optional[Dict[str, torch.Tensor]] = None
        self.current_train_set: List[str] = []
        self.current_pool_set: List[str] = []
        self.annotated_set: List[Any] = []
        self.selected_image: dict = {}
        self.selected_set = []
        self.created_at = datetime.now()

    def clear(self):
        """Clear workspace state."""
        self.current_train_set.clear()
        self.current_pool_set.clear()
        self.selected_image = {}
        self.selected_set.clear()
        self.annotated_set.clear()
        self.feature_dict = None


class ActiveLearningService:
    """Main service managing multiple workspace states."""

    def __init__(self):
        self._workspace_states: Dict[str, ActiveLearningState] = {}
        self._lock = threading.RLock()

        self.foundation_model_manager = FoundationModelManager()
        self.specialist_model_manager = SpecialistModelManager()
        self.executor = ThreadPoolExecutor(max_workers=4)

        self.foundation_model = self.foundation_model_manager
        self.specialist_model = self.specialist_model_manager

    def _get_workspace_state(self, workspace_id: str) -> ActiveLearningState:
        """Get or create a workspace state."""
        with self._lock:
            if workspace_id not in self._workspace_states:
                self._workspace_states[workspace_id] = ActiveLearningState(workspace_id)
                Logger.info(f"Created workspace state for: {workspace_id}")
            return self._workspace_states[workspace_id]

    def update_feature_dict_keys(self, workspace_id: str, old_base_path, new_base_path):
        """Update feature dictionary keys for a specific workspace."""
        state = self._get_workspace_state(workspace_id)
        if not state.feature_dict:
            return

        old_base = str(old_base_path)
        new_base = str(new_base_path)
        updated_dict = {}

        for old_key, feature in state.feature_dict.items():
            if old_key.startswith(old_base):
                new_key = old_key.replace(old_base, new_base, 1)
                updated_dict[new_key] = feature
            else:
                updated_dict[old_key] = feature

        state.feature_dict = updated_dict

    async def get_feature_dict(self, workspace_id: str, batch_size, device, active_dataset: ActiveDataset):
        """Get feature dictionary for a specific workspace."""
        dataset = ConcatDataset([active_dataset.get_train_dataset(), active_dataset.get_pool_dataset()])
        dataloader = DataLoader(dataset, batch_size=batch_size)
        model, preprocess, _ = await self.foundation_model.initialize(device)

        feature_dict = {}
        if model is None:
            Logger.error("model is none")
            return feature_dict
        if preprocess is None:
            Logger.error("preprocess is none")
            return feature_dict

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
                Logger.debug(f"Storing feature for workspace {workspace_id}, key: '{case_name}'")
                feature_dict[case_name] = feature_batch[i]

        return feature_dict

    def get_state(self, workspace_id: str) -> ActiveLearningStateResponse:
        """Get state for a specific workspace."""
        state = self._get_workspace_state(workspace_id)
        return ActiveLearningStateResponse(
            train_count=len(state.current_train_set),
            pool_count=len(state.current_pool_set),
            annotated_count=len(state.annotated_set)
        )

    def get_annotated_set(self, workspace_id: str) -> List[Any]:
        """Get annotated set for workspace."""
        state = self._get_workspace_state(workspace_id)
        return state.annotated_set

    def get_config(self, workspace_id: str) -> Any:
        state = self._get_workspace_state(workspace_id)
        return state.config

    def get_param_config(self, workspace_id: str) -> dict[str, str | Any]:
        state = self._get_workspace_state(workspace_id)


    def get_train_set(self, workspace_id: str) -> List[str]:
        """Get training set for workspace."""
        state = self._get_workspace_state(workspace_id)
        return state.current_train_set

    def get_pool_set(self, workspace_id: str) -> List[str]:
        """Get pool set for workspace."""
        state = self._get_workspace_state(workspace_id)
        return state.current_pool_set

    def get_selected_set(self, workspace_id: str) -> List[Any]:
        """Get selected set for workspace."""
        state = self._get_workspace_state(workspace_id)
        return state.selected_set

    def get_selected_image(self, workspace_id: str) -> dict:
        """Get selected image for workspace."""
        state = self._get_workspace_state(workspace_id)
        return state.selected_image

    async def update_config(self, workspace_id: str,
                            config_request: ActiveLearningConfigRequest) -> ActiveLearningConfigResponse:
        """Update active learning configuration for workspace."""
        state = self._get_workspace_state(workspace_id)

        try:
            state.config.budget = config_request.budget
            state.config.model = config_request.model
            state.config.device = torch.device(config_request.device)
            state.config.batch_size = config_request.batch_size
            state.config.loaded_feature_weight = config_request.loaded_feature_weight
            state.config.sharp_factor = config_request.sharp_factor
            state.config.loaded_feature_only = config_request.loaded_feature_only
            state.config.model_ckpt = config_request.model_ckpt

            Logger.info(f"Updated config for workspace {workspace_id}, budget: {state.config.budget}")
            state.feature_dict = None

            return ActiveLearningConfigResponse(
                message=f"Configuration updated successfully for workspace {workspace_id}",
                config=config_request.model_dump()
            )
        except Exception as e:
            Logger.error(f"Failed to update config for workspace {workspace_id}: {str(e)}")
            return ActiveLearningConfigResponse(
                success=False,
                message=f"Failed to update configuration: {str(e)}"
            )

    async def active_select(self, workspace_id: str, train_set, pool_set, budget, model_ckpt,
                            batch_size, device, loaded_feature_weight, sharp_factor, loaded_feature_only):
        """Perform active selection for workspace."""
        state = self._get_workspace_state(workspace_id)

        try:
            train_dataset = ExtendableDataset(ImageDataset(train_set, image_channels=1, image_size=settings.IMAGE_SIZE))
            pool_dataset = ExtendableDataset(ImageDataset(pool_set, image_channels=1, image_size=settings.IMAGE_SIZE))

            Logger.info(f"Workspace {workspace_id} - train_dataset: {len(train_dataset)}")
            Logger.info(f"Workspace {workspace_id} - pool_dataset: {len(pool_dataset)}")

            active_dataset = ActiveDataset(train_dataset, pool_dataset)

            if state.feature_dict is None:
                state.feature_dict = await self.get_feature_dict(workspace_id, batch_size, device, active_dataset)

            active_selector = KMeanSelector(
                batch_size=4,
                num_workers=1,
                pin_memory=True,
                metric="l2",
                feature_dict=state.feature_dict,
                loaded_feature_weight=loaded_feature_weight,
                sharp_factor=sharp_factor,
                loaded_feature_only=loaded_feature_only,
            )

            await self.specialist_model.load_model(str(settings.MODELS_DIR) + "/" + model_ckpt)
            return active_selector.select_next_batch(active_dataset, budget, self.specialist_model.model, device)

        except Exception as e:
            Logger.error(f"Failed to select batch for workspace {workspace_id}: {str(e)}")
            return None

    def predict_pseudo_label(self, workspace_id: str, image_pil):
        """Predict pseudo label for workspace (uses shared specialist model)."""
        image = F.to_tensor(image_pil)
        image = image.unsqueeze(0)
        _, _, H, W = image.shape
        image = self.specialist_model.processor.preprocess(image)

        with torch.no_grad():
            pred = self.specialist_model.model(image)
            pseudo_label = pred.argmax(1)

        pseudo_label = self.specialist_model.processor.postprocess(pseudo_label, [H, W])
        return pseudo_label[0]

    def clear(self, workspace_id: str):
        """Clear specific workspace data."""
        with self._lock:
            if workspace_id in self._workspace_states:
                self._workspace_states[workspace_id].clear()
                Logger.info(f"Cleared workspace: {workspace_id}")

    def delete_workspace(self, workspace_id: str):
        """Delete entire workspace."""
        with self._lock:
            if workspace_id in self._workspace_states:
                del self._workspace_states[workspace_id]
                Logger.info(f"Deleted workspace: {workspace_id}")

    def get_workspace_stats(self) -> Dict[str, dict]:
        """Get stats for all workspaces."""
        with self._lock:
            stats = {}
            for workspace_id, state in self._workspace_states.items():
                stats[workspace_id] = {
                    'train_count': len(state.current_train_set),
                    'pool_count': len(state.current_pool_set),
                    'annotated_count': len(state.annotated_set),
                    'has_feature_dict': state.feature_dict is not None,
                    'feature_dict_size': len(state.feature_dict) if state.feature_dict else 0
                }
            return stats

    @contextmanager
    def workspace(self, workspace_id: str) -> Generator['ActiveLearningContext', None, None]:
        """Context manager for workspace operations."""
        yield ActiveLearningContext(self, workspace_id)


# Global service instance
active_learning_service = ActiveLearningService()
