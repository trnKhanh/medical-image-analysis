from typing import Any, List

from entry.demo.web.models.requests import ActiveLearningConfigRequest
from entry.demo.web.models.responses import (ActiveLearningConfigResponse,
                                             ActiveLearningStateResponse)
from entry.demo.web.services.interfaces import ActiveLearningProtocol


class ActiveLearningContext:
    def __init__(self, service: ActiveLearningProtocol, workspace_id: str):
        self._service = service
        self._workspace_id = workspace_id

    def get_config(self) -> Any:
        return self._service.get_config(self._workspace_id)

    def get_param_config(self) -> dict[str, str | Any]:
        return self._service.get_param_config(self._workspace_id)

    def get_state(self) -> ActiveLearningStateResponse:
        return self._service.get_state(self._workspace_id)

    def get_annotated_set(self) -> List[Any]:
        return self._service.get_annotated_set(self._workspace_id)

    def get_train_set(self) -> List[str]:
        return self._service.get_train_set(self._workspace_id)

    def get_pool_set(self) -> List[str]:
        return self._service.get_pool_set(self._workspace_id)

    def get_selected_set(self) -> List[Any]:
        return self._service.get_selected_set(self._workspace_id)

    def get_selected_image(self) -> dict:
        return self._service.get_selected_image(self._workspace_id)

    async def update_config(self, config_request: ActiveLearningConfigRequest) -> ActiveLearningConfigResponse:
        return await self._service.update_config(self._workspace_id, config_request)

    async def active_select(self, train_set, pool_set, budget, model_ckpt, batch_size, device,
                            loaded_feature_weight, sharp_factor, loaded_feature_only):
        return await self._service.active_select(
            self._workspace_id, train_set, pool_set, budget, model_ckpt,
            batch_size, device, loaded_feature_weight, sharp_factor, loaded_feature_only
        )

    def predict_pseudo_label(self, image_pil):
        return self._service.predict_pseudo_label(self._workspace_id, image_pil)

    def update_feature_dict_keys(self, old_base_path, new_base_path):
        return self._service.update_feature_dict_keys(self._workspace_id, old_base_path, new_base_path)

    def clear(self):
        return self._service.clear(self._workspace_id)

    @property
    def workspace_id(self) -> str:
        return self._workspace_id
