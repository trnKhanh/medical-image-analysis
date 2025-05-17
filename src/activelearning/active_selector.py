from abc import ABC, abstractmethod
from typing import Any

from torch import nn

from datasets.active_dataset import ActiveDataset


class ActiveSelector(ABC):
    @abstractmethod
    def select_next_batch(
        self,
        active_dataset: ActiveDataset,
        select_num: int,
        model: nn.Module,
    ) -> Any:
        pass
