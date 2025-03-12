from typing import Any
from abc import ABC, abstractmethod
from pathlib import Path
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    @staticmethod
    @abstractmethod
    def find_samples(data_path: Path | str, require_label: bool = True) -> list[dict]:
        pass

    @abstractmethod
    def get_sample(self, index: int, normalize: bool = True) -> Any:
        pass
