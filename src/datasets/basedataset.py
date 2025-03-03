from abc import ABC, abstractmethod
from pathlib import Path
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    @staticmethod
    @abstractmethod
    def get_samples(data_path: Path | str) -> list[str]:
        pass
