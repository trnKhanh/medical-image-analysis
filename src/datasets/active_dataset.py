from pathlib import Path
import json

from .common import ExtendableDataset
from utils import get_path


class ActiveDataset:
    def __init__(
        self,
        labeled_dataset: ExtendableDataset,
        pool_dataset: ExtendableDataset,
    ):
        self.labeled_dataset = labeled_dataset
        self.pool_dataset = pool_dataset

    def data_list(self):
        data_dict = {
            "labeled_image_idx": self.labeled_dataset.image_idx,
            "pool_image_idx": self.pool_dataset.image_idx,
        }
        return data_dict

    def save_data_list(self, save_path: Path | str):
        save_path = get_path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(self.data_list(), f)

    def load_data_list(self, data_list: Path | str | dict):
        if isinstance(data_list, (Path, str)):
            with open(data_list, "r") as f:
                data_dict = json.load(f)
        else:
            data_dict = data_list

        self.labeled_dataset.image_idx = data_dict["labeled_image_idx"]
        self.pool_dataset.image_idx = data_dict["pool_image_idx"]

    def extend_train_set(self, new_image_idx: list[int] = []):
        self.labeled_dataset.image_idx.extend(new_image_idx)
        for idx in new_image_idx:
            self.pool_dataset.image_idx.remove(idx)

    def get_train_dataset(self):
        return self.labeled_dataset

    def get_pool_dataset(self):
        return self.get_pool_dataset

    def get_size(self):
        return len(self.labeled_dataset), len(self.pool_dataset)
