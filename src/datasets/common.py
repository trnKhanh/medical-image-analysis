from pathlib import Path

from .basedataset import BaseDataset


class ExtendableDataset(BaseDataset):
    @staticmethod
    def find_samples(
        data_path: Path | str, require_label: bool = True
    ) -> list[dict]:
        raise RuntimeError(
            "ExtendableDataset does not have find_samples function"
        )

    def __init__(
        self, dataset: BaseDataset, image_idx: list | None = None
    ):
        self.dataset = dataset
        self.case_name_to_idx = {}

        for id in range(len(self.dataset)):
            sample = self.dataset[id]

            self.case_name_to_idx[sample["case_name"]] = id

        if image_idx is None:
            image_idx = list(self.case_name_to_idx.keys())

        self.image_idx = image_idx

    def __len__(self):
        return len(self.image_idx)

    def __getitem__(self, index):
        return self.get_sample(index)

    def get_sample(self, index: int, normalize: bool = True):
        case_name = self.image_idx[index]

        return self.dataset.get_sample(
            self.case_name_to_idx[case_name], normalize
        )
