from pathlib import Path

from torch.utils.data import Dataset
from monai.transforms.transform import Transform


class BTCVDataset(Dataset):
    def __init__(
        self,
        work_dir: Path | None = None,
        data_dir: Path | None = None,
        download: bool = True,
        transform: Transform | None = None,
    ):
        if work_dir:
            self.work_dir = work_dir
        else:
            self.work_dir = Path.cwd()

        if data_dir:
            self.data_dir = data_dir
        else:
            self.data_dir = self.work_dir / "data/btcv"

        if download:
            self.__download_data()

        self.image_paths = self.__get_image_paths()

        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        # TODO: read and return image here

    def __download_data(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # TODO: Download data

    def __get_image_paths(self) -> list[Path]:
        # TODO: Obtain image paths here
        return []
