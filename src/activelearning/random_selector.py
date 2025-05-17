import torch
from torch import nn

from .active_selector import ActiveSelector
from datasets.active_dataset import ActiveDataset


class RandomSelector(ActiveSelector):
    def select_next_batch(
        self,
        active_dataset: ActiveDataset,
        select_num: int,
        model: nn.Module,
        device: torch.device,
    ):
        labeled_size, pool_size = active_dataset.get_size()
        scores = torch.rand(pool_size)

        _, indices = torch.sort(scores, descending=True)
        selected_samples = [
            active_dataset.pool_dataset.image_idx[id]
            for id in indices[:select_num]
        ]

        return selected_samples
