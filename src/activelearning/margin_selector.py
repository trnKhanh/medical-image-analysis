import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from .active_selector import ActiveSelector
from datasets.active_dataset import ActiveDataset


class MarginSelector(ActiveSelector):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
        smooth: float = 1e-8,
    ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.smooth = smooth

    def cal_scores(
        self,
        active_dataset: ActiveDataset,
        model: nn.Module,
        device: torch.device,
    ):
        dataloader = DataLoader(
            dataset=active_dataset.get_pool_dataset(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        score_list, case_name_list = [], []

        for sampled_batch in tqdm(dataloader):
            image_batch = sampled_batch["image"].to(device)
            case_name = sampled_batch["case_name"]

            model.eval()
            with torch.no_grad():
                pred = model(image_batch)
                prob = pred.softmax(1)
                top_2 = prob.topk(2, dim=1)[0]
                margin = -1 * (top_2[:, 0] - top_2[:, 1])
                scores = margin.mean(dim=[-2, -1])

            score_list.extend(scores)
            case_name_list.extend(case_name)

        return score_list, case_name_list

    def select_next_batch(
        self,
        active_dataset: ActiveDataset,
        select_num: int,
        model: nn.Module,
        device: torch.device,
    ):
        labeled_size, pool_size = active_dataset.get_size()
        if labeled_size == 0:
            scores = torch.rand(pool_size)

            _, indices = torch.sort(scores, descending=True)
            selected_samples = [
                active_dataset.pool_dataset.image_idx[id]
                for id in indices[:select_num]
            ]
        else:
            score_list, case_name_list = self.cal_scores(
                active_dataset, model, device
            )
            score_tensor = torch.stack(score_list, dim=0)

            _, indices = torch.sort(score_tensor, descending=True)
            selected_samples = [
                case_name_list[id]
                for id in indices[:select_num]
            ]

        return selected_samples
