from pathlib import Path
from typing import Literal, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from sklearn.cluster import kmeans_plusplus
import h5py

from .active_selector import ActiveSelector
from datasets.active_dataset import ActiveDataset

from utils import get_path


def image_wise_grad(loss, model, last_layer_name="decoder.seg_output.weight"):
    """
    grad_embeddimg: a tensor of shape [C*D,] for one sample
    """

    model.zero_grad()
    loss.backward(retain_graph=True)
    last_layer_param = dict(model.named_parameters())[last_layer_name]
    last_layer_grad = last_layer_param.grad
    grad_embeddimg = (
        last_layer_grad.detach().flatten().clone()
    )  # clone is important!

    model.zero_grad()

    return grad_embeddimg


class BADGESelector(ActiveSelector):
    def __init__(
        self,
        dice_loss: Callable,
        ce_loss: Callable,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
        smooth: float = 1e-8,
        multiple_loss: Literal["add", "sep"] = "add",
    ) -> None:
        self.dice_loss = dice_loss
        self.ce_loss = ce_loss
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.multiple_loss = multiple_loss
        self.smooth = smooth

    def cal_scores(
        self,
        active_dataset: ActiveDataset,
        model: nn.Module,
        device: torch.device,
    ):
        model.eval()

        pool_dataset = active_dataset.get_pool_dataset()

        dataloader = DataLoader(
            dataset=pool_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        grad_embed_list = []
        case_name_list = []

        for sampled_batch in tqdm(dataloader):
            image_batch = sampled_batch["image"].to(device)
            case_name = sampled_batch["case_name"]
            case_name_list.extend(case_name)

            outputs = model(image_batch)
            preds = outputs.softmax(1).argmax(1)

            if self.multiple_loss == "sep":
                ce_loss = self.ce_loss(outputs, preds)
                grad_embed_ce = image_wise_grad(ce_loss, model)

                dice_loss = self.dice_loss(outputs, preds)
                grad_embed_dice = image_wise_grad(dice_loss, model)

                grad_embed = torch.cat([ce_loss, dice_loss])
            else:
                ce_loss = self.ce_loss(outputs, preds)
                dice_loss = self.dice_loss(outputs, preds)

                loss = ce_loss + dice_loss
                grad_embed = image_wise_grad(loss, model)

            grad_embed_list.append(grad_embed)

        grad_embed = torch.stack(grad_embed_list, dim=0).cpu().numpy()

        return (np.array(case_name_list), grad_embed)

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
            case_name_list, grad_embed = (
                self.cal_scores(active_dataset, model, device)
            )
            _, selected_indices = kmeans_plusplus(X=grad_embed, n_clusters=select_num)

            selected_samples = list(case_name_list[selected_indices])

        return selected_samples
