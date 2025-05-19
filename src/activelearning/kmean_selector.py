from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import pairwise_distances

from .active_selector import ActiveSelector
from datasets.active_dataset import ActiveDataset


def kcenter_greedy(dist_mat, n_data, budget, init_idx):
    assert (
        dist_mat.shape[0] == n_data
    ), "Size of distance matrix and number of data doesn't match!"

    # init
    all_indices = np.arange(n_data)
    labeled_indices = np.zeros((n_data,), dtype=np.bool_)
    labeled_indices[init_idx] = True

    # sample
    for _ in tqdm(range(budget), desc="k-center greedy"):
        mat = dist_mat[~labeled_indices, :][:, labeled_indices]
        # for all the unselected points, find its nearest neighbor in selected points
        mat_min = mat.min(axis=1)
        # find nearest neighbor with largest distance as the next selected point
        q_index_ = mat_min.argmax()
        q_index = all_indices[~labeled_indices][q_index_]
        labeled_indices[q_index] = True

    selected_idx = all_indices[labeled_indices]
    all_else_idx = all_indices[~labeled_indices]
    newly_selected_idx = list(set(selected_idx) - set(init_idx))

    return newly_selected_idx


class KMeanSelector(ActiveSelector):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
        smooth: float = 1e-8,
        metric: Literal["cosine", "l1", "l2", "haversine"] = "cosine",
    ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.smooth = smooth
        self.metric = metric

    def cal_scores(
        self,
        active_dataset: ActiveDataset,
        model: nn.Module,
        device: torch.device,
    ):
        labeled_dataset = active_dataset.get_train_dataset()
        pool_dataset = active_dataset.get_pool_dataset()
        all_dataset = ConcatDataset([labeled_dataset, pool_dataset])

        core_list = labeled_dataset.image_idx
        all_list = labeled_dataset.image_idx + pool_dataset.image_idx

        dataloader = DataLoader(
            dataset=all_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        feat_list = []

        for sampled_batch in tqdm(dataloader):
            image_batch = sampled_batch["image"].to(device)
            case_name = sampled_batch["case_name"]

            model.eval()
            with torch.no_grad():
                feat = model.get_enc_feature(image_batch)
                feat = feat.cpu().numpy()

            feat_list.append(feat)

        feats = np.concatenate(feat_list, axis=0)
        feat_dist_mat = pairwise_distances(feats, metric=self.metric)

        return np.array(core_list), np.array(all_list), feats, feat_dist_mat

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
            core_list, all_list, feats, feat_dist_mat = self.cal_scores(
                active_dataset, model, device
            )
            selected_sample_ids = kcenter_greedy(
                dist_mat=feat_dist_mat,
                n_data=len(all_list),
                budget=select_num,
                init_idx=np.arange(len(core_list)),
            )

            selected_samples = list(all_list[selected_sample_ids].tolist())

        return selected_samples
