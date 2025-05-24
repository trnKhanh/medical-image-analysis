from pathlib import Path
from typing import Literal

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


def kcenter_greedy(dist_mat, n_data, budget, init_idx, coreset_criteria="min"):
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
        if coreset_criteria == "min":
            mat_min = mat.min(axis=1)
        elif coreset_criteria == "sum":
            mat_min = mat.sum(axis=1)
        else:
            raise RuntimeError(
                f"coreset_criteria {coreset_criteria} is undefined"
            )

        # find nearest neighbor with largest distance as the next selected point
        q_index_ = mat_min.argmax()
        q_index = all_indices[~labeled_indices][q_index_]
        labeled_indices[q_index] = True

    selected_idx = all_indices[labeled_indices]
    all_else_idx = all_indices[~labeled_indices]
    newly_selected_idx = list(set(selected_idx) - set(init_idx))

    return newly_selected_idx


class CoresetSelector(ActiveSelector):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
        smooth: float = 1e-8,
        metric: Literal["cosine", "l1", "l2", "haversine"] = "cosine",
        coreset_criteria: Literal["sum", "min"] = "min",
        feature_path: Path | str | None = None,
        loaded_feature_weight: float = 0.0,
    ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.smooth = smooth
        self.metric = metric
        self.feature_path = get_path(feature_path) if feature_path else None
        self.coreset_criteria = coreset_criteria
        self.loaded_feature_weight = loaded_feature_weight

    def cal_scores(
        self,
        active_dataset: ActiveDataset,
        model: nn.Module | None,
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
        loaded_feat_list = []

        for sampled_batch in tqdm(dataloader):
            image_batch = sampled_batch["image"].to(device)
            case_name = sampled_batch["case_name"]

            if model:
                model.eval()
                with torch.no_grad():
                    feat = model.get_enc_feature(image_batch)
                    feat = feat.cpu().numpy()
                feat_list.append(feat)

            if self.feature_path:
                for idx in range(len(case_name)):
                    case = case_name[idx]
                    feature_file = self.feature_path / f"{case}.h5"
                    with h5py.File(feature_file, "r") as h5f:
                        loaded_feat_ds = h5f["feature"]
                        assert isinstance(loaded_feat_ds, h5py.Dataset)
                        loaded_feat = loaded_feat_ds[:]

                    loaded_feat_list.append(loaded_feat)

        if len(loaded_feat_list):
            loaded_feats = np.stack(loaded_feat_list, axis=0)
            loaded_feat_dist_mat = pairwise_distances(
                loaded_feats, metric=self.metric
            )
        else:
            loaded_feats = None
            loaded_feat_dist_mat = 0

        if len(feat_list):
            feats = np.concatenate(feat_list, axis=0)
            feat_dist_mat = pairwise_distances(feats, metric=self.metric)
        else:
            feats = None
            feat_dist_mat = 0

        final_dist_mat = (
            self.loaded_feature_weight * loaded_feat_dist_mat
            + (1 - self.loaded_feature_weight) * feat_dist_mat
        )

        return (
            np.array(core_list),
            np.array(all_list),
            loaded_feats,
            feats,
            final_dist_mat,
        )

    def select_next_batch(
        self,
        active_dataset: ActiveDataset,
        select_num: int,
        model: nn.Module,
        device: torch.device,
    ):
        labeled_size, pool_size = active_dataset.get_size()
        if labeled_size == 0 and self.loaded_feature_weight == 0:
            scores = torch.rand(pool_size)

            _, indices = torch.sort(scores, descending=True)
            selected_samples = [
                active_dataset.pool_dataset.image_idx[id]
                for id in indices[:select_num]
            ]
        elif labeled_size == 0:
            if self.feature_path:
                core_list, all_list, loaded_feats, feats, feat_dist_mat = (
                    self.cal_scores(active_dataset, None, device)
                )
                _, selected_indices = kmeans_plusplus(
                    X=loaded_feats, n_clusters=select_num
                )
                selected_samples = list(all_list[selected_indices])
            else:
                scores = torch.rand(pool_size)

                _, indices = torch.sort(scores, descending=True)
                selected_samples = [
                    active_dataset.pool_dataset.image_idx[id]
                    for id in indices[:select_num]
                ]
        else:
            core_list, all_list, loaded_feats, feats, feat_dist_mat = (
                self.cal_scores(active_dataset, model, device)
            )
            selected_sample_ids = kcenter_greedy(
                dist_mat=feat_dist_mat,
                n_data=len(all_list),
                budget=select_num,
                init_idx=np.arange(len(core_list)),
                coreset_criteria=self.coreset_criteria,
            )

            selected_samples = list(all_list[selected_sample_ids])

        return selected_samples
