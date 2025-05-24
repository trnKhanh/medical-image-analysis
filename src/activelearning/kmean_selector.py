from pathlib import Path
from typing import Literal

import h5py
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from sklearn.cluster import kmeans_plusplus

from .active_selector import ActiveSelector
from datasets.active_dataset import ActiveDataset

from utils import get_path


class KMeanSelector(ActiveSelector):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
        smooth: float = 1e-8,
        metric: Literal["cosine", "l1", "l2", "haversine"] = "cosine",
        feature_path: Path | str | None = None,
        coreset_criteria: Literal["mean", "min"] = "min",
        loaded_feature_weight: float = 1.0,
        loaded_feature_only: bool = False,
        sharp_factor: float = 1.0,
        softmax: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.smooth = smooth
        self.metric = metric
        self.feature_path = get_path(feature_path) if feature_path else None
        self.coreset_criteria = coreset_criteria
        self.loaded_feature_weight = loaded_feature_weight
        self.loaded_feature_only = loaded_feature_only
        self.sharp_factor = sharp_factor
        self.softmax = softmax

    def get_features(
        self,
        dataset,
        model: nn.Module | None,
        device: torch.device,
    ):
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        feat_list = []
        loaded_feat_list = []
        case_name_list = []
        for sampled_batch in tqdm(dataloader):
            image_batch = sampled_batch["image"].to(device)
            case_name = sampled_batch["case_name"]
            case_name_list.extend(case_name)

            if model and not self.loaded_feature_only:
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

        total_feat_list = []

        if len(feat_list) and not self.loaded_feature_only:
            feats = np.concatenate(feat_list, axis=0)
            total_feat_list.append(feats)
        else:
            feats = None

        if len(loaded_feat_list):
            loaded_feats = np.stack(loaded_feat_list, axis=0)
            if feats is None:
                scale_factor = 1
            else:
                D1 = feats.shape[-1]
                D2 = loaded_feats.shape[-1]
                scale_factor = np.sqrt(D1 / D2 * self.loaded_feature_weight)

            total_feat_list.append(loaded_feats * scale_factor)
        else:
            loaded_feats = None

        total_feats = np.concatenate(total_feat_list, axis=1)

        return total_feats, np.array(case_name_list)

    def cal_scores(
        self,
        active_dataset: ActiveDataset,
        model: nn.Module | None,
        device: torch.device,
    ):
        labeled_size, pool_size = active_dataset.get_size()
        pool_dataset = active_dataset.get_pool_dataset()

        pool_feats, pool_case_names = self.get_features(
            pool_dataset, model, device
        )

        if labeled_size > 0:
            labeled_dataset = active_dataset.get_train_dataset()
            labeled_feats, labeled_case_names = self.get_features(
                labeled_dataset, model, device
            )
            pool2labeled_dist_mat = pairwise_distances(
                pool_feats, labeled_feats, metric=self.metric
            )
        else:
            labeled_feats = None
            labeled_case_names = None
            pool2labeled_dist_mat = None

        return (
            labeled_feats,
            pool_feats,
            labeled_case_names,
            pool_case_names,
            pool2labeled_dist_mat,
        )

    def select_next_batch(
        self,
        active_dataset: ActiveDataset,
        select_num: int,
        model: nn.Module,
        device: torch.device,
    ):
        (
            labeled_feats,
            pool_feats,
            labeled_case_names,
            pool_case_names,
            pool2labeled_dist_mat,
        ) = self.cal_scores(active_dataset, model, device)

        if pool2labeled_dist_mat is not None:
            if self.coreset_criteria == "min":
                sample_weight = pool2labeled_dist_mat.min(axis=1)
            else:
                sample_weight = pool2labeled_dist_mat.mean(axis=1)

            if self.softmax:
                sample_weight = torch.from_numpy(sample_weight)
                sample_weight = (sample_weight * self.sharp_factor).softmax(0)
                sample_weight = sample_weight.numpy()
            else:
                sample_weight = sample_weight ** self.sharp_factor
                sample_weight = sample_weight / sample_weight.sum()
        else:
            sample_weight = None

        _, selected_indices = kmeans_plusplus(
            X=pool_feats, n_clusters=select_num, sample_weight=sample_weight
        )

        selected_samples = list(pool_case_names[selected_indices].tolist())

        return selected_samples
