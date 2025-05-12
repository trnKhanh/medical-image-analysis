"""
More details can be checked at https://github.com/Shathe/SemiSeg-Contrastive
Thanks the authors for providing such a model to achieve the class-level separation.
"""

from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from memories.feature_memory import FeatureMemory


class PrototypeContrastiveLoss(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 3,
        memory_cls: Callable = FeatureMemory,
        memory_kwargs: dict = {"elements_per_class": 32},
    ):
        super().__init__()
        self.model = model
        self.num_classes = num_classes + 1  # Includes background class
        self.prototype_memory = memory_cls(
            num_classes=num_classes, **memory_kwargs
        )

    # def add_features_from_sample_learned(self, model, features, class_labels):
    def update_memory(
        self,
        features: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ):
        correct_prediction_mask = torch.logical_and(
            predictions == labels, predictions > 0
        )
        features = features.permute(0, 2, 3, 1)
        correct_labels = labels[correct_prediction_mask]
        correct_features = features[correct_prediction_mask, ...]

        with torch.no_grad():
            self.model.eval()
            proj_correct_features = self.model.projection_head(correct_features)
            self.model.train()

        self.prototype_memory.add_features_from_sample_learned(
            self.model, proj_correct_features, correct_labels
        )

    def forward(self, features: torch.Tensor, class_labels: torch.Tensor):
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(-1, C).contiguous()
        class_labels = class_labels.reshape(-1)

        pred_features = self.model.prediction_head(
            self.model.projection_head(features)
        )
        loss = 0
        for c in range(self.num_classes):
            # get features of an specific class
            mask_c = class_labels == c
            features_c = pred_features[mask_c, :]
            memory_c = self.prototype_memory.memory[c]  # N, C

            # get the self-attention MLPs both for memory features vectors (projected vectors) and network feature vectors (predicted vectors)
            selector = self.model.__getattr__(
                "contrastive_class_selector_" + str(c)
            )
            selector_memory = self.model.__getattr__(
                "contrastive_class_selector_memory" + str(c)
            )

            if (
                memory_c is not None
                and features_c.shape[0] > 1
                and memory_c.shape[0] > 1
            ):
                memory_c = torch.from_numpy(memory_c).to(self.model.device)

                # L2 normalize vectors
                memory_c = F.normalize(memory_c, dim=1)  # N, C
                features_c_norm = F.normalize(features_c, dim=1)  # M, C

                # compute similarity. All elements with all elements
                similarities = torch.mm(
                    features_c_norm, memory_c.transpose(1, 0)
                )  # MxN
                distances = (
                    1 - similarities
                )  # values between [0, 2] where 0 means same vectors
                # M (elements), N (memory)

                # now weight every sample

                learned_weights_features = selector(
                    features_c.detach()
                )  # detach for trainability
                learned_weights_features_memory = selector_memory(memory_c)

                # self-atention in the memory featuers-axis and on the learning contrsative featuers-axis
                learned_weights_features = torch.sigmoid(
                    learned_weights_features
                )
                rescaled_weights = (
                    learned_weights_features.shape[0]
                    / learned_weights_features.sum(dim=0)
                ) * learned_weights_features
                rescaled_weights = rescaled_weights.repeat(
                    1, distances.shape[1]
                )
                distances = distances * rescaled_weights

                learned_weights_features_memory = torch.sigmoid(
                    learned_weights_features_memory
                )
                learned_weights_features_memory = (
                    learned_weights_features_memory.permute(1, 0)
                )
                rescaled_weights_memory = (
                    learned_weights_features_memory.shape[0]
                    / learned_weights_features_memory.sum(dim=0)
                ) * learned_weights_features_memory
                rescaled_weights_memory = rescaled_weights_memory.repeat(
                    distances.shape[0], 1
                )
                distances = distances * rescaled_weights_memory

                loss = loss + distances.mean()

        return loss / self.num_classes
