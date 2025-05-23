# each class has its corresponding point embedding
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder_prompt_large import MaskDecoder_prompt_large
from .prompt_encoder_prompt_class import PromptEncoder_prompt_class
import numpy as np

from skimage.measure import label
import cv2


def MaskToBoxSimple(mask):
    mask = mask.squeeze()
    # find coordinates of points in the region
    row, col = np.argwhere(mask).T
    # find the four corner coordinates
    y0, x0 = row.min(), col.min()
    y1, x1 = row.max(), col.max()

    return [x0, y0, x1, y1]


class Sam_dualmask_same_prompt_class_random_large(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder_prompt_class,
        mask_decoders: list[MaskDecoder_prompt_large],
        dropout_rate: float = 0.0,
        num_points_prompt: tuple[int, int] = (1, 2),
        bbox_change_rate: tuple[float, float] = (0.1, 0.2),
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoders = nn.ModuleList(mask_decoders)
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False
        )
        self.dropout_rate = dropout_rate
        self.feature_dropout = nn.Dropout2d(dropout_rate)
        self.num_points_prompt = num_points_prompt
        self.bbox_change_rate = bbox_change_rate

        dim_in = self.mask_decoders[0].transformer_dim // 16  # 16
        feat_dim = dim_in * 2  # 32
        num_classes = self.mask_decoders[0].num_mask_tokens

        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

        for class_c in range(num_classes):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1),
            )
            self.__setattr__(
                "contrastive_class_selector_" + str(class_c), selector
            )

        for class_c in range(num_classes):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1),
            )
            self.__setattr__(
                "contrastive_class_selector_memory" + str(class_c), selector
            )

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self,
        batched_input,
        multimask_output,
        image_size,
        prompt_idx=-1,
        prompt_mode=None,
        image_embeddings=None,
    ):
        # prompt_idx indicates which branch is used to generate prompts
        if isinstance(batched_input, list):
            outputs = self.forward_test(batched_input, multimask_output)
        else:
            outputs = self.forward_train(
                batched_input,
                multimask_output,
                image_size,
                prompt_idx,
                prompt_mode,
                image_embeddings,
            )
        return outputs

    def _get_prompt_embeddings(self, low_res_masks, image_size, prompt):
        # generate prompts based on the coarse prediction
        (
            points_prompt,
            points_prompt_random,
            fit_boxes_prompt,
            loose_boxes_prompt,
            mask_prompt,
        ) = self.prompt_generate_random_fast(low_res_masks, image_size, True)

        sparse_embeddings = sparse_embeddings_r = dense_embeddings = None
        if prompt == "point":
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points_prompt, boxes=None, masks=None
            )
            sparse_embeddings_r, _ = self.prompt_encoder(
                points=points_prompt_random, boxes=None, masks=None
            )
        elif prompt == "box":
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=fit_boxes_prompt, masks=None
            )
            sparse_embeddings_r, _ = self.prompt_encoder(
                points=None, boxes=loose_boxes_prompt, masks=None
            )
        elif prompt == "mask":
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=None, masks=mask_prompt
            )
        elif prompt == "point-box":
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points_prompt, boxes=fit_boxes_prompt, masks=None
            )
            sparse_embeddings_r, _ = self.prompt_encoder(
                points=points_prompt_random,
                boxes=loose_boxes_prompt,
                masks=None,
            )
        elif prompt == "point-mask":
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points_prompt, boxes=None, masks=mask_prompt
            )
            sparse_embeddings_r, _ = self.prompt_encoder(
                points=points_prompt_random, boxes=None, masks=None
            )
        elif prompt == "box-mask":
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=fit_boxes_prompt, masks=mask_prompt
            )
            sparse_embeddings_r, _ = self.prompt_encoder(
                points=None, boxes=loose_boxes_prompt, masks=None
            )
        elif prompt == "all":
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points_prompt, boxes=fit_boxes_prompt, masks=mask_prompt
            )
            sparse_embeddings_r, _ = self.prompt_encoder(
                points=points_prompt_random,
                boxes=loose_boxes_prompt,
                masks=mask_prompt,
            )
        else:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=None, masks=None
            )

        return sparse_embeddings, sparse_embeddings_r, dense_embeddings

    def get_image_embeddings(self, batched_input):
        input_images = self.preprocess(batched_input)
        image_embeddings = self.image_encoder(input_images)
        return image_embeddings

    def forward_train(
        self,
        batched_input,
        multimask_output,
        image_size,
        prompt_idx,
        prompt,
        image_embeddings=None,
    ):
        if image_embeddings is None:
            image_embeddings = self.get_image_embeddings(batched_input)

        if prompt_idx >= 0:
            prompt_iter = itertools.cycle(prompt)
            for i in range(prompt_idx + 1):
                prompt = next(prompt_iter)

        if prompt_idx >= 0:
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=None, masks=None
                )

                sparse_embeddings = sparse_embeddings.detach()
                dense_embeddings = dense_embeddings.detach()
        else:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=None, masks=None
            )

        if self.dropout_rate > 0:
            dropout_image_embeddings = self.feature_dropout(image_embeddings)
        else:
            dropout_image_embeddings = image_embeddings

        low_res_logits = [
            torch.zeros(1) for _ in range(len(self.mask_decoders))
        ]
        iou_predictions = [
            torch.zeros(1) for _ in range(len(self.mask_decoders))
        ]
        dense_features = [
            torch.zeros(1) for _ in range(len(self.mask_decoders))
        ]

        low_res_logits_r = [
            torch.zeros(1) for _ in range(len(self.mask_decoders))
        ]
        iou_predictions_r = [
            torch.zeros(1) for _ in range(len(self.mask_decoders))
        ]
        dense_features_r = [
            torch.zeros(1) for _ in range(len(self.mask_decoders))
        ]

        assemble_low_res_logits = torch.zeros(1, device=self.device)

        for id, mask_decoder in enumerate(self.mask_decoders):
            if id == prompt_idx:
                continue

            low_res_logits[id], iou_predictions[id], dense_features[id] = (
                mask_decoder(
                    image_embeddings=dropout_image_embeddings,
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
            )
            
            # Obtain the pseudo labels used for generating prompts
            with torch.no_grad():
                if self.dropout_rate > 0:
                    # If dropout_rate > 0, pass the raw image_embeddings
                    raw_low_res_logit, _, _ = mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=self.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=multimask_output,
                    )

                    assemble_low_res_logits = (
                        assemble_low_res_logits + raw_low_res_logit.softmax(1)
                    )
                else:
                    assemble_low_res_logits = (
                        assemble_low_res_logits + low_res_logits[id].softmax(1)
                    )

        assemble_low_res_logits /= len(self.mask_decoders) - 1

        if prompt_idx >= 0 and prompt_idx < len(self.mask_decoders):
            sparse_embeddings, sparse_embeddings_r, dense_embeddings = (
                self._get_prompt_embeddings(
                    assemble_low_res_logits, image_size, prompt
                )
            )

            (
                low_res_logits[prompt_idx],
                iou_predictions[prompt_idx],
                dense_features[prompt_idx],
            ) = self.mask_decoders[prompt_idx](
                image_embeddings=dropout_image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            (
                low_res_logits_r[prompt_idx],
                iou_predictions_r[prompt_idx],
                dense_features_r[prompt_idx],
            ) = self.mask_decoders[prompt_idx](
                image_embeddings=dropout_image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings_r,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

        masks = [torch.zeros(1) for _ in range(len(self.mask_decoders))]

        for id in range(len(self.mask_decoders)):
            assert isinstance(low_res_logits[id], torch.Tensor)

            masks[id] = self.postprocess_masks(
                low_res_logits[id],
                input_size=(image_size, image_size),
                original_size=(image_size, image_size),
            )

        outputs = {
            "masks": masks,
            "iou_predictions": iou_predictions,
            "low_res_logits": low_res_logits,
            "low_res_logits_r": low_res_logits_r,
            "dense_features": dense_features,
            "dense_features_r": dense_features_r,
        }

        return outputs

    @torch.no_grad()
    def forward_test(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack(
            [self.preprocess(x["image"]) for x in batched_input], dim=0
        )
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(
            batched_input, image_embeddings
        ):
            if "point_coords" in image_record:
                points = (
                    image_record["point_coords"],
                    image_record["point_labels"],
                )
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions, _ = self.mask_decoder1(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def _get_bbox(self, binary_mask, max_change_rate=0.1):
        h, w = binary_mask.shape
        y_list, x_list = np.where(binary_mask == 1)
        x1, x2, y1, y2 = x_list.min(), x_list.max(), y_list.min(), y_list.max()

        fit_x_change = np.floor((x2 - x1) * max_change_rate)
        fit_y_change = np.floor((y2 - y1) * max_change_rate)
        fit_x1 = np.clip(x1 + np.random.randint(-fit_x_change, 1), 0, w - 1)
        fit_x2 = np.clip(x2 + np.random.randint(0, fit_x_change + 1), 0, w - 1)
        fit_y1 = np.clip(y1 + np.random.randint(-fit_y_change, 1), 0, h - 1)
        fit_y2 = np.clip(y2 + np.random.randint(0, fit_y_change + 1), 0, h - 1)

        return np.array([[fit_x1, fit_y1], [fit_x2, fit_y2]])

    def prompt_generate_random_fast(
        self, coarse_mask, image_size, israndom=False
    ):  # generate point prompts
        b, num_class, h, w = coarse_mask.shape

        coarse_mask_np = torch.argmax(coarse_mask, dim=1)
        coarse_mask_np = F.interpolate(
            coarse_mask_np.unsqueeze(1).float(),
            (image_size, image_size),
            mode="nearest",
        ).squeeze(1)
        coarse_mask_np = coarse_mask_np.detach().cpu().numpy()

        # points: BxNx2 tensor & boxes
        # points_prompt = np.zeros([b, num_class, 2])
        # points_label = np.zeros([b, num_class])
        # points_prompt_random = np.zeros([b, num_class, 2])
        num_points = np.random.randint(
            self.num_points_prompt[0], self.num_points_prompt[1] + 1, num_class
        )
        cum_num_points = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(num_points)]
        )
        total_points = np.sum(num_points)
        points_prompt = np.zeros([b, total_points, 2])
        points_label = np.zeros([b, total_points])
        points_prompt_random = np.zeros([b, total_points, 2])
        fit_boxes_prompt = np.zeros([b, num_class - 1, 2, 2])
        loose_boxes_prompt = np.zeros([b, num_class - 1, 2, 2])
        boxes_label = np.zeros([b, num_class - 1])

        from datetime import datetime

        st_time = datetime.now()
        for idx in range(b):  # iterate over each image
            for cls in range(num_class):  # find points for each class
                cls_slice = slice(cum_num_points[cls], cum_num_points[cls + 1])
                # obtain the binary mask
                mask_cls = (coarse_mask_np[idx] == cls).astype(np.uint8)
                if mask_cls.max() > 0:
                    region_mask = np.array(label(mask_cls, connectivity=2))
                    region_ids, region_sizes = np.unique(
                        region_mask, return_counts=True
                    )
                    if region_ids[0] == 0:
                        region_ids = region_ids[1:]
                        region_sizes = region_sizes[1:]

                    region_sizes, region_ids = zip(
                        *sorted(zip(region_sizes, region_ids), reverse=True)
                    )

                    binary_msk = np.where(region_mask == region_ids[0], 1, 0)

                    if israndom:
                        cY_r, cX_r = np.where(binary_msk == 1)
                        # random_idx = np.random.randint(0, len(cX_r))
                        # points_prompt_random[idx,cls,0], points_prompt_random[idx,cls,1] = int(cX_r[random_idx]), int(cY_r[random_idx])
                        random_idx = np.random.randint(
                            0, len(cX_r), num_points[cls]
                        )
                        random_coords = np.stack(
                            [cX_r[random_idx], cY_r[random_idx]], axis=1
                        )
                        points_prompt_random[idx, cls_slice] = random_coords

                    # Calculates the distance to the closest zero pixel for each pixel of the source image.
                    # Ref from RITM: https://github.com/SamsungLabs/ritm_interactive_segmentation/blob/aa3bb52a77129e477599b5edfd041535bc67b259/isegm/data/points_sampler.py
                    # NOTE: numpy and opencv have inverse definition of row and column
                    # NOTE: SAM and opencv have the same definition
                    padded_mask = np.uint8(
                        np.pad(binary_msk, ((1, 1), (1, 1)), "constant")
                    )
                    dist_img = cv2.distanceTransform(
                        padded_mask, distanceType=cv2.DIST_L2, maskSize=5
                    ).astype(np.float32)[1:-1, 1:-1]
                    # cY, cX = np.where(dist_img==dist_img.max())
                    # random_idx = np.random.randint(0, len(cX))
                    # points_prompt[idx,cls,0], points_prompt[idx,cls,1] = int(cX[random_idx]), int(cY[random_idx])
                    cY, cX = np.where(dist_img == dist_img.max())
                    random_idx = np.random.randint(0, len(cX), num_points[cls])
                    center_coords = np.stack(
                        [cX[random_idx], cY[random_idx]], axis=1
                    )
                    points_prompt[idx, cls_slice] = center_coords

                    if cls > 0:
                        points_label[idx, cls_slice] = cls

                        fit_boxes_prompt[idx, cls - 1] = self._get_bbox(
                            binary_msk, self.bbox_change_rate[0]
                        )
                        loose_boxes_prompt[idx, cls - 1] = self._get_bbox(
                            binary_msk, self.bbox_change_rate[1]
                        )
                else:
                    (
                        points_prompt[idx, cls_slice, 0],
                        points_prompt[idx, cls_slice, 1],
                    ) = (points_prompt[idx, 0, 0], points_prompt[idx, 0, 1])
                    (
                        points_prompt_random[idx, cls_slice, 0],
                        points_prompt_random[idx, cls_slice, 1],
                    ) = (points_prompt[idx, 0, 0], points_prompt[idx, 0, 1])
                    points_label[idx, cls_slice] = 0
        mask_prompt = F.interpolate(
            torch.tensor(coarse_mask_np).unsqueeze(1).float(),
            self.prompt_encoder.mask_input_size,
            mode="nearest",
        ).to(coarse_mask.device)
        points_prompt = torch.tensor(points_prompt).to(coarse_mask.device)
        points_label = torch.tensor(points_label).to(coarse_mask.device)
        points_prompt = (points_prompt, points_label)

        fit_boxes_prompt = torch.tensor(fit_boxes_prompt).to(coarse_mask.device)
        boxes_label = torch.tensor(boxes_label).to(coarse_mask.device)
        fit_boxes_prompt = (fit_boxes_prompt, boxes_label)

        if israndom:
            points_prompt_random = torch.tensor(points_prompt_random).to(
                coarse_mask.device
            )
            points_prompt_random = (points_prompt_random, points_label)

            loose_boxes_prompt = torch.tensor(loose_boxes_prompt).to(
                coarse_mask.device
            )
            loose_boxes_prompt = (loose_boxes_prompt, boxes_label)

            return (
                points_prompt,
                points_prompt_random,
                fit_boxes_prompt,
                loose_boxes_prompt,
                mask_prompt,
            )

        return points_prompt, fit_boxes_prompt, mask_prompt

    def _prompt_generate_random_fast(
        self, coarse_mask, img_size, israndom=False
    ):  # generate point prompts
        b, num_class, h, w = coarse_mask.shape

        coarse_mask_np = torch.argmax(coarse_mask, dim=1)
        coarse_mask_np = F.interpolate(
            coarse_mask_np.unsqueeze(1).float(),
            (img_size, img_size),
            mode="nearest",
        ).squeeze(1)
        coarse_mask_np = coarse_mask_np.detach().cpu().numpy()

        # points: BxNx2 tensor & boxes
        # points_prompt = np.zeros([b, num_class, 2])
        # points_label = np.zeros([b, num_class])
        # points_prompt_random = np.zeros([b, num_class, 2])
        num_points = np.random.randint(1, 3, num_class)
        cum_num_points = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(num_points)]
        )
        total_points = np.sum(num_points)
        points_prompt = np.zeros([b, total_points, 2])
        points_label = np.zeros([b, total_points])
        points_prompt_random = np.zeros([b, total_points, 2])
        fit_boxes_prompt = np.zeros([b, num_class - 1, 2, 2])
        loose_boxes_prompt = np.zeros([b, num_class - 1, 2, 2])
        boxes_label = np.zeros([b, num_class - 1])
        for idx in range(b):  # iterate over each image
            for cls in range(num_class):  # find points for each class
                cls_slice = slice(cum_num_points[cls], cum_num_points[cls + 1])
                # obtain the binary mask
                mask_cls = (coarse_mask_np[idx] == cls).astype(np.uint8)
                if mask_cls.max() > 0:
                    label_msk, region_ids = label(
                        mask_cls, connectivity=2, return_num=True
                    )
                    ratio_list, regionid_list = [], []
                    for region_id in range(1, region_ids + 1):
                        # find coordinates of points in the region
                        binary_msk = np.where(label_msk == region_id, 1, 0)

                        # clean some region that is abnormally small
                        r = np.sum(binary_msk) / np.sum(mask_cls)
                        # print('curr mask over all mask ratio', r)
                        ratio_list.append(r)
                        regionid_list.append(region_id)

                    ratio_list, regionid_list = zip(
                        *sorted(zip(ratio_list, regionid_list))
                    )
                    regionid_list = regionid_list[::-1]

                    binary_msk = np.where(label_msk == regionid_list[0], 1, 0)

                    if israndom:
                        cY_r, cX_r = np.where(binary_msk == 1)
                        # random_idx = np.random.randint(0, len(cX_r))
                        # points_prompt_random[idx,cls,0], points_prompt_random[idx,cls,1] = int(cX_r[random_idx]), int(cY_r[random_idx])
                        random_idx = np.random.randint(
                            0, len(cX_r), num_points[cls]
                        )
                        random_coords = np.stack(
                            [cX_r[random_idx], cY_r[random_idx]], axis=1
                        )
                        points_prompt_random[idx, cls_slice] = random_coords

                    # Calculates the distance to the closest zero pixel for each pixel of the source image.
                    # Ref from RITM: https://github.com/SamsungLabs/ritm_interactive_segmentation/blob/aa3bb52a77129e477599b5edfd041535bc67b259/isegm/data/points_sampler.py
                    # NOTE: numpy and opencv have inverse definition of row and column
                    # NOTE: SAM and opencv have the same definition
                    padded_mask = np.uint8(
                        np.pad(binary_msk, ((1, 1), (1, 1)), "constant")
                    )
                    dist_img = cv2.distanceTransform(
                        padded_mask, distanceType=cv2.DIST_L2, maskSize=5
                    ).astype(np.float32)[1:-1, 1:-1]
                    # cY, cX = np.where(dist_img==dist_img.max())
                    # random_idx = np.random.randint(0, len(cX))
                    # points_prompt[idx,cls,0], points_prompt[idx,cls,1] = int(cX[random_idx]), int(cY[random_idx])
                    cY, cX = np.where(dist_img == dist_img.max())
                    random_idx = np.random.randint(0, len(cX), num_points[cls])
                    center_coords = np.stack(
                        [cX[random_idx], cY[random_idx]], axis=1
                    )
                    points_prompt[idx, cls_slice] = center_coords

                    if cls > 0:
                        points_label[idx, cls_slice] = cls

                        fit_boxes_prompt[idx, cls - 1] = self._get_bbox(
                            binary_msk, 0.1
                        )
                        loose_boxes_prompt[idx, cls - 1] = self._get_bbox(
                            binary_msk, 0.3
                        )
                else:
                    (
                        points_prompt[idx, cls_slice, 0],
                        points_prompt[idx, cls_slice, 1],
                    ) = (points_prompt[idx, 0, 0], points_prompt[idx, 0, 1])
                    (
                        points_prompt_random[idx, cls_slice, 0],
                        points_prompt_random[idx, cls_slice, 1],
                    ) = (points_prompt[idx, 0, 0], points_prompt[idx, 0, 1])
                    points_label[idx, cls_slice] = 0
        mask_prompt = F.interpolate(
            torch.tensor(coarse_mask_np).unsqueeze(1).float(),
            self.prompt_encoder.mask_input_size,
            mode="nearest",
        ).to(coarse_mask.device)
        points_prompt = torch.tensor(points_prompt).to(coarse_mask.device)
        points_label = torch.tensor(points_label).to(coarse_mask.device)
        points_prompt = (points_prompt, points_label)

        fit_boxes_prompt = torch.tensor(fit_boxes_prompt).to(coarse_mask.device)
        boxes_label = torch.tensor(boxes_label).to(coarse_mask.device)
        fit_boxes_prompt = (fit_boxes_prompt, boxes_label)

        if israndom:
            points_prompt_random = torch.tensor(points_prompt_random).to(
                coarse_mask.device
            )
            points_prompt_random = (points_prompt_random, points_label)

            loose_boxes_prompt = torch.tensor(loose_boxes_prompt).to(
                coarse_mask.device
            )
            loose_boxes_prompt = (loose_boxes_prompt, boxes_label)

            return (
                points_prompt,
                points_prompt_random,
                fit_boxes_prompt,
                loose_boxes_prompt,
                mask_prompt,
            )

        return points_prompt, fit_boxes_prompt, mask_prompt
