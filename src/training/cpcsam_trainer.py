import os
import random
from datetime import datetime
import json
from typing import Dict, List, Literal, Sequence
import logging
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Optimizer
import torchvision.transforms.functional as F

from PIL import Image

from rich.logging import RichHandler
from rich.console import Console

from tqdm import tqdm

from .base_trainer import BaseTrainer
from datasets import ACDCDataset, TwoStreamBatchSampler
from losses.compound_losses import DiceAndCELoss, DualBranchDiceAndCELoss
from losses.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from metric.metric import HD
from scheduler.lr_scheduler import PolyLRScheduler

from utils import get_path, dummy_context
from models.segment_anything import (
    LoRA_Sam,
    sam_model_registry,
    test_single_volume,
    test_single_volume_prompt,
)
from transforms.normalization import ZScoreNormalize
from transforms.image_transform import (
    RandomGamma,
    RandomContrast,
    RandomBrightness,
    RandomGaussianBlur,
    RandomGaussianNoise,
    SimulateLowRes,
)
from transforms.joint_transform import (
    JointResize,
    RandomRotation,
    RandomAffine,
    RandomCrop2D,
    MirrorTransform,
)
from transforms.common import RandomTransform, ComposeTransform


class CPCSAMTrainer(BaseTrainer):
    PROMPT_MODE = Literal[
        "point",
        "box",
        "mask",
        "point-mask",
        "point-box",
        "box-mask",
        "all",
    ]

    def __init__(
        self,
        work_path: Path | str = Path.cwd(),
        device: torch.device | str = torch.device("cuda"),
        seed: int = 12345,
        # Model parameters
        in_channels: int = 3,
        num_classes: int = 2,
        patch_size: int | tuple[int, int] | None = None,
        image_size: int | tuple[int, int] | None = None,
        sam_name: str = "vit_b_dualmask_same_prompt_class_random_large",
        model_ckpt: Path | str | None = None,
        lora_rank: int = 4,
        lora_ckpt: Path | str | None = None,
        promptmode: list[PROMPT_MODE] = ["point"],
        # Data parameters
        data_path: Path | str = "data",
        labeled_ratio: float = 1.0,
        labeled_num: int | None = 1,
        do_augment: bool = False,
        do_normalize: bool = False,
        batch_size: int = 32,
        labeled_batch_ratio: float = 0.5,
        num_workers: int = 1,
        pin_memory: bool = True,
        # Training parameters
        optimizer_name: Literal["adam", "adamw", "sgd"] = "adamw",
        optimizer_kwargs: dict = {},
        num_epochs: int = 1000,
        warmup_iter: int = 5000,
        start_lr: float = 1e-3,
        lr_scheduler_name: Literal["poly"] = "poly",
        save_freq_epoch: int = 100,
        valid_freq_iter: int = 200,
        save_metric_name: Literal["dice", "hd"] = "dice",
        loss_name: Literal["dice+ce"] = "dice+ce",
        dice_weight: float = 0.8,
        consistency_weight_1: float = 0.4,
        consistency_weight_2: float = 0.05,
        early_stop_max_patience: int = 200,
        # Inference parameters
        stride: int | tuple[int, ...] | list[int] | None = None,
        # Log parameters
        verbose: bool = True,
        log_path: Path | str | None = None,
        config_path: Path | str | None = None,
        log_mode: str = "a",
        log_override: bool = False,
    ):
        self.work_path = get_path(work_path)
        self.device = torch.device("cpu")
        self.to(device)

        self._set_seed(seed)

        # >>> Model parameters
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.image_size = image_size
        self.sam_name = sam_name
        self.model_ckpt = model_ckpt
        self.lora_rank = lora_rank
        self.lora_ckpt = lora_ckpt
        self.promptmode = promptmode
        # <<< Model parameters

        # >>> Data parameters
        self.data_path = data_path
        self.labeled_ratio = labeled_ratio
        self.labeled_num = labeled_num
        self.do_augment = do_augment
        self.do_normalize = do_normalize
        self.batch_size = batch_size
        self.labeled_batch_size = batch_size * labeled_batch_ratio
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        # <<< Data parameters

        # >>> Training parameters
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs
        self.num_epochs = num_epochs
        self.warmup_iter = warmup_iter
        self.start_lr = start_lr
        self.lr_scheduler_name = lr_scheduler_name
        self.save_freq_epoch = save_freq_epoch
        self.valid_freq_iter = valid_freq_iter
        self.save_metric_name = save_metric_name
        self.loss_name = loss_name
        self.dice_weight = dice_weight
        self.consistency_weight_1 = consistency_weight_1
        self.consistency_weight_2 = consistency_weight_2
        self.early_stop_max_patience = early_stop_max_patience

        self.current_epoch = 0
        # <<< Training parameters

        # >>> Inference parameters
        self.stride = stride
        # <<< Inference parameters

        # >>> Log parameters
        self.verbose = verbose
        self.log_path = log_path
        self.config_path = config_path
        self.log_mode = log_mode
        self.log_override = log_override
        # <<< Log parameters

    def initialize(self):
        self._setup_logger()
        self._build_model()
        if self.lora_ckpt:
            self.load_model_checkpoint(self.lora_ckpt)

    def _set_seed(self, seed: int):
        self.seed = seed
        torch.manual_seed(seed)

    def _setup_logger(self):
        self.logger = logging.getLogger("MIA.CPCSAMTrainer")
        self.logger.setLevel(logging.DEBUG)

        self._setup_log_file()
        self._setup_log_shell()

    def _setup_log_file(self):
        assert self.logger is not None

        if not self.log_path:
            return

        self.log_path = get_path(self.log_path)

        if self.log_path.exists() and not self.log_override:
            current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = (
                self.log_path.parent
                / f"{self.log_path.stem}@{current_time_str}{self.log_path.suffix}"
            )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(self.log_path, self.log_mode)
        file_formatter = logging.Formatter(
            fmt="%(levelname)s <%(asctime)s>: %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def _add_config_file(self):
        assert self.logger is not None

        if not self.config_path:
            return

        self.config_path = get_path(self.config_path)

        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        self.config_file_handler = logging.FileHandler(self.config_path, "w")
        self.logger.addHandler(self.config_file_handler)

    def _remove_config_file(self):
        self.logger.removeHandler(self.config_file_handler)

    def _setup_log_shell(self):
        assert self.logger is not None

        if not self.verbose:
            return

        shell_handler = RichHandler(
            console=Console(stderr=True),
            rich_tracebacks=True,
            show_time=False,
            show_path=False,
            show_level=False,
            keywords=["Training summary", "Epoch", "Train", "Valid"],
        )
        shell_formatter = logging.Formatter(fmt="%(message)s")
        shell_handler.setFormatter(shell_formatter)
        self.logger.addHandler(shell_handler)

    def _build_model(self):
        self.sam, self.img_embedding_size = sam_model_registry[self.sam_name](
            image_size=self.image_size,
            num_classes=self.num_classes,
            checkpoint=self.model_ckpt,
            pixel_mean=[0, 0, 0],
            pixel_std=[1, 1, 1],
        )
        self.model = LoRA_Sam(self.sam, self.lora_rank)

    def load_model_checkpoint(self, lora_ckpt: str | Path):
        if self.model is None:
            raise RuntimeError("Model is not built before loading checkpoint")
        try:
            self.model.load_lora_parameters(str(lora_ckpt))
            self.logger.info(f"Loaded model lora checkpoint from {lora_ckpt}")
        except Exception as e:
            self.logger.warn(f"Failed to load model lora checkpoint from {lora_ckpt}")
            self.logger.exception(e)

    def save_model_checkpoint(self, lora_ckpt: str | Path):
        if self.model is None:
            raise RuntimeError("Model is not built before saving checkpoint")
        try:
            self.model.load_lora_parameters(str(lora_ckpt))
            self.logger.info(f"Saved model lora checkpoint to {lora_ckpt}")
        except Exception as e:
            self.logger.warn(f"Failed to save model lora checkpoint to {lora_ckpt}")
            self.logger.exception(e)


    def patients_to_slices(self, dataset: Literal["ACDC"], patiens_num):
        ref_dict = {}
        if dataset == "ACDC":
            ref_dict = {
                "1": 32,
                "3": 68,
                "7": 136,
                "14": 256,
                "21": 396,
                "28": 512,
                "35": 664,
                "140": 1312,
            }
        else:
            self.logger.error("Dataset not found")
        return ref_dict[str(patiens_num)]

    def get_data(self):
        train_dataset = ACDCDataset(
            data_path=self.data_path,
            split="train",
            normalize=self._get_train_normalize(),
            transform=self._get_train_transform(),
            logger=self.logger,
        )
        total_slices = len(train_dataset)
        labeled_slices = self.patients_to_slices("ACDC", self.labeled_num)
        labeled_indices = list(range(0, labeled_slices))
        unlabeled_indices = list(range(labeled_slices, total_slices))
        batch_sampler = TwoStreamBatchSampler(
            labeled_indices,
            unlabeled_indices,
            self.batch_size,
            self.batch_size - self.labeled_batch_size,
        )

        def worker_init_fn(worker_id):
            random.seed(self.seed + worker_id)

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            pin_memory=self.pin_memory,
        )

        valid_dataset = ACDCDataset(
            data_path=self.data_path,
            split="valid",
            normalize=self._get_valid_normalize(),
            transform=self._get_valid_transform(),
            logger=self.logger,
        )

        valid_dataloader = DataLoader(
            dataset=valid_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=1,
            pin_memory=True,
        )

        return (
            train_dataset,
            valid_dataset,
            train_dataloader,
            valid_dataloader,
        )

    def _get_train_transform(self):
        transforms = []
        if self.do_augment:
            transforms.append(
                RandomTransform(RandomAffine(scale=(0.7, 1.4)), p=0.2)
            )
            transforms.append(
                RandomTransform(RandomAffine(degrees=(-15, 15)), p=0.2)
            )
            transforms.append(
                RandomTransform(RandomGaussianNoise(sigma=(0, 0.1)), p=0.1)
            )
            transforms.append(
                RandomTransform(RandomGaussianBlur(sigma=(0.5, 1)), p=0.2)
            )
            transforms.append(
                RandomTransform(RandomBrightness(brightness=0.25), p=0.15)
            )
            transforms.append(
                RandomTransform(RandomContrast(contrast=0.25), p=0.15)
            )
            transforms.append(
                RandomTransform(SimulateLowRes(scale=(0.5, 1)), p=0.15)
            )
            transforms.append(
                RandomTransform(RandomGamma(gamma=(0.7, 1.5)), p=0.1)
            )

        if self.image_size:
            transforms.append(JointResize(self.image_size))

        return ComposeTransform(transforms)

    def _get_train_normalize(self):
        if self.do_normalize:
            return ZScoreNormalize()
        else:
            return None

    def _get_valid_transform(self):
        transforms = []
        if self.image_size:
            transforms.append(JointResize(self.image_size))
        return ComposeTransform(transforms)

    def _get_valid_normalize(self):
        if self.do_normalize:
            return ZScoreNormalize()
        else:
            return None

    def _get_optimizer(
        self,
        model: nn.Module,
    ):
        parameters = filter(lambda p: p.requires_grad, model.parameters())

        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(parameters, **self.optimizer_kwargs)
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(parameters, **self.optimizer_kwargs)
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(parameters, **self.optimizer_kwargs)
        else:
            raise ValueError(f'Optimizer "{self.optimizer_name}" not supported')

        if self.lr_scheduler_name == "poly":
            lr_scheduler = PolyLRScheduler(
                optimizer, self.start_lr, self.num_epochs, self.warmup_epochs
            )
        else:
            raise ValueError(
                f'Learning rate scheduler "{self.lr_scheduler_name}" not supported'
            )

        return optimizer, lr_scheduler

    def _get_loss(self):
        if self.loss_name == "DICE+CE":
            supervised_loss = DiceAndCELoss(
                dice_loss=MemoryEfficientSoftDiceLoss,
                dice_kwargs={"smooth": 1e-5, "do_bg": False},
                ce_loss=torch.nn.CrossEntropyLoss,
                ce_kwargs={},
                default_dice_weight=0.8,
            )
            unsupervised_loss = DualBranchDiceAndCELoss(
                dice_loss=MemoryEfficientSoftDiceLoss,
                dice_kwargs={"smooth": 1e-5, "do_bg": False},
                ce_loss=torch.nn.CrossEntropyLoss,
                ce_kwargs={},
                default_dice_weight=0.8,
            )
        else:
            raise ValueError(f"Loss function {self.loss_name} not found")

        return supervised_loss, unsupervised_loss

    def _get_save_metric(self):
        if self.save_metric_name == "HD":
            save_metric = HD()
        elif self.save_metric_name == "loss":
            save_metric, _ = self._get_loss()
        else:
            raise ValueError(
                f"Metric function {self.save_metric_name} not found"
            )

        return save_metric

    def _print_train_info(self):
        self._add_config_file()

        self.logger.info(f"Training summary")
        self.logger.info(f"seed: {self.seed}")
        self.logger.info(f"device: {self.device}")
        self.logger.info(f"start_epoch: {self.current_epoch}")
        self.logger.info(f"num_epochs: {self.num_epochs}")
        self.logger.info(f'log_file: "{self.log_path}"')

        self.logger.info(f"model: {self.model}")
        self.logger.info(f"  pretrained_model: {self.model_ckpt}")
        self.logger.info(f"  image_size: {self.image_size}")

        self.logger.info(f'data: "{self.data_path}"')
        self.logger.info(f"  train_slices: {len(self.train_dataset)}")
        self.logger.info(
            f"  labeled_patients (slices): {self.labeled_num} ({self.patients_to_slices('ACDC', self.labeled_num)})"
        )
        self.logger.info(f"  valid_slices: {len(self.valid_dataset)}")
        self.logger.info(f"  do_augment: {self.do_augment}")
        if self.do_augment:
            self.logger.info(
                f"{json.dumps(self._get_train_transform().get_params_dict(), indent=1)}"
            )
        self.logger.info(f"  normalize: {self.do_normalize}")
        self.logger.info(f"  batch_size: {self.batch_size}")
        self.logger.info(f"  labeled_batch_size: {self.labeled_batch_size}")
        self.logger.info(f"  num_workers: {self.num_workers}")
        self.logger.info(f"  pin_memory: {self.pin_memory}")

        self.logger.info(f"optimizer: {self.optimizer_name}")
        self.logger.info(f"  warmup_steps: {self.warmup_epochs}")
        self.logger.info(f"  lr_scheduler: {self.lr_scheduler_name}")
        self.logger.info(f"  start_lr: {self.start_lr}")
        self.logger.info(f"  optimizer_kwargs: {self.optimizer_kwargs}")
        self.logger.info(f"loss_fn: {self.loss_name}")
        self.logger.info(f"save_metric: {self.save_metric_name}")

        self._remove_config_file()

    def on_train_start(self):
        assert self.model is not None

        if self.lora_ckpt:
            self.load_model_checkpoint(self.lora_ckpt)

        self.model.train()
        self.model.to(self.device)

        self.current_epoch = 0
        self.current_iter = 0
        self.current_patience = 0

        self.optimizer, self.lr_scheduler = self._get_optimizer(
            self.model,
        )
        self.supervised_loss, self.unsupervised_loss = self._get_loss()
        self.save_metric = self._get_save_metric()

        self._best_valid_metric = torch.inf
        self._cur_valid_metric = torch.inf

        (
            self.train_dataset,
            self.valid_dataset,
            self.train_dataloader,
            self.valid_dataloader,
        ) = self.get_data()

        self._print_train_info()
        self._check_data_sanity()

    def _check_data_sanity(self):
        sanity_path = self.work_path / "sanity"
        sanity_path.mkdir(parents=True, exist_ok=True)

        for i in range(50):
            image, _ = self.train_dataset[0]
            image_pil: Image.Image = F.to_pil_image(image)
            image_pil.save(str(sanity_path / f"{i + 1}.png"))

    def on_train_end(self):
        self.save_state_dict(self.work_path / f"ckpt/final_model.pth")
        self.logger.info("")
        self.logger.info("")

    def on_epoch_start(self):
        self._epoch_start_time = time.time()

        self.logger.info("")
        self.logger.info(f"Epoch {self.current_epoch}:")

    def on_epoch_end(self):
        self.current_epoch += 1

        self._epoch_end_time = time.time()
        time_elapsed = self._epoch_end_time - self._epoch_start_time
        self.logger.info(f"Epoch time elapsed: {time_elapsed:.3f} seconds")

    def on_train_epoch_start(self):
        self._train_start_time = time.time()
        self.logger.info("Train")

        self.lr_scheduler.step(self.current_epoch)
        self.logger.info(f"LR: {self.lr_scheduler.get_last_lr()}")
        self.epoch_train_outputs = []
        self.train_tqdm = tqdm(self.train_dataloader)

        self.model.train()

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % self.save_freq == 0:
            self.save_state_dict(
                self.work_path / f"ckpt/epoch_{self.current_epoch}.pth"
            )

        train_tps = torch.stack(
            [o["tp_hard"] for o in self.epoch_train_outputs]
        ).sum(0)
        train_fps = torch.stack(
            [o["fp_hard"] for o in self.epoch_train_outputs]
        ).sum(0)
        train_fns = torch.stack(
            [o["fn_hard"] for o in self.epoch_train_outputs]
        ).sum(0)

        global_dc_per_class = [
            (2 * i / (2 * i + j + k)).item()
            for i, j, k in zip(train_tps, train_fps, train_fns)
        ]
        self.logger.info(f"DSC per class: {global_dc_per_class}")
        self.logger.info(
            f"Mean DSC: {torch.Tensor(global_dc_per_class).mean()}"
        )
        train_loss = (
            torch.stack([o["loss"] for o in self.epoch_train_outputs])
            .mean()
            .item()
        )
        self.logger.info(f"Loss ({self.loss}): {train_loss}")

        train_metric = (
            torch.stack([o["metric"] for o in self.epoch_train_outputs])
            .mean()
            .item()
        )
        self.logger.info(f"Metric ({self.save_metric}): {train_metric}")

        self._train_end_time = time.time()
        time_elapsed = self._train_end_time - self._train_start_time
        self.logger.info(f"Train time elapsed: {time_elapsed:.3f} seconds")

    def on_valid_epoch_start(self):
        self._valid_start_time = time.time()
        self.logger.info("Valid")

        self.model.eval()
        self.valid_tqdm = tqdm(self.valid_dataloader)
        self.epoch_valid_outputs = []

    def on_valid_epoch_end(self):
        valid_tps = torch.stack(
            [o["tp_hard"] for o in self.epoch_valid_outputs]
        ).sum(0)
        valid_fps = torch.stack(
            [o["fp_hard"] for o in self.epoch_valid_outputs]
        ).sum(0)
        valid_fns = torch.stack(
            [o["fn_hard"] for o in self.epoch_valid_outputs]
        ).sum(0)

        global_dc_per_class = [
            (2 * i / (2 * i + j + k)).item()
            for i, j, k in zip(valid_tps, valid_fps, valid_fns)
        ]

        self.logger.info(f"DSC per class: {global_dc_per_class}")
        self.logger.info(
            f"Mean DSC: {torch.Tensor(global_dc_per_class).mean()}"
        )

        valid_loss = (
            torch.stack([o["loss"] for o in self.epoch_valid_outputs])
            .mean()
            .item()
        )
        self.logger.info(f"Loss ({self.loss}): {valid_loss}")

        valid_metric = (
            torch.stack([o["metric"] for o in self.epoch_valid_outputs])
            .mean()
            .item()
        )
        self.logger.info(f"Metric ({self.save_metric}): {valid_metric}")

        self._cur_valid_metric = valid_metric

        if self._cur_valid_metric < self._best_valid_metric:
            self._best_valid_metric = self._cur_valid_metric
            self.logger.info(
                f"New best metric ({self.save_metric}): {self._cur_valid_metric}"
            )
            self.save_state_dict(self.work_path / "ckpt/best_model.pth")
            self.save_state_dict(
                self.work_path
                / f"ckpt/epoch_{self.current_epoch}_{self._best_valid_metric:.3f}.pth"
            )
            self.current_patience = 0
        else:
            self.current_patience += 1

        self._valid_end_time = time.time()
        time_elapsed = self._valid_start_time - self._valid_start_time
        self.logger.info(f"Valid time elapsed: {time_elapsed:.3f} seconds")

    def _get_one_hot_output(self, output: torch.Tensor):
        output_seg = output.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(
            output.shape, device=output.device, dtype=torch.int32
        )
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)

        return predicted_segmentation_onehot

    def train_step(self, sampled_batch):
        image_batch, label_batch = (
            sampled_batch["image"],
            sampled_batch["label"],
        )  #  [b, c, h, w], [b, h, w]

        image_batch, label_batch = image_batch.to(self.device), label_batch.to(
            self.device
        )

        assert image_batch.max() <= 3, f"image_batch max: {image_batch.max()}"
        multimask_output = True

        # the first round

        outputs = self.model(image_batch, multimask_output, self.image_size)

        outputs1 = outputs["low_res_logits1"]
        outputs2 = outputs["low_res_logits2"]

        labeled_outputs1 = outputs1[: self.labeled_batch_size]
        labeled_outputs2 = outputs2[: self.labeled_batch_size]
        labeled_label_batch = label_batch[: self.labeled_batch_size]

        supervised_loss1, loss_ce1, loss_dice1 = self.supervised_loss(
            labeled_outputs1,
            labeled_label_batch,
            self.dice_weight,
        )
        supervised_loss2, loss_ce2, loss_dice2 = self.supervised_loss(
            labeled_outputs2,
            labeled_label_batch,
            self.dice_weight,
        )
        loss1 = supervised_loss1 + supervised_loss2
        self.optimizer.zero_grad()
        loss1.backward()
        self.optimizer.step()

        # the second round
        if self.current_iter < self.warmup_iter:
            supervised_round2_loss1 = loss_round2_ce1 = loss_round2_dice1 = 0
            supervised_round2_loss1_r = loss_round2_ce1_r = (
                loss_round2_dice1_r
            ) = 0

            consistency_loss2 = 0.0
            consistency_loss1_r = 0.0
            loss2 = 0.0
        else:
            outputs_round2 = self.model(
                image_batch,
                multimask_output,
                self.image_size,
                1,
                self.promptmode,
            )
            outputs_round2_1 = outputs_round2["low_res_logits1"]
            outputs_round2_1_r = outputs_round2["low_res_logits1_r"]
            outputs_round2_2 = outputs_round2["low_res_logits2"]
            outputs_round2_soft1 = torch.softmax(outputs_round2_1, dim=1)
            outputs_round2_soft1_r = torch.softmax(outputs_round2_1_r, dim=1)
            outputs_round2_soft2 = torch.softmax(outputs_round2_2, dim=1)

            labeled_outputs_round2_1 = outputs_round2_1[
                : self.labeled_batch_size
            ]
            labeled_outputs_round2_1_r = outputs_round2_1_r[
                : self.labeled_batch_size
            ]

            supervised_round2_loss1, loss_round2_ce1, loss_round2_dice1 = (
                self.supervised_loss(
                    labeled_outputs_round2_1,
                    labeled_label_batch,
                    self.dice_weight,
                )
            )

            (
                supervised_round2_loss1_r,
                loss_round2_ce1_r,
                loss_round2_dice1_r,
            ) = self.supervised_loss(
                labeled_outputs_round2_1_r,
                labeled_label_batch,
                self.dice_weight,
            )

            outputs_round2_soft1 = (
                outputs_round2_soft1 + outputs_round2_soft1_r
            ) / 2.0
            pseudo_outputs1 = torch.argmax(
                outputs_round2_soft1[self.labeled_batch_size :].detach(),
                dim=1,
                keepdim=False,
            ).long()

            consistency_loss2 = self.supervised_loss(
                outputs_round2_2[self.labeled_batch_size :], pseudo_outputs1
            )
            consistency_loss1_r = self.supervised_loss(
                outputs_round2_1_r[self.labeled_batch_size :], pseudo_outputs1
            )

            loss2 = (
                supervised_round2_loss1
                + supervised_round2_loss1_r
                + self.consistency_weight_1 * consistency_loss2
                + self.consistency_weight_2 * consistency_loss1_r
            )
            self.optimizer.zero_grad()
            loss2.backward()
            self.optimizer.step()

        # the third round
        if self.current_iter < self.warmup_iter:
            supervised_round3_loss2 = loss_round3_ce1 = loss_round3_dice1 = 0.0
            supervised_round3_loss2_r = loss_round3_ce1_r = (
                loss_round3_dice1_r
            ) = 0.0

            consistency_loss1 = 0.0
            consistency_loss2_r = 0.0
            loss3 = 0.0

        else:
            outputs_round3 = self.model(
                image_batch,
                multimask_output,
                self.image_size,
                0,
                self.promptmode,
            )
            outputs_round3_1 = outputs_round3["low_res_logits1"]
            outputs_round3_2 = outputs_round3["low_res_logits2"]
            outputs_round3_2_r = outputs_round3["low_res_logits2_r"]
            outputs_round3_soft1 = torch.softmax(outputs_round3_1, dim=1)
            outputs_round3_soft2 = torch.softmax(outputs_round3_2, dim=1)
            outputs_round3_soft2_r = torch.softmax(outputs_round3_2_r, dim=1)

            labeled_outputs_round3_2 = outputs_round3_2[
                : self.labeled_batch_size
            ]
            labeled_outputs_round3_2_r = outputs_round3_2_r[
                : self.labeled_batch_size
            ]

            supervised_round3_loss2, loss_round3_ce1, loss_round3_dice1 = (
                self.supervised_loss(
                    labeled_outputs_round3_2,
                    labeled_label_batch,
                    self.dice_weight,
                )
            )
            (
                supervised_round3_loss2_r,
                loss_round3_ce1_r,
                loss_round3_dice1_r,
            ) = self.supervised_loss(
                labeled_outputs_round3_2_r,
                labeled_label_batch,
                self.dice_weight,
            )

            outputs_round3_soft2 = (
                outputs_round3_soft2 + outputs_round3_soft2_r
            ) / 2.0
            pseudo_outputs2 = torch.argmax(
                outputs_round3_soft2[self.labeled_batch_size :].detach(),
                dim=1,
                keepdim=False,
            )

            consistency_loss1 = self.supervised_loss(
                outputs_round3_1[self.labeled_batch_size :], pseudo_outputs2
            )
            consistency_loss2_r = self.supervised_loss(
                outputs_round3_2_r[self.labeled_batch_size :], pseudo_outputs2
            )

            loss3 = (
                supervised_round3_loss2
                + supervised_round3_loss2_r
                + self.consistency_weight_1 * consistency_loss1
                + self.consistency_weight_2 * consistency_loss2_r
            )
            self.optimizer.zero_grad()
            loss3.backward()
            self.optimizer.step()

        loss = loss1 + loss2 + loss3
        self.epoch_train_outputs.append(
            {
                "loss": loss,
                "loss1": loss1,
                "loss2": loss2,
                "loss3": loss3,
            }
        )

    def valid_step(self, sampled_batch):
        metric = test_single_volume(
            image=sampled_batch["image"],
            label=sampled_batch["label"],
            net=self.model,
            classes=self.num_classes,
            patch_size=self.image_size,
            loss_fn=self.supervised_loss,
        )
        prompt_metric = test_single_volume_prompt(
            image=sampled_batch["image"],
            label=sampled_batch["label"],
            net=self.model,
            classes=self.num_classes,
            promptidx=1,
            promptmode=self.promptmode,
            patch_size=self.image_size,
            loss_fn=self.supervised_loss,
        )

        self.epoch_valid_outputs.append(
            {
                "metric": torch.Tensor(metric),
                "prompt_metric": torch.Tensor(prompt_metric),
            }
        )

        metric = (
            torch.stack([o["metric"][:, :] for o in self.epoch_valid_outputs])
            .mean(0)
            .tolist()
        )
        prompt_metric = (
            torch.stack(
                [o["prompt_metric"][:, 0] for o in self.epoch_valid_outputs]
            )
            .mean(0)
            .tolist()
        )

        self.valid_tqdm.set_postfix(dict(metric=metric, prompt_metric=prompt_metric))

    def train(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            if self.is_finished():
                break
            self.on_epoch_start()
            self.on_train_epoch_start()
            for sampled_batch in self.train_tqdm:
                self.train_step(sampled_batch)
            self.on_train_epoch_end()

            with torch.no_grad():
                self.on_valid_epoch_start()
                for sampled_batch in self.valid_tqdm:
                    self.valid_step(sampled_batch)
                self.on_valid_epoch_end()
            self.on_epoch_end()

        self.on_train_end()

    def is_finished(self):
        if (
            isinstance(self.early_stop_max_patience, int)
            and self.early_stop_max_patience > 0
        ):
            return self.current_patience >= self.early_stop_max_patience

        return self.current_epoch >= self.num_epochs

    def run_training(self):
        self.train()
        self.perform_real_test()

    def perform_real_test(self):
        pass

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
        }

    def load_state_dict(self, save_path: str | Path):
        self.load_model_checkpoint(save_path)

    def save_state_dict(self, save_path: str | Path):
        save_path = get_path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), str(save_path))
        self.logger.info(f'Saved new checkpoint to "{save_path}"')

    def to(self, device: torch.device | str):
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda" and torch.cuda.is_available():
            self.device = device
        elif device.type == "mps" and torch.backends.mps.is_available():
            self.device = device
        else:
            self.device = torch.device("cpu")
