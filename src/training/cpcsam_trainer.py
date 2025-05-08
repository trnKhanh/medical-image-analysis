import random
from datetime import datetime
import json
from typing import Literal
import logging
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F

import numpy as np
import pandas as pd

from PIL import Image

from rich.logging import RichHandler
from rich.console import Console

from tqdm import tqdm

from .base_trainer import BaseTrainer
from datasets import ACDCDataset, TwoStreamBatchSampler
from losses.compound_losses import DiceAndCELoss, DualBranchDiceAndCELoss
from losses.dice import DiceLoss
from scheduler.lr_scheduler import PolyLRScheduler

from utils import get_path, draw_mask
from models.segment_anything import (
    LoRA_Sam,
    sam_model_registry,
    test_single_volume,
    test_single_volume_prompt,
    test_single_volume_mean,
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
    RandomRotation90,
)
from transforms.common import (
    RandomTransform,
    ComposeTransform,
    RandomChoiceTransform,
)


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
        num_classes: int = 3,
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
        num_epochs: int = 10000,
        min_iter: int = 10000,
        warmup_iter: int = 5000,
        start_lr: float = 1e-3,
        lr_scheduler_name: Literal["poly"] = "poly",
        lr_warmup_iter: int = 5000,
        save_freq_epoch: int = 100,
        valid_freq_iter: int = 200,
        save_metric_name: Literal["dice", "hd", "loss"] = "dice",
        loss_name: Literal["dice+ce"] = "dice+ce",
        dice_weight: float = 0.8,
        consistency_weight_1: float = 0.4,
        consistency_weight_2: float = 0.05,
        early_stop_max_patience: int | None = None,
        # Inference parameters
        stride: int | tuple[int, ...] | list[int] | None = None,
        # Log parameters
        verbose: bool = True,
        log_path: Path | str | None = None,
        config_path: Path | str | None = None,
        log_mode: str = "a",
        log_override: bool = False,
        exp_name: str = "",
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
        self.labeled_batch_size = round(batch_size * labeled_batch_ratio)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        # <<< Data parameters

        # >>> Training parameters
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs
        self.num_epochs = num_epochs
        self.min_iter = min_iter
        self.warmup_iter = warmup_iter
        self.start_lr = start_lr
        self.lr_scheduler_name = lr_scheduler_name
        self.lr_warmup_iter = lr_warmup_iter
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

        self.exp_name = exp_name

    def initialize(self):
        self._set_snapshot_work_dir()
        self._setup_logger()
        self._build_model()
        if self.lora_ckpt:
            self.load_model_checkpoint(self.lora_ckpt)

    def _set_snapshot_work_dir(self):
        current_time_str = datetime.now().strftime("%Y%m%d_%H")
        snapshot_list = [
            f"ACDC",
            f"{current_time_str}",
            f"patchsize-{self.patch_size}",
            f"imagesize-{self.image_size}",
            f"lora-{self.lora_rank}",
            f"prompt-{self.promptmode}",
            f"labeled-{self.labeled_num}",
            f"batchsize-{self.batch_size}",
            f"optimizer-{self.optimizer_name}",
            f"lrscheduler-{self.lr_scheduler_name}",
            f"lrwarmupiter-{self.lr_warmup_iter}"
            f"warmupiter-{self.warmup_iter}",
            f"startlr-{self.start_lr}",
            f"dice-{self.dice_weight}",
            f"coe1-{self.consistency_weight_1}",
            f"coe2-{self.consistency_weight_2}",
            f"dice-{self.dice_weight}",
            f"epoch-{self.num_epochs}",
        ]
        if self.exp_name:
            snapshot_list.append(self.exp_name)
        snapshot_str = "_".join(snapshot_list)
        self.work_path = self.work_path / snapshot_str
        self.work_path.mkdir(parents=True, exist_ok=True)

    def _set_seed(self, seed: int):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

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
            self.config_path = self.work_path / "config.txt"

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
            self.logger.warn(
                f"Failed to load model lora checkpoint from {lora_ckpt}"
            )
            self.logger.exception(e)

    def save_model_checkpoint(self, lora_ckpt: str | Path):
        if self.model is None:
            raise RuntimeError("Model is not built before saving checkpoint")
        try:
            self.model.save_lora_parameters(str(lora_ckpt))
            self.logger.info(f"Saved model lora checkpoint to {lora_ckpt}")
        except Exception as e:
            self.logger.warn(
                f"Failed to save model lora checkpoint to {lora_ckpt}"
            )
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

    def _worker_init_fn(self, worker_id):
        random.seed(self.seed + worker_id)

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

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            # worker_init_fn=self._worker_init_fn,
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
                RandomTransform(
                    ComposeTransform(
                        [
                            RandomRotation90(),
                            RandomChoiceTransform(
                                [MirrorTransform((-2)), MirrorTransform((-1))]
                            ),
                        ]
                    ),
                    p=0.5,
                )
            )
            transforms.append(
                RandomTransform(RandomAffine(degrees=(-20, 20)), p=0.5)
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
        return None

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
            optimizer = torch.optim.Adam(
                parameters,
                betas=(0.9, 0.999),
                weight_decay=0.1,
                **self.optimizer_kwargs,
            )
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(parameters, **self.optimizer_kwargs)
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                parameters,
                momentum=0.9,
                weight_decay=0.001,
                **self.optimizer_kwargs,
            )
        else:
            raise ValueError(f'Optimizer "{self.optimizer_name}" not supported')

        if self.lr_scheduler_name == "poly":
            lr_scheduler = PolyLRScheduler(
                optimizer,
                self.start_lr,
                self.max_iterations,
                self.lr_warmup_iter,
            )
        else:
            raise ValueError(
                f'Learning rate scheduler "{self.lr_scheduler_name}" not supported'
            )

        return optimizer, lr_scheduler

    def _get_loss(self):
        if self.loss_name == "dice+ce":
            supervised_loss = DiceAndCELoss(
                dice_loss=DiceLoss,
                dice_kwargs={
                    "num_classes": self.num_classes,
                    "smooth": 1e-5,
                    "do_bg": True,
                },
                ce_loss=torch.nn.CrossEntropyLoss,
                ce_kwargs={},
                default_dice_weight=0.8,
            )
            unsupervised_loss = DualBranchDiceAndCELoss(
                dice_loss=DiceLoss,
                dice_kwargs={
                    "num_classes": self.num_classes,
                    "smooth": 1e-5,
                    "do_bg": True,
                },
                ce_loss=torch.nn.CrossEntropyLoss,
                ce_kwargs={},
                default_dice_weight=0.8,
            )
        else:
            raise ValueError(f"Loss function {self.loss_name} not found")

        return supervised_loss, unsupervised_loss

    def _print_train_info(self):
        self._add_config_file()

        self.logger.info(f"Training summary")
        self.logger.info("")
        self.logger.info(f"device: {self.device}")
        self.logger.info(f"seed: {self.seed}")
        self.logger.info(f'log_file: "{self.log_path}"')

        self.logger.info(f"model: {self.model}")
        self.logger.info(f"  num_classes: {self.num_classes}")
        self.logger.info(f"  patch_size: {self.patch_size}")
        self.logger.info(f"  image_size: {self.image_size}")
        self.logger.info(f"  pretrained_model: {self.model_ckpt}")
        self.logger.info(f"  sam_name: {self.sam_name}")
        self.logger.info(f"  model_ckpt: {self.model_ckpt}")
        self.logger.info(f"  lora_rank: {self.lora_rank}")
        self.logger.info(f"  lora_ckpt: {self.lora_ckpt}")
        self.logger.info(f"  promptmode: {self.promptmode}")

        self.logger.info(f'data: "{self.data_path}"')
        self.logger.info(f"  train_size (slices): {len(self.train_dataset)}")
        self.logger.info(
            f"  labeled_patients (slices): {self.labeled_num} ({self.patients_to_slices('ACDC', self.labeled_num)})"
        )
        self.logger.info(f"  valid_size (volumns): {len(self.valid_dataset)}")
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
        self.logger.info(f"  lr_warmup_iter: {self.lr_warmup_iter}")
        self.logger.info(f"  lr_scheduler: {self.lr_scheduler_name}")
        self.logger.info(f"  start_lr: {self.start_lr}")
        self.logger.info(f"  optimizer_kwargs: {self.optimizer_kwargs}")
        self.logger.info(f"loss_fn: {self.loss_name}")
        self.logger.info(f"save_metric: {self.save_metric_name}")
        self.logger.info(f"start_epoch: {self.current_epoch}")
        self.logger.info(f"num_epochs: {self.num_epochs}")
        self.logger.info(f"warmup_iter: {self.warmup_iter}")
        self.logger.info(f"save_freq_epoch: {self.save_freq_epoch}")
        self.logger.info(f"valid_freq_iter: {self.valid_freq_iter}")
        self.logger.info(f"dice_weight: {self.dice_weight}")
        self.logger.info(f"consistency_weight_1: {self.consistency_weight_1}")
        self.logger.info(f"consistency_weight_2: {self.consistency_weight_2}")
        self.logger.info(
            f"early_stop_max_patience: {self.early_stop_max_patience}"
        )

        self._remove_config_file()

    def on_train_start(self):
        assert self.model is not None

        if self.lora_ckpt:
            self.load_model_checkpoint(self.lora_ckpt)

        self.model.train()
        self.model.to(self.device)

        (
            self.train_dataset,
            self.valid_dataset,
            self.train_dataloader,
            self.valid_dataloader,
        ) = self.get_data()
        self.max_iterations = self.num_epochs * len(self.train_dataloader)

        self.current_epoch = 0
        self.current_iter = 0
        self.current_patience = 0

        self.optimizer, self.lr_scheduler = self._get_optimizer(
            self.model,
        )
        self.supervised_loss, self.unsupervised_loss = self._get_loss()

        self._best_valid_metric = torch.inf
        self._cur_valid_metric = torch.inf

        self._best_valid_prompt_metric = torch.inf
        self._cur_valid_prompt_metric = torch.inf

        self._print_train_info()
        self._check_data_sanity()

    def _check_data_sanity(self):
        sanity_path = self.work_path / "sanity"
        sanity_path.mkdir(parents=True, exist_ok=True)

        for i in range(50):
            sample = self.train_dataset[i % 2]
            image = sample["image"]
            label = sample["label"]
            image_pil = F.to_pil_image(image).convert("RGB")
            mask_overlay = draw_mask(image_pil, label)
            mask_overlay_pil = Image.fromarray(mask_overlay)
            mask_overlay_pil.save(str(sanity_path / f"{i + 1}.png"))

    def on_train_end(self):
        self.save_state_dict(self.work_path / f"ckpt/final_model")
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

        for h in self.logger.handlers:
            h.flush()

    def on_train_epoch_start(self):
        self._train_start_time = time.time()
        self.logger.info("Train")

        self.epoch_train_outputs = []
        self.train_tqdm = iter(self.train_dataloader)

        self.model.train()

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % self.save_freq_epoch == 0:
            self.save_state_dict(self.work_path / f"epoch_{self.current_epoch}")

        train_losses = (
            torch.stack([o["loss"] for o in self.epoch_train_outputs])
            .mean(0)
            .tolist()
        )
        self.logger.info(f"Loss ({self.loss_name}): {train_losses}")

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
        metric = torch.stack(
            [o["metric"][:, :] for o in self.epoch_valid_outputs]
        ).mean(0)
        loss = torch.stack([o["loss"] for o in self.epoch_valid_outputs]).mean()
        prompt_metric = torch.stack(
            [o["prompt_metric"][:, :] for o in self.epoch_valid_outputs]
        ).mean(0)
        prompt_loss = torch.stack(
            [o["prompt_loss"] for o in self.epoch_valid_outputs]
        ).mean()

        self.logger.info(f"         DSC per class: {metric[:, 0].tolist()}")
        self.logger.info(
            f"(prompt) DSC per class: {prompt_metric[:, 0].tolist()}"
        )
        self.logger.info(f"         HD per class: {metric[:, 1].tolist()}")
        self.logger.info(
            f"(prompt) HD per class: {prompt_metric[:, 1].tolist()}"
        )
        self.logger.info(f"         avg DSC: {metric[:, 0].mean().item()}")
        self.logger.info(
            f"(prompt) avg DSC: {prompt_metric[:, 0].mean().item()}"
        )
        self.logger.info(f"         avg HD: {metric[:, 1].mean().item()}")
        self.logger.info(
            f"(prompt) avg HD: {prompt_metric[:, 1].mean().item()}"
        )
        self.logger.info(f"         loss: {loss.item()}")
        self.logger.info(f"(prompt) loss: {prompt_loss.item()}")

        if self.save_metric_name == "dice":
            self._cur_valid_metric = metric[:, 0].mean()
            self._cur_valid_prompt_metric = prompt_metric[:, 0].mean()
        elif self.save_metric_name == "hd":
            self._cur_valid_metric = metric[:, 1].mean()
            self._cur_valid_prompt_metric = prompt_metric[:, 1].mean()
        elif self.save_metric_name == "loss":
            self._cur_valid_metric = loss
            self._cur_valid_prompt_metric = prompt_loss

        is_improved = False

        if self._cur_valid_metric < self._best_valid_metric:
            self._best_valid_metric = self._cur_valid_metric
            self.logger.info(
                f"New best metric ({self.save_metric_name}): {self._cur_valid_metric}"
            )
            self.save_state_dict(self.work_path / "best_model")
            self.save_state_dict(
                self.work_path
                / f"iter_{self.current_iter}_{self._best_valid_metric:.3f}"
            )
            is_improved = True

        if self._cur_valid_prompt_metric < self._best_valid_prompt_metric:
            self._best_valid_prompt_metric = self._cur_valid_prompt_metric
            self.logger.info(
                f"New best prompt metric ({self.save_metric_name}): {self._cur_valid_prompt_metric}"
            )
            self.save_state_dict(self.work_path / "best_model_prompt")
            self.save_state_dict(
                self.work_path
                / f"iter_{self.current_iter}_{self._best_valid_metric:.3f}_prompt"
            )
            is_improved = True

        if is_improved:
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
        self.logger.info(f"Iteration {self.current_iter}:")

        self.lr_scheduler.step(self.current_iter)
        self.logger.info(f"lr: {self.lr_scheduler.get_last_lr()}")

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

        image_embeddings = self.model.sam.get_image_embeddings(image_batch)
        outputs = self.model(
            image_batch,
            multimask_output,
            self.image_size,
            image_embeddings=image_embeddings,
        )

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

        # the second round
        if self.current_iter < self.warmup_iter:
            supervised_round2_loss1 = torch.zeros(1, device=self.device)
            loss_round2_ce1 = torch.zeros(1, device=self.device)
            loss_round2_dice1 = torch.zeros(1, device=self.device)

            supervised_round2_loss1_r = torch.zeros(1, device=self.device)
            loss_round2_ce1_r = torch.zeros(1, device=self.device)
            loss_round2_dice1_r = torch.zeros(1, device=self.device)

            consistency_loss2 = torch.zeros(1, device=self.device)
            consistency_loss1_r = torch.zeros(1, device=self.device)
            loss2 = torch.zeros(1, device=self.device)
        else:
            outputs_round2 = self.model(
                image_batch,
                multimask_output,
                self.image_size,
                1,
                self.promptmode,
                image_embeddings=image_embeddings,
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

            consistency_loss2, _, _ = self.supervised_loss(
                outputs_round2_2[self.labeled_batch_size :],
                pseudo_outputs1,
                dice_weight=0.5,
            )
            consistency_loss1_r, _, _ = self.supervised_loss(
                outputs_round2_1_r[self.labeled_batch_size :],
                pseudo_outputs1,
                dice_weight=0.5,
            )

            loss2 = (
                supervised_round2_loss1
                + supervised_round2_loss1_r
                + self.consistency_weight_1 * consistency_loss2
                + self.consistency_weight_2 * consistency_loss1_r
            )

        # the third round
        if self.current_iter < self.warmup_iter:
            supervised_round3_loss2 = torch.zeros(1, device=self.device)
            loss_round3_ce1 = torch.zeros(1, device=self.device)
            loss_round3_dice1 = torch.zeros(1, device=self.device)

            supervised_round3_loss2_r = torch.zeros(1, device=self.device)
            loss_round3_ce1_r = torch.zeros(1, device=self.device)
            loss_round3_dice1_r = torch.zeros(1, device=self.device)

            consistency_loss1 = torch.zeros(1, device=self.device)
            consistency_loss2_r = torch.zeros(1, device=self.device)
            loss3 = torch.zeros(1, device=self.device)

        else:
            outputs_round3 = self.model(
                image_batch,
                multimask_output,
                self.image_size,
                0,
                self.promptmode,
                image_embeddings=image_embeddings,
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

            consistency_loss1, _, _ = self.supervised_loss(
                outputs_round3_1[self.labeled_batch_size :],
                pseudo_outputs2,
                dice_weight=0.5,
            )
            consistency_loss2_r, _, _ = self.supervised_loss(
                outputs_round3_2_r[self.labeled_batch_size :],
                pseudo_outputs2,
                dice_weight=0.5,
            )

            loss3 = (
                supervised_round3_loss2
                + supervised_round3_loss2_r
                + self.consistency_weight_1 * consistency_loss1
                + self.consistency_weight_2 * consistency_loss2_r
            )

        loss = loss1 + loss2 + loss3
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        losses = torch.Tensor([loss, loss1, loss2, loss3])
        self.logger.info(f"Loss: {losses.tolist()}")
        self.epoch_train_outputs.append({"loss": losses})
        self.current_iter += 1
        self.logger.info(f"")

    def valid_step(self, sampled_batch):
        metric, loss = test_single_volume(
            image=sampled_batch["image"],
            label=sampled_batch["label"],
            net=self.model,
            classes=self.num_classes + 1,
            patch_size=[self.image_size, self.image_size],
            loss_fn=self.supervised_loss,
        )
        prompt_metric, prompt_loss = test_single_volume_prompt(
            image=sampled_batch["image"],
            label=sampled_batch["label"],
            net=self.model,
            classes=self.num_classes + 1,
            promptidx=1,
            promptmode=self.promptmode,
            patch_size=[self.image_size, self.image_size],
            loss_fn=self.supervised_loss,
        )

        self.epoch_valid_outputs.append(
            {
                "metric": torch.Tensor(metric),
                "loss": loss,
                "prompt_metric": torch.Tensor(prompt_metric),
                "prompt_loss": prompt_loss,
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

        self.valid_tqdm.set_postfix(
            dict(metric=metric, prompt_metric=prompt_metric)
        )

    def train(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            if self.is_finished():
                break
            self.on_epoch_start()
            self.on_train_epoch_start()
            for sampled_batch in self.train_tqdm:
                if self.is_finished():
                    break
                self.train_step(sampled_batch)
                self.valid()
            self.on_train_epoch_end()
            self.on_epoch_end()

        self.on_train_end()

    def valid(self):
        if (self.current_iter) % self.valid_freq_iter == 0:
            with torch.no_grad():
                self.on_valid_epoch_start()
                for sampled_batch in self.valid_tqdm:
                    self.valid_step(sampled_batch)
                self.on_valid_epoch_end()

    def is_finished(self):
        if self.current_iter < self.min_iter:
            return False

        if self.early_stop_max_patience:
            return self.current_patience >= self.early_stop_max_patience

        return self.current_epoch >= self.num_epochs

    def run_training(self):
        self.train()
        self.perform_real_test()

    def perform_real_test(self):
        test_dataset = ACDCDataset(
            data_path=self.data_path,
            split="test",
            normalize=self._get_valid_normalize(),
            transform=self._get_valid_transform(),
            logger=self.logger,
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=1,
            pin_memory=True,
        )

        metric_list = 0.0

        first_total = 0.0
        second_total = 0.0
        third_total = 0.0
        first_list = np.zeros([len(test_dataset), 4])
        second_list = np.zeros([len(test_dataset), 4])
        third_list = np.zeros([len(test_dataset), 4])
        count = 0
        multimask_output = True
        test_save_path = self.work_path / "test_save"
        test_save_path.mkdir(parents=True, exist_ok=True)

        for data_batch in tqdm(test_dataloader):
            image = data_batch["image"]
            label = data_batch["label"]
            case = data_batch["case_name"]

            metric_i = test_single_volume_mean(
                data_path=get_path(self.data_path),
                image=image,
                label=label,
                net=self.model,
                classes=self.num_classes + 1,
                multimask_output=True,
                patch_size=[self.image_size, self.image_size],
                input_size=[self.image_size, self.image_size],
                test_save_path=test_save_path,
                case=case,
                z_spacing=ACDCDataset.Z_SPACING,
            )
            first_metric = metric_i[0]
            second_metric = metric_i[1]
            third_metric = metric_i[2]

            first_total += np.asarray(first_metric)
            second_total += np.asarray(second_metric)
            third_total += np.asarray(third_metric)

            # save
            first_list[count, 0] = first_metric[0]
            first_list[count, 1] = first_metric[1]
            first_list[count, 2] = first_metric[2]
            first_list[count, 3] = first_metric[3]

            second_list[count, 0] = second_metric[0]
            second_list[count, 1] = second_metric[1]
            second_list[count, 2] = second_metric[2]
            second_list[count, 3] = second_metric[3]

            third_list[count, 0] = third_metric[0]
            third_list[count, 1] = third_metric[1]
            third_list[count, 2] = third_metric[2]
            third_list[count, 3] = third_metric[3]

            count += 1

        avg_metric1 = np.nanmean(first_list, axis=0)
        avg_metric2 = np.nanmean(second_list, axis=0)
        avg_metric3 = np.nanmean(third_list, axis=0)

        # save:
        write_csv = self.work_path / f"test_mean.csv"
        save = pd.DataFrame(
            {
                "RV-dice": first_list[:, 0],
                "RV-hd95": first_list[:, 1],
                "RV-asd": first_list[:, 2],
                "RV-jc": first_list[:, 3],
                "Myo-dice": second_list[:, 0],
                "Myo-hd95": second_list[:, 1],
                "Myo-asd": second_list[:, 2],
                "Myo-jc": second_list[:, 3],
                "LV-dice": third_list[:, 0],
                "LV-hd95": third_list[:, 1],
                "LV-asd": third_list[:, 2],
                "LV-jc": third_list[:, 3],
            }
        )
        save.to_csv(write_csv, index=False, sep=",")

        self.logger.info("Real test results:")
        self.logger.info(f"  RV: {avg_metric1}")
        self.logger.info(f"  Myo: {avg_metric2}")
        self.logger.info(f"  LV: {avg_metric3}")
        self.logger.info(
            f"  avg: {(avg_metric1 + avg_metric2 + avg_metric3) / 3}"
        )

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer,
            "current_iter": self.current_iter,
            "current_epoch": self.current_epoch,
        }

    def load_state_dict(self, save_path: str | Path):
        save_path = get_path(save_path)
        self.load_model_checkpoint(save_path / "model.pth")

        training_state = torch.load(save_path / "training_state.pth")
        self.optimizer.load_state_dict(training_state["optimizer"])
        self.current_epoch = training_state["current_epoch"]
        self.current_iter = training_state["current_iter"]

    def save_state_dict(self, save_path: str | Path):
        save_path = get_path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        self.save_model_checkpoint(save_path / "model.pth")

        state_dict = self.state_dict()
        state_dict.pop("model")
        torch.save(state_dict, save_path / "training_state.pth")

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
