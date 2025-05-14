import random
import os
from functools import partial
from datetime import datetime
import json
from typing import Literal, Any
import logging
import time
from pathlib import Path

import wandb
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
from losses.dice_loss import DiceLoss
from losses.contrastive_loss import PrototypeContrastiveLoss
from losses.adv_loss import VAT2d
from memories.feature_memory import FeatureMemory
from scheduler.lr_scheduler import PolyLRScheduler
from scheduler.ramps import SigmoidRampUp

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


class CPCSAMConfig(object):
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
        dropout_rate: float = 0.0,
        num_points_prompt: int | list[int] | tuple[int, int] = (1, 2),
        bbox_change_rate: float | list[float] | tuple[float, float] = (
            0.1,
            0.2,
        ),
        # Data parameters
        dataset: Literal["ACDC"] = "ACDC",
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
        maximum_save_metric: bool | None = None,
        ## Loss parameters
        loss_name: Literal["dice+ce"] = "dice+ce",
        dice_weight: float = 0.8,
        ### Loss 2: cross-prompting loss
        loss2_weight: float = 1.0,
        loss2_weight_rampup_interval: int = 100,
        loss2_weight_rampup_iter: int = 0,
        consistency_weight_1: float = 0.4,
        consistency_weight_2: float = 0.05,
        early_stop_max_patience: int | None = None,
        ### Loss 3: contrastive loss
        loss3_weight: float = 0.1,
        loss3_weight_rampup_interval: int = 100,
        loss3_weight_rampup_iter: int = 15000,
        use_contrastive_loss: bool = False,
        contrastive_dropout_rate: float = 0.0,
        contrastive_weight: float = 0.1,
        use_adv_loss: bool = False,
        adv_weight: float = 1.0,
        adv_loss_kwargs: dict = {
            "xi": 10.0,
            "epi": 6.0,
            "ip": 1,
        },
        # Inference parameters
        stride: int | tuple[int, ...] | list[int] | None = None,
        # Misc parameters
        exp_name: str = "",
        **kwargs,
    ):
        self._config_dict = {}

        self.seed = seed

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
        self.dropout_rate = dropout_rate
        self.num_points_prompt = num_points_prompt
        self.bbox_change_rate = bbox_change_rate
        # <<< Model parameters

        # >>> Data parameters
        self.dataset = dataset
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
        self.maximum_save_metric = maximum_save_metric
        self.loss_name = loss_name
        self.dice_weight = dice_weight
        # > Loss 2: cross-prompting loss
        self.loss2_weight = loss2_weight
        self.loss2_weight_rampup_interval = loss2_weight_rampup_interval
        self.loss2_weight_rampup_iter = loss2_weight_rampup_iter
        self.consistency_weight_1 = consistency_weight_1
        self.consistency_weight_2 = consistency_weight_2
        self.early_stop_max_patience = early_stop_max_patience
        # > Loss 3: contrastive loss
        self.loss3_weight = loss3_weight
        self.loss3_weight_rampup_interval = loss3_weight_rampup_interval
        self.loss3_weight_rampup_iter = loss3_weight_rampup_iter
        self.use_contrastive_loss = use_contrastive_loss
        self.contrastive_dropout_rate = contrastive_dropout_rate
        self.contrastive_weight = contrastive_weight
        self.use_adv_loss = use_adv_loss
        self.adv_weight = adv_weight
        self.adv_loss_kwargs = adv_loss_kwargs
        # <<< Training parameters

        # >>> Inference parameters
        self.stride = stride
        # <<< Inference parameters

        self.exp_name = exp_name

    def __setattr__(self, name: str, value: Any, /) -> None:
        if hasattr(self, "_config_dict"):
            self._config_dict[name] = value

        super(CPCSAMConfig, self).__setattr__(name, value)

    def save(self, save_path: str | Path):
        save_path = get_path(save_path)

        with open(save_path, "w") as f:
            json.dump(self._config_dict, f, indent=2)

    def load(self, save_path: str | Path):
        save_path = get_path(save_path)

        with open(save_path, "r") as f:
            data = json.load(f)

        for k, v in data.items():
            self.__setattr__(k, v)

        return self


def _worker_init_fn(worker_id):
    seed = int(os.environ["CPCSAM_SEED"] or 0)
    random.seed(seed + worker_id)


class CPCSAMTrainer(BaseTrainer):
    def __init__(
        self,
        work_path: Path | str = Path.cwd(),
        device: torch.device | str = torch.device("cuda"),
        config: CPCSAMConfig | dict | str | Path | None = None,
        # Log parameters
        verbose: bool = True,
        log_path: Path | str | None = None,
        config_path: Path | str | None = None,
        log_mode: str = "a",
        log_override: bool = False,
        use_wandb: bool = False,
        wandb_api_key: str | None = None,
        **kwargs,
    ):
        if isinstance(config, CPCSAMConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = CPCSAMConfig(**config)
        elif isinstance(config, (str, Path)):
            self.config = CPCSAMConfig().load(config)
        else:
            self.config = CPCSAMConfig()

        self.work_path = get_path(work_path)
        self.device = torch.device("cpu")
        self.to(device)

        self._set_seed(self.config.seed)

        self.current_epoch = 0

        # >>> Log parameters
        self.verbose = verbose
        self.log_path = log_path
        self.config_path = config_path
        self.log_mode = log_mode
        self.log_override = log_override
        self.use_wandb = use_wandb
        self.wandb_api_key = wandb_api_key
        # <<< Log parameters

    def initialize(self):
        self._set_snapshot_work_dir()
        self._setup_wandb()
        self._setup_logger()
        self._build_model()
        if self.config.lora_ckpt:
            self.load_model_checkpoint(self.config.lora_ckpt)
        self.model.to(self.device)

    def _set_snapshot_work_dir(self):
        current_time_str = datetime.now().strftime("%Y%m%d_%H")
        snapshot_list = [
            f"ACDC",
            f"{current_time_str}",
            f"patchsz-{self.config.patch_size}",
            f"imgsz-{self.config.image_size}",
            f"lora-{self.config.lora_rank}",
            f"prompt-{self.config.promptmode}",
            f"dropout-{self.config.dropout_rate}",
            f"point-{self.config.num_points_prompt}",
            f"bbox-{self.config.bbox_change_rate}",
            f"labeled-{self.config.labeled_num}",
            f"batchsz-{self.config.batch_size}",
            f"epoch-{self.config.num_epochs}",
            f"optimizer-{self.config.optimizer_name}",
            f"lr-{self.config.lr_scheduler_name}",
            f"lrwarm-{self.config.lr_warmup_iter}",
            f"warmiter-{self.config.warmup_iter}",
            f"startlr-{self.config.start_lr}",
            f"dice-{self.config.dice_weight}",
            f"coe1-{self.config.consistency_weight_1}",
            f"coe2-{self.config.consistency_weight_2}",
        ]
        if self.config.use_contrastive_loss:
            snapshot_list.append("contrastive")
        if self.config.use_adv_loss:
            snapshot_list.append("adv")
        if self.config.exp_name:
            snapshot_list.append(self.config.exp_name)
        snapshot_str = "_".join(snapshot_list)
        self.work_path = self.work_path / snapshot_str
        self.work_path.mkdir(parents=True, exist_ok=True)

    def _setup_wandb(self):
        if self.use_wandb:
            wandb.login(key=self.wandb_api_key)
            self.wandb_runner = wandb.init(
                dir=self.work_path / "wandb",
                project="CPC-SAM",
                name=self.work_path.stem,
                config=self.config._config_dict,
            )

            wandb.define_metric("train_epoch")
            wandb.define_metric("train/epoch/*", step_metric="train_epoch")
            wandb.define_metric("train_iter")
            wandb.define_metric("train/iter/*", step_metric="train_iter")
            wandb.define_metric("valid_step")
            wandb.define_metric("valid/*", step_metric="valid_step")

    def _set_seed(self, seed: int):
        os.environ["CPCSAM_SEED"] = str(seed)

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
            self.log_path = self.work_path / "log.txt"

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

    def _save_config(self):
        if not self.config_path:
            config_json = self.work_path / "config.json"
        else:
            self.config_path = get_path(self.config_path)
            config_json = (
                self.config_path.parent / f"{self.config_path.stem}.json"
            )

        self.config.save(config_json)
        if self.use_wandb:
            wandb.log_artifact(
                config_json,
                name=f"config_{self.wandb_runner.id}",
                type="config",
                aliases=["json"],
            )

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
        self.sam, self.img_embedding_size = sam_model_registry[
            self.config.sam_name
        ](
            image_size=self.config.image_size,
            num_classes=self.config.num_classes,
            checkpoint=self.config.model_ckpt,
            pixel_mean=[0, 0, 0],
            pixel_std=[1, 1, 1],
            dropout_rate=self.config.dropout_rate,
            num_points_prompt=self.config.num_points_prompt,
            bbox_change_rate=self.config.bbox_change_rate,
        )
        self.model = LoRA_Sam(self.sam, self.config.lora_rank)

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

    def get_data(self):
        train_dataset = ACDCDataset(
            data_path=self.config.data_path,
            split="train",
            normalize=self._get_train_normalize(),
            transform=self._get_train_transform(),
            logger=self.logger,
        )
        total_slices = len(train_dataset)
        labeled_slices = self.patients_to_slices(
            "ACDC", self.config.labeled_num
        )
        labeled_indices = list(range(0, labeled_slices))
        unlabeled_indices = list(range(labeled_slices, total_slices))
        batch_sampler = TwoStreamBatchSampler(
            labeled_indices,
            unlabeled_indices,
            self.config.batch_size,
            self.config.batch_size - self.config.labeled_batch_size,
        )

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.config.num_workers,
            worker_init_fn=_worker_init_fn,
            pin_memory=self.config.pin_memory,
        )

        valid_dataset = ACDCDataset(
            data_path=self.config.data_path,
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
        if self.config.do_augment:
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

        if self.config.image_size:
            transforms.append(JointResize(self.config.image_size))

        return ComposeTransform(transforms)

    def _get_train_normalize(self):
        if self.config.do_normalize:
            return ZScoreNormalize()
        else:
            return None

    def _get_valid_transform(self):
        return None

    def _get_valid_normalize(self):
        if self.config.do_normalize:
            return ZScoreNormalize()
        else:
            return None

    def _get_optimizer(
        self,
        model: nn.Module,
    ):
        parameters = filter(lambda p: p.requires_grad, model.parameters())

        if self.config.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                parameters,
                betas=(0.9, 0.999),
                weight_decay=0.1,
                **self.config.optimizer_kwargs,
            )
        elif self.config.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                parameters,
                betas=(0.9, 0.999),
                weight_decay=0.1,
                **self.config.optimizer_kwargs,
            )
        elif self.config.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                parameters,
                momentum=0.9,
                weight_decay=0.001,
                **self.config.optimizer_kwargs,
            )
        else:
            raise ValueError(
                f'Optimizer "{self.config.optimizer_name}" not supported'
            )

        if self.config.lr_scheduler_name == "poly":
            lr_scheduler = PolyLRScheduler(
                optimizer,
                self.config.start_lr,
                self.max_iterations,
                self.config.lr_warmup_iter,
            )
        else:
            raise ValueError(
                f'Learning rate scheduler "{self.config.lr_scheduler_name}" not supported'
            )

        return optimizer, lr_scheduler

    def _setup_loss(self):
        if self.config.loss_name == "dice+ce":
            self.supervised_loss = DiceAndCELoss(
                dice_loss=DiceLoss,
                dice_kwargs={
                    "num_classes": self.config.num_classes,
                    "smooth": 1e-5,
                    "do_bg": True,
                },
                ce_loss=torch.nn.CrossEntropyLoss,
                ce_kwargs={},
                default_dice_weight=0.8,
            )
            self.unsupervised_loss = DualBranchDiceAndCELoss(
                dice_loss=DiceLoss,
                dice_kwargs={
                    "num_classes": self.config.num_classes,
                    "smooth": 1e-5,
                    "do_bg": True,
                },
                ce_loss=torch.nn.CrossEntropyLoss,
                ce_kwargs={},
                default_dice_weight=0.8,
            )
        else:
            raise ValueError(f"Loss function {self.config.loss_name} not found")

        if self.config.use_contrastive_loss:
            self.contrastive_loss = PrototypeContrastiveLoss(
                self.model.sam,
                num_classes=self.config.num_classes,
                memory_cls=FeatureMemory,
                memory_kwargs={"elements_per_class": 32},
            )

        if self.config.use_adv_loss:
            self.adv_loss = VAT2d(**self.config.adv_loss_kwargs)

        # Setup rampup functions for loss weights
        self.loss2_weight_rampup = SigmoidRampUp(
            self.config.loss2_weight,
            max_steps=self.config.loss2_weight_rampup_iter,
            interval=self.config.loss2_weight_rampup_interval,
        )

        self.loss3_weight_rampup = SigmoidRampUp(
            self.config.loss3_weight,
            max_steps=self.config.loss3_weight_rampup_iter,
            interval=self.config.loss3_weight_rampup_interval,
        )

    def _print_train_info(self):
        self._add_config_file()
        self._save_config()

        self.logger.info(f"Training summary")
        self.logger.info("")
        self.logger.info(f"device: {self.device}")
        self.logger.info(f"seed: {self.seed}")
        self.logger.info(f'log_file: "{self.log_path}"')
        if self.use_wandb:
            self.logger.info(f"wandb_run_id: {self.wandb_runner.id}")

        self.logger.info(f"model: {self.model}")
        self.logger.info(f"  num_classes: {self.config.num_classes}")
        self.logger.info(f"  patch_size: {self.config.patch_size}")
        self.logger.info(f"  image_size: {self.config.image_size}")
        self.logger.info(f"  pretrained_model: {self.config.model_ckpt}")
        self.logger.info(f"  sam_name: {self.config.sam_name}")
        self.logger.info(f"  model_ckpt: {self.config.model_ckpt}")
        self.logger.info(f"  lora_rank: {self.config.lora_rank}")
        self.logger.info(f"  lora_ckpt: {self.config.lora_ckpt}")
        self.logger.info(f"  promptmode: {self.config.promptmode}")
        self.logger.info(f"  dropout_rate: {self.config.dropout_rate}")
        self.logger.info(
            f"  num_points_prompt: {self.config.num_points_prompt}"
        )
        self.logger.info(f"  bbox_change_rate: {self.config.bbox_change_rate}")

        self.logger.info(f"data: {self.config.dataset}")
        self.logger.info(f"  data_path: {self.config.data_path}")
        self.logger.info(f"  train_size (slices): {len(self.train_dataset)}")
        self.logger.info(
            f"  labeled_patients (slices): {self.config.labeled_num} ({self.patients_to_slices('ACDC', self.config.labeled_num)})"
        )
        self.logger.info(f"  valid_size (volumns): {len(self.valid_dataset)}")
        self.logger.info(f"  do_augment: {self.config.do_augment}")
        if self.config.do_augment:
            self.logger.info(
                f"{json.dumps(self._get_train_transform().get_params_dict(), indent=1)}"
            )
        self.logger.info(f"  normalize: {self.config.do_normalize}")
        self.logger.info(f"  batch_size: {self.config.batch_size}")
        self.logger.info(
            f"  labeled_batch_size: {self.config.labeled_batch_size}"
        )
        self.logger.info(f"  num_workers: {self.config.num_workers}")
        self.logger.info(f"  pin_memory: {self.config.pin_memory}")

        self.logger.info(f"optimizer: {self.config.optimizer_name}")
        self.logger.info(f"  lr_warmup_iter: {self.config.lr_warmup_iter}")
        self.logger.info(f"  lr_scheduler: {self.config.lr_scheduler_name}")
        self.logger.info(f"  start_lr: {self.config.start_lr}")
        self.logger.info(f"  optimizer_kwargs: {self.config.optimizer_kwargs}")
        self.logger.info(f"loss_fn: {self.config.loss_name}")
        self.logger.info(f"save_metric: {self.config.save_metric_name}")
        self.logger.info(f"start_epoch: {self.current_epoch}")
        self.logger.info(f"num_epochs: {self.config.num_epochs}")
        self.logger.info(f"warmup_iter: {self.config.warmup_iter}")
        self.logger.info(f"save_freq_epoch: {self.config.save_freq_epoch}")
        self.logger.info(f"valid_freq_iter: {self.config.valid_freq_iter}")
        self.logger.info(f"dice_weight: {self.config.dice_weight}")
        self.logger.info(f"loss2_weight: {self.config.loss2_weight}")
        self.logger.info(
            f"loss2_weight_rampup_iter: {self.config.loss2_weight_rampup_iter}"
        )
        self.logger.info(
            f"loss2_weight_rampup_interval: {self.config.loss2_weight_rampup_interval}"
        )
        self.logger.info(
            f"consistency_weight_1: {self.config.consistency_weight_1}"
        )
        self.logger.info(
            f"consistency_weight_2: {self.config.consistency_weight_2}"
        )
        self.logger.info(
            f"early_stop_max_patience: {self.config.early_stop_max_patience}"
        )
        self.logger.info(f"loss3_weight: {self.config.loss3_weight}")
        self.logger.info(
            f"loss3_weight_rampup_iter: {self.config.loss3_weight_rampup_iter}"
        )
        self.logger.info(
            f"loss3_weight_rampup_interval: {self.config.loss3_weight_rampup_interval}"
        )
        if self.config.use_contrastive_loss:
            self.logger.info(
                f"contrastive_dropout_rate: {self.config.contrastive_dropout_rate}"
            )
            self.logger.info(
                f"contrastive_weight: {self.config.contrastive_weight}"
            )

        if self.config.use_adv_loss:
            self.logger.info(f"adv_weight: {self.config.adv_weight}")

        self._remove_config_file()

        if self.use_wandb and self.config_path:
            wandb.log_artifact(
                self.config_path,
                name=f"config_{self.wandb_runner.id}",
                type="config",
                aliases=["txt"],
            )

    def on_train_start(self):
        assert self.model is not None

        if self.config.lora_ckpt:
            self.load_model_checkpoint(self.config.lora_ckpt)

        self.model.train()
        self.model.to(self.device)

        (
            self.train_dataset,
            self.valid_dataset,
            self.train_dataloader,
            self.valid_dataloader,
        ) = self.get_data()
        self.max_iterations = self.config.num_epochs * len(
            self.train_dataloader
        )

        self.current_epoch = 0
        self.current_iter = 0
        self.current_patience = 0

        self.optimizer, self.lr_scheduler = self._get_optimizer(
            self.model,
        )
        self._setup_loss()

        if self.config.maximum_save_metric is None:
            if self.config.save_metric_name == "dice":
                self.config.maximum_save_metric = True
            elif self.config.save_metric_name == "hd":
                self.config.maximum_save_metric = False
            elif self.config.save_metric_name == "loss":
                self.config.maximum_save_metric = False
            else:
                raise ValueError(
                    f"{self.config.save_metric_name} is not a valid save metric"
                )

        default_metric = torch.tensor(
            -torch.inf if self.config.maximum_save_metric else torch.inf
        )
        self._best_valid_metric = default_metric
        self._cur_valid_metric = default_metric

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
        ckpt_path = self.work_path / f"ckpt/final_model"
        self.save_state_dict(ckpt_path, True)
        if self.use_wandb:
            wandb.log_model(
                ckpt_path,
                name=f"model_{self.wandb_runner.id}",
                aliases=[f"epoch_{self.current_epoch}", "final"],
            )

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
        if (self.current_epoch + 1) % self.config.save_freq_epoch == 0:
            ckpt_path = self.work_path / f"epoch_{self.current_epoch}"
            self.save_state_dict(ckpt_path, True)

            if self.use_wandb:
                wandb.log_model(
                    ckpt_path,
                    name=f"model_{self.wandb_runner.id}",
                    aliases=[f"epoch_{self.current_epoch}"],
                )

        train_losses = (
            torch.stack([o["loss"] for o in self.epoch_train_outputs])
            .mean(0)
            .tolist()
        )
        self.logger.info(f"Loss ({self.config.loss_name}): {train_losses}")

        self._train_end_time = time.time()
        time_elapsed = self._train_end_time - self._train_start_time
        self.logger.info(f"Train time elapsed: {time_elapsed:.3f} seconds")

        if self.use_wandb:
            train_metric = {
                "train/epoch/losses/loss": train_losses[0],
                "train/epoch/losses/loss1": train_losses[1],
                "train/epoch/losses/loss2": train_losses[2],
                "train/epoch/losses/loss3": train_losses[3],
                "train_epoch": self.current_epoch,
                "train_iter": self.current_iter,
            }
            self.wandb_runner.log(train_metric)

    def on_valid_epoch_start(self):
        self._valid_start_time = time.time()
        self.logger.info("Valid")

        self.model.eval()
        self.valid_tqdm = tqdm(self.valid_dataloader)
        self.epoch_valid_outputs = []

    def _is_improved(self, old_metric, new_metric, maximum):
        if maximum:
            return old_metric < new_metric
        else:
            return old_metric > new_metric

    def on_valid_epoch_end(self):
        metric = torch.stack(
            [o["metric"][:, :] for o in self.epoch_valid_outputs]
        ).nanmean(0)
        loss = torch.stack(
            [o["loss"] for o in self.epoch_valid_outputs]
        ).nanmean()

        dsc_per_class = metric[:, 0]
        avg_dsc = dsc_per_class.mean()

        hd_per_class = metric[:, 1]
        avg_hd = hd_per_class.nanmean()

        self.logger.info(f"DSC per class: {dsc_per_class.tolist()}")
        self.logger.info(f"HD per class: {hd_per_class.tolist()}")
        self.logger.info(f"avg DSC: {avg_dsc.item()}")
        self.logger.info(f"avg HD: {avg_hd.item()}")
        self.logger.info(f"loss: {loss.item()}")

        if self.config.save_metric_name == "dice":
            self._cur_valid_metric = avg_dsc
        elif self.config.save_metric_name == "hd":
            self._cur_valid_metric = avg_hd
        elif self.config.save_metric_name == "loss":
            self._cur_valid_metric = loss
        else:
            raise ValueError(
                f"{self.config.save_metric_name} is not a valid save metric"
            )

        if self.use_wandb:
            valid_metric = {
                "valid/metric/dsc": avg_dsc.item(),
                "valid/metric/hd95": avg_hd.item(),
                "valid/metric/loss": loss.item(),
                "train_epoch": self.current_epoch,
                "train_iter": self.current_iter,
                "valid_step": self.current_iter,
            }
            for i in range(self.config.num_classes):
                valid_metric[f"valid/metric_per_cls/dsc/class_{i}"] = (
                    dsc_per_class[i].item()
                )
                valid_metric[f"valid/metric_per_cls/hd95/class_{i}"] = (
                    hd_per_class[i].item()
                )

            self.wandb_runner.log(valid_metric)

        is_improved = False

        if self._is_improved(
            self._best_valid_metric,
            self._cur_valid_metric,
            self.config.maximum_save_metric,
        ):
            self._best_valid_metric = self._cur_valid_metric
            self.logger.info(
                f"New best metric ({self.config.save_metric_name}): {self._cur_valid_metric}"
            )
            self.save_state_dict(self.work_path / "best_model")

            ckpt_path = (
                self.work_path
                / f"iter_{self.current_iter}_{self._best_valid_metric:.4f}"
            )
            self.save_state_dict(ckpt_path)

            if self.use_wandb:
                wandb.log_model(
                    ckpt_path,
                    name=f"best_model_{self.wandb_runner.id}",
                    aliases=[
                        f"iter_{self.current_iter}",
                        f"{self.config.save_metric_name}_{self._best_valid_metric:.4f}",
                    ],
                )
            is_improved = True

        if is_improved:
            self.current_patience = 0
            text_lines = [
                f"iter={self.current_iter}",
                f"epoch={self.current_epoch})",
                f"metric={self._best_valid_metric.item():.4f}",
                f"dsc="
                + "["
                + ", ".join([f"{x:.4f}" for x in dsc_per_class.tolist()])
                + "]",
                f"average_dsc={avg_dsc.item():.4f}",
                f"hd95="
                + "["
                + ", ".join([f"{x:.4f}" for x in hd_per_class.tolist()])
                + "]",
                f"average_hd95={avg_hd.item():.4f}",
                f"loss={loss.item():.4f}",
            ]
            self.wandb_runner.alert(
                title="Improved Performance",
                text="; ".join(text_lines),
                level="INFO",
            )
        else:
            self.current_patience += 1
            if self.config.early_stop_max_patience:
                alert_threshold = self.config.early_stop_max_patience * 0.5
                if self.current_patience >= alert_threshold:
                    self.wandb_runner.alert(
                        title="Performance Stagnation",
                        text=f"Performance is not improved for {self.current_patience} step",
                        level="WARN",
                    )

        self._valid_end_time = time.time()
        time_elapsed = self._valid_start_time - self._valid_start_time
        self.logger.info(f"current_patience: {self.current_patience}")
        self.logger.info(f"Valid time elapsed: {time_elapsed:.3f} seconds")

    def _get_one_hot_output(self, output: torch.Tensor):
        output_seg = output.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(
            output.shape, device=output.device, dtype=torch.int32
        )
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)

        return predicted_segmentation_onehot

    def train_step(self, sampled_batch):
        self.model.train()

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
            self.config.image_size,
            image_embeddings=image_embeddings,
        )
        num_decoders = len(self.model.sam.mask_decoders)

        labeled_label_batch = label_batch[: self.config.labeled_batch_size]

        loss1 = torch.zeros(1, device=self.device)
        for i in range(num_decoders):
            low_res_logits = outputs["low_res_logits"][i]
            labeled_low_res_logits = low_res_logits[
                : self.config.labeled_batch_size
            ]
            supervised_loss, _, _ = self.supervised_loss(
                labeled_low_res_logits,
                labeled_label_batch,
                self.config.dice_weight,
            )
            loss1 += supervised_loss

        labeled_features_list = []
        labeled_predictions_list = []

        unlabeled_features_list = []
        unlabeled_predictions_list = []

        if self.config.use_contrastive_loss:
            for i in range(num_decoders):
                dense_features = outputs["dense_features"][i]
                soft_masks = outputs["masks"][i].softmax(1)

                labeled_features_list.append(
                    dense_features[: self.config.labeled_batch_size]
                )
                labeled_predictions_list.append(
                    soft_masks[: self.config.labeled_batch_size].argmax(1)
                )

                unlabeled_features_list.append(
                    dense_features[self.config.labeled_batch_size :]
                )
                unlabeled_predictions_list.append(
                    soft_masks[self.config.labeled_batch_size :].argmax(1)
                )

        # the second round
        supervised_loss_round2 = torch.zeros(1, device=self.device)
        supervised_loss_round2_r = torch.zeros(1, device=self.device)

        consistency_loss_round2 = torch.zeros(1, device=self.device)
        consistency_loss_round2_r = torch.zeros(1, device=self.device)

        loss2 = torch.zeros(1, device=self.device)

        if self.current_iter >= self.config.warmup_iter:
            for prompt_idx in range(num_decoders):
                outputs_round2 = self.model(
                    image_batch,
                    multimask_output,
                    self.config.image_size,
                    prompt_idx,
                    self.config.promptmode,
                    image_embeddings,
                )
                dense_features = outputs_round2["dense_features"][prompt_idx]
                dense_features_r = outputs_round2["dense_features_r"][
                    prompt_idx
                ]

                low_res_logits_prompt = outputs_round2["low_res_logits"][
                    prompt_idx
                ]
                low_res_logits_prompt_r = outputs_round2["low_res_logits_r"][
                    prompt_idx
                ]
                # Update the list of labeled and unlabled features/predictions
                # if we use contrastive loss
                if self.config.use_contrastive_loss:
                    labeled_features_list.append(
                        dense_features[: self.config.labeled_batch_size]
                    )
                    unlabeled_features_list.append(
                        dense_features[self.config.labeled_batch_size :]
                    )
                    labeled_predictions_list.append(
                        low_res_logits_prompt[: self.config.labeled_batch_size]
                        .softmax(1)
                        .argmax(1)
                    )
                    unlabeled_predictions_list.append(
                        low_res_logits_prompt[self.config.labeled_batch_size :]
                        .softmax(1)
                        .argmax(1)
                    )

                # Compute the supervised loss of the prompt output
                labeled_low_res_logits_prompt = low_res_logits_prompt[
                    : self.config.labeled_batch_size
                ]
                labeled_low_res_logits_prompt_r = low_res_logits_prompt_r[
                    : self.config.labeled_batch_size
                ]

                supervised_loss, _, _ = self.supervised_loss(
                    labeled_low_res_logits_prompt,
                    labeled_label_batch,
                    self.config.dice_weight,
                )
                supervised_loss_r, _, _ = self.supervised_loss(
                    labeled_low_res_logits_prompt_r,
                    labeled_label_batch,
                    self.config.dice_weight,
                )

                supervised_loss_round2 += supervised_loss
                supervised_loss_round2_r += supervised_loss_r

                # Compute the consistency loss between the prompt output and raw
                # logits of other decoders
                ensemble_low_res_logits_prompt = (
                    low_res_logits_prompt.softmax(1)
                    + low_res_logits_prompt_r.softmax(1)
                ) / 2.0
                pseudo_label_prompt = (
                    ensemble_low_res_logits_prompt[
                        self.config.labeled_batch_size :
                    ]
                    .detach()
                    .argmax(1)
                    .long()
                )

                for id in range(num_decoders):
                    if id != prompt_idx:
                        consistency_loss, _, _ = self.supervised_loss(
                            outputs_round2["low_res_logits"][id][
                                self.config.labeled_batch_size :
                            ],
                            pseudo_label_prompt,
                            0.5,
                        )
                        consistency_loss_round2 += consistency_loss

                consistency_loss_r, _, _ = self.supervised_loss(
                    low_res_logits_prompt_r[self.config.labeled_batch_size :],
                    pseudo_label_prompt,
                    0.5,
                )
                consistency_loss_round2_r += consistency_loss_r

            loss2 = (
                supervised_loss_round2
                + supervised_loss_round2_r
                + self.config.consistency_weight_1 * consistency_loss_round2
                + self.config.consistency_weight_2 * consistency_loss_round2_r
            )

        contrastive_loss = torch.zeros(1, device=self.device)
        if self.config.use_contrastive_loss:
            labeled_features = torch.cat(labeled_features_list, dim=0)
            labeled_predictions = torch.cat(labeled_predictions_list, dim=0)
            unlabeled_features = torch.cat(unlabeled_features_list, dim=0)
            unlabeled_predictions = torch.cat(unlabeled_predictions_list, dim=0)
            labeled_labels = labeled_label_batch.repeat(
                len(labeled_features_list), 1, 1
            )

            self.contrastive_loss.update_memory(
                features=labeled_features,
                predictions=labeled_predictions,
                labels=labeled_labels,
            )
            contrastive_loss = contrastive_loss + self.contrastive_loss(
                labeled_features,
                labeled_labels,
                self.config.contrastive_dropout_rate,
            )
            contrastive_loss = contrastive_loss + self.contrastive_loss(
                unlabeled_features,
                unlabeled_predictions,
                self.config.contrastive_dropout_rate,
            )

        loss3 = torch.zeros(1, device=self.device)

        if self.config.use_contrastive_loss:
            loss3 = loss3 + self.config.contrastive_weight * contrastive_loss

        loss2_weight = self.loss2_weight_rampup.step(self.current_iter)
        loss3_weight = self.loss3_weight_rampup.step(self.current_iter)

        loss = loss1 + loss2_weight * loss2 + loss3_weight * loss3

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        losses = torch.Tensor(
            [
                loss.detach(),
                loss1.detach(),
                loss2.detach(),
                loss3.detach(),
            ]
        )
        self.logger.info(f"Loss: {losses.tolist()}")
        self.epoch_train_outputs.append({"loss": losses})
        self.logger.info(f"")

        if self.use_wandb:
            lr = self.lr_scheduler.get_last_lr()[0]
            train_metric = {
                "train/iter/lr": lr,
                "train/iter/loss2_weight": loss2_weight,
                "train/iter/loss3_weight": loss3_weight,
                "train/iter/losses/loss": losses[0],
                "train/iter/losses/loss1": losses[1],
                "train/iter/losses/loss2": losses[2],
                "train/iter/losses/loss3": losses[3],
                "train_epoch": self.current_epoch,
                "train_iter": self.current_iter,
            }
            self.wandb_runner.log(train_metric)

        self.current_iter += 1

    def valid_step(self, sampled_batch):
        metric, loss = test_single_volume(
            image=sampled_batch["image"],
            label=sampled_batch["label"],
            net=self.model,
            classes=self.config.num_classes + 1,
            patch_size=[self.config.image_size, self.config.image_size],
            loss_fn=self.supervised_loss,
        )

        self.epoch_valid_outputs.append(
            {
                "metric": torch.Tensor(metric),
                "loss": loss,
            }
        )

    def train(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.config.num_epochs):
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
        if (self.current_iter) % self.config.valid_freq_iter == 0:
            with torch.no_grad():
                self.on_valid_epoch_start()
                for sampled_batch in self.valid_tqdm:
                    self.valid_step(sampled_batch)
                self.on_valid_epoch_end()

    def is_finished(self):
        if self.current_iter < self.config.min_iter:
            return False

        if self.config.early_stop_max_patience:
            is_finished = (
                self.current_patience >= self.config.early_stop_max_patience
            )
            if is_finished:
                self.logger.info(
                    "Exceeded maximum patience. Training will be early stopped"
                )
            return is_finished

        return self.current_epoch >= self.config.num_epochs

    def run_training(self):
        self.train()
        self.perform_real_test()

    def perform_real_test(self):
        best_model_path = self.work_path / "best_model"
        if best_model_path.exists():
            try:
                self.load_state_dict(self.work_path / "best_model")
            except:
                pass

        test_dataset = ACDCDataset(
            data_path=self.config.data_path,
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

        multimask_output = True
        test_save_path = self.work_path / "test_save"
        test_save_path.mkdir(parents=True, exist_ok=True)

        metric_list = []

        for data_batch in tqdm(test_dataloader):
            image = data_batch["image"]
            label = data_batch["label"]
            case = data_batch["case_name"][0]

            metric_i = test_single_volume_mean(
                data_path=get_path(self.config.data_path),
                image=image,
                label=label,
                net=self.model,
                classes=self.config.num_classes + 1,
                multimask_output=multimask_output,
                patch_size=[self.config.image_size, self.config.image_size],
                input_size=[self.config.image_size, self.config.image_size],
                test_save_path=test_save_path,
                case=case,
                z_spacing=ACDCDataset.Z_SPACING,
            )

            metric_list.append(metric_i)

        metric_tensor = torch.from_numpy(
            np.array(metric_list)
        )  # N, C, 4 (DSC, HD, ASD, JSD)
        classes = test_dataset.CLASSES
        metric_name = {0: "DSC", 1: "HD95", 2: "ASD", 3: "JSD"}

        dataframe_dict = {}
        for class_id in classes.keys():
            if class_id == 0:
                continue

            for metric_id in metric_name.keys():
                field_name = f"{classes[class_id]}-{metric_name[metric_id]}"
                dataframe_dict[field_name] = metric_tensor[
                    :, class_id - 1, metric_id
                ].tolist()

        if self.use_wandb:
            wandb_table = wandb.Table(
                columns=list(dataframe_dict.keys()),
                data=list(zip(*list(dataframe_dict.values()))),
            )
            self.wandb_runner.log({"test_performance": wandb_table})

        avg_metric = metric_tensor.mean(0)
        self.logger.info("Real test results:")
        for id in classes.keys():
            if id == 0:
                continue

            self.logger.info(f"  {classes[id]}: {avg_metric[id-1].tolist()}")

        self.logger.info(f"Average: {avg_metric.mean(0).tolist()}")

        # save:
        write_csv = self.work_path / f"test_mean.csv"
        save = pd.DataFrame(dataframe_dict)
        save.to_csv(write_csv, index=False, sep=",")

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer,
            "current_iter": self.current_iter,
            "current_epoch": self.current_epoch,
        }

    def load_state_dict(self, save_path: str | Path):
        save_path = get_path(save_path)
        model_path = save_path / "model.pth"
        training_state_path = save_path / "training_state.pth"
        if model_path.is_file():
            self.load_model_checkpoint(model_path)

        if training_state_path.is_file():
            training_state = torch.load(training_state_path)
            self.optimizer.load_state_dict(training_state["optimizer"])
            self.current_epoch = training_state["current_epoch"]
            self.current_iter = training_state["current_iter"]

    def save_state_dict(
        self, save_path: str | Path, save_training_state: bool = False
    ):
        save_path = get_path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        self.save_model_checkpoint(save_path / "model.pth")

        if save_training_state:
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
