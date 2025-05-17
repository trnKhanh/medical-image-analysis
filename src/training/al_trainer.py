import random
import os
from functools import partial
from datetime import datetime
import json
from typing import Literal, Any, Sequence
import logging
import time
from pathlib import Path

import wandb
import torch
from torch import nn
import torch.nn.functional as N
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader

from medpy import metric

import numpy as np
import pandas as pd

from PIL import Image

from rich.logging import RichHandler
from rich.console import Console

from tqdm import tqdm

from .base_trainer import BaseTrainer
from datasets import ACDCDataset, ActiveDataset, ExtendableDataset
from losses.compound_losses import DiceAndCELoss, DualBranchDiceAndCELoss
from losses.dice_loss import DiceLoss
from losses.contrastive_loss import PrototypeContrastiveLoss
from losses.adv_loss import VAT2d
from memories.feature_memory import FeatureMemory
from scheduler.lr_scheduler import PolyLRScheduler
from scheduler.ramps import SigmoidRampUp

from utils import get_path, draw_mask, dummy_context

from models.unet import UNet

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

from activelearning import ActiveSelector, RandomSelector, EntropySelector


class ALConfig(object):
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
        in_channels: int = 1,
        num_classes: int = 3,
        block_type: Literal["plain"] = "plain",
        block_normalization: Literal["batch", "instance"] = "batch",
        dropout_prob: float = 0.1,
        deep_supervision: bool = False,
        ds_layer: int = 3,
        patch_size: int | tuple[int, int] | None = None,
        image_size: int | tuple[int, int] | None = None,
        model_ckpt: Path | str | None = None,
        # Data parameters
        dataset: Literal["ACDC"] = "ACDC",
        data_path: Path | str = "data",
        do_augment: bool = False,
        do_normalize: bool = False,
        batch_size: int = 32,
        num_workers: int = 1,
        pin_memory: bool = True,
        # Training parameters
        num_rounds: int = 5,
        budget: int = 10,
        active_selector_name: Literal["random", "entropy"] = "random",
        optimizer_name: Literal["adam", "adamw", "sgd"] = "adamw",
        optimizer_kwargs: dict = {},
        grad_norm: float = 10.0,
        num_iters: int = 4000,
        start_lr: float = 1e-3,
        lr_scheduler_name: Literal["poly", "none"] = "poly",
        lr_warmup_iter: int = 5000,
        save_freq_epoch: int = 100,
        valid_freq_iter: int = 200,
        valid_mode: Literal["volumn", "slice"] = "volumn",
        save_metric_name: Literal["dice", "hd", "loss"] = "dice",
        maximum_save_metric: bool | None = None,
        loss_name: Literal["dice+ce"] = "dice+ce",
        dice_weight: float = 0.8,
        early_stop_max_patience: int | None = None,
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
        self.block_type = block_type
        self.block_normalization = block_normalization
        self.dropout_prob = dropout_prob
        self.deep_supervision = deep_supervision
        self.ds_layer = ds_layer

        if patch_size is not None and isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            self.patch_size = patch_size

        if image_size is not None and isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size

        self.model_ckpt = model_ckpt
        # <<< Model parameters

        # >>> Data parameters
        self.dataset = dataset
        self.data_path = data_path
        self.do_augment = do_augment
        self.do_normalize = do_normalize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        # <<< Data parameters

        # >>> Training parameters
        self.num_rounds = num_rounds
        self.budget = budget
        self.active_selector_name = active_selector_name
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs
        self.grad_norm = grad_norm
        self.num_iters = num_iters
        self.start_lr = start_lr
        self.lr_scheduler_name = lr_scheduler_name
        self.lr_warmup_iter = lr_warmup_iter
        self.save_freq_epoch = save_freq_epoch
        self.valid_freq_iter = valid_freq_iter
        self.valid_mode = valid_mode
        self.save_metric_name = save_metric_name
        self.maximum_save_metric = maximum_save_metric
        self.early_stop_max_patience = early_stop_max_patience
        self.loss_name = loss_name
        self.dice_weight = dice_weight
        # <<< Training parameters

        # >>> Inference parameters
        self.stride = stride
        # <<< Inference parameters

        self.exp_name = exp_name

    def __setattr__(self, name: str, value: Any, /) -> None:
        if hasattr(self, "_config_dict"):
            self._config_dict[name] = value

        super(ALConfig, self).__setattr__(name, value)

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


class ALTrainer(BaseTrainer):
    def __init__(
        self,
        work_path: Path | str = Path.cwd(),
        device: torch.device | str = torch.device("cuda"),
        config: ALConfig | dict | str | Path | None = None,
        resume: str | Path | None = None,
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
        if isinstance(config, ALConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = ALConfig(**config)
        elif isinstance(config, (str, Path)):
            self.config = ALConfig().load(config)
        else:
            self.config = ALConfig()

        self.work_path = get_path(work_path)
        self.device = torch.device("cpu")
        self.to(device)

        self._set_seed(self.config.seed)

        self.resume = resume

        self.current_epoch = 0
        self.current_round = 0

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

        self.model.to(self.device)

    def _set_snapshot_work_dir(self):
        current_time_str = datetime.now().strftime("%Y%m%d_%H")
        snapshot_list = [
            f"ACDC",
            f"{current_time_str}",
            f"round-{self.config.num_rounds}",
            f"budget-{self.config.budget}",
            f"selector-{self.config.active_selector_name}",
            f"patchsz-{self.config.patch_size}",
            f"imgsz-{self.config.image_size}",
            f"batchsz-{self.config.batch_size}",
            f"epoch-{self.config.num_iters}",
            f"optimizer-{self.config.optimizer_name}",
            f"lr-{self.config.lr_scheduler_name}",
            f"lrwarm-{self.config.lr_warmup_iter}",
            f"startlr-{self.config.start_lr}",
            f"dice-{self.config.dice_weight}",
        ]
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
                project="active-learning",
                name=self.work_path.stem,
                config=self.config._config_dict,
            )

            for i in range(self.config.num_rounds):
                wandb.define_metric(f"round_{i}_train_epoch")
                wandb.define_metric(
                    f"round_{i}/train/epoch/*",
                    step_metric=f"round_{i}_train_epoch",
                )

                wandb.define_metric(f"round_{i}_train_iter")
                wandb.define_metric(
                    f"round_{i}/train/iter/*",
                    step_metric=f"round_{i}_train_iter",
                )

                wandb.define_metric(f"round_{i}_valid_step")
                wandb.define_metric(
                    f"round_{i}/valid/*", step_metric=f"round_{i}_valid_step"
                )

            wandb.define_metric("round_step")
            wandb.define_metric("test/*", step_metric="round_step")

    def _set_seed(self, seed: int):
        os.environ["AL_SEED"] = str(seed)

        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def _setup_logger(self):
        self.logger = logging.getLogger("MIA.ALTrainer")
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
            self.wandb_runner.log_artifact(
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
        self.model = UNet(
            dimension=2,
            input_channels=self.config.in_channels,
            output_classes=self.config.num_classes + 1,
            channels_list=[32, 64, 128, 256, 512],
            deep_supervision=self.config.deep_supervision,
            ds_layer=self.config.ds_layer,
            block_type=self.config.block_type,
            dropout_prob=self.config.dropout_prob,
            normalization=self.config.block_normalization,
        )

        if self.config.model_ckpt:
            self.load_model_checkpoint(self.config.model_ckpt)

    def load_model_checkpoint(self, ckpt: str | Path):
        if self.model is None:
            raise RuntimeError("Model is not built before loading checkpoint")

        try:
            state_dict = torch.load(ckpt, map_location=self.device)
            self.model.load_state_dict(state_dict)

            self.logger.info(f"Loaded model checkpoint to {ckpt}")
        except Exception as e:
            self.logger.warn(f"Failed to load model checkpoint from {ckpt}")
            self.logger.exception(e)

    def save_model_checkpoint(self, ckpt: str | Path):
        if self.model is None:
            raise RuntimeError("Model is not built before saving checkpoint")

        try:
            torch.save(self.model.state_dict(), ckpt)

            self.logger.info(f"Saved model checkpoint to {ckpt}")
        except Exception as e:
            self.logger.warn(f"Failed to save model checkpoint to {ckpt}")
            self.logger.exception(e)

    def get_data(self):
        labeled_dataset = ACDCDataset(
            data_path=self.config.data_path,
            split="train",
            normalize=self._get_train_normalize(),
            transform=self._get_train_transform(),
            logger=self.logger,
            image_channels=self.config.in_channels,
            image_size=self.config.image_size,
        )
        pool_dataset = ACDCDataset(
            data_path=self.config.data_path,
            split="train",
            normalize=self._get_train_normalize(),
            transform=None,
            logger=self.logger,
            image_channels=self.config.in_channels,
            image_size=self.config.image_size,
        )
        ex_labeled_dataset = ExtendableDataset(labeled_dataset, [])
        ex_pool_dataset = ExtendableDataset(pool_dataset)

        active_dataset = ActiveDataset(ex_labeled_dataset, ex_pool_dataset)

        valid_dataset = ACDCDataset(
            data_path=self.config.data_path,
            split="valid",
            normalize=self._get_valid_normalize(),
            transform=self._get_valid_transform(),
            logger=self.logger,
            image_channels=self.config.in_channels,
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
            labeled_dataset,
            pool_dataset,
            valid_dataset,
            active_dataset,
            valid_dataloader,
        )

    def get_train_dataloader(self, active_dataset: ActiveDataset):
        train_dataloader = DataLoader(
            dataset=active_dataset.get_train_dataset(),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        return train_dataloader

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

        return ComposeTransform(transforms)

    def _get_train_normalize(self):
        if self.config.do_normalize:
            return ZScoreNormalize()
        else:
            return None

    def _get_valid_transform(self):
        transforms = []
        return ComposeTransform(transforms)

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
                **self.config.optimizer_kwargs,
            )
        elif self.config.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                parameters,
                betas=(0.9, 0.999),
                **self.config.optimizer_kwargs,
            )
        elif self.config.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                parameters,
                momentum=0.9,
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
                self.config.num_iters,
                self.config.lr_warmup_iter,
            )
        elif self.config.lr_scheduler_name == "none":
            lr_scheduler = None
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
        else:
            raise ValueError(f"Loss function {self.config.loss_name} not found")

    def _setup_active_selector(self):
        if self.config.active_selector_name == "random":
            self.active_selector = RandomSelector()
        elif self.config.active_selector_name == "entropy":
            self.active_selector = EntropySelector(
                self.config.batch_size,
                self.config.num_workers,
                self.config.pin_memory,
            )
        else:
            raise ValueError(
                f"ActiveSelector {self.config.active_selector_name} not found"
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
        self.logger.info(f"  in_channels: {self.config.in_channels}")
        self.logger.info(f"  num_classes: {self.config.num_classes}")
        self.logger.info(f"  block_type: {self.config.block_type}")
        self.logger.info(
            f"  block_normalization: {self.config.block_normalization}"
        )
        self.logger.info(f"  deep_supervision: {self.config.deep_supervision}")
        self.logger.info(f"  ds_layer: {self.config.ds_layer}")
        self.logger.info(f"  patch_size: {self.config.patch_size}")
        self.logger.info(f"  image_size: {self.config.image_size}")
        self.logger.info(f"  pretrained_model: {self.config.model_ckpt}")
        self.logger.info(f"  model_ckpt: {self.config.model_ckpt}")

        self.logger.info(f"data: {self.config.dataset}")
        self.logger.info(f"  data_path: {self.config.data_path}")
        self.logger.info(
            f"  active_dataset_size (slices): {self.active_dataset.get_size()}"
        )
        self.logger.info(f"  valid_size (volumns): {len(self.valid_dataset)}")
        self.logger.info(f"  do_augment: {self.config.do_augment}")
        if self.config.do_augment:
            self.logger.info(
                f"{json.dumps(self._get_train_transform().get_params_dict(), indent=1)}"
            )
        self.logger.info(f"  normalize: {self.config.do_normalize}")
        self.logger.info(f"  batch_size: {self.config.batch_size}")
        self.logger.info(f"  num_workers: {self.config.num_workers}")
        self.logger.info(f"  pin_memory: {self.config.pin_memory}")

        self.logger.info(f"num_rounds: {self.config.num_rounds}")
        self.logger.info(f"budget: {self.config.budget}")
        self.logger.info(f"active_selector: {self.config.active_selector_name}")
        self.logger.info(f"valid_mode: {self.config.valid_mode}")
        self.logger.info(f"optimizer: {self.config.optimizer_name}")
        self.logger.info(f"  lr_warmup_iter: {self.config.lr_warmup_iter}")
        self.logger.info(f"  lr_scheduler: {self.config.lr_scheduler_name}")
        self.logger.info(f"  start_lr: {self.config.start_lr}")
        self.logger.info(f"  optimizer_kwargs: {self.config.optimizer_kwargs}")
        self.logger.info(f"loss_fn: {self.config.loss_name}")
        self.logger.info(f"save_metric: {self.config.save_metric_name}")
        self.logger.info(
            f"early_stop_max_patience: {self.config.early_stop_max_patience}"
        )
        self.logger.info(f"start_epoch: {self.current_epoch}")
        self.logger.info(f"num_iters: {self.config.num_iters}")
        self.logger.info(f"save_freq_epoch: {self.config.save_freq_epoch}")
        self.logger.info(f"valid_freq_iter: {self.config.valid_freq_iter}")
        self.logger.info(f"dice_weight: {self.config.dice_weight}")
        self._remove_config_file()

        if self.use_wandb and self.config_path:
            self.wandb_runner.log_artifact(
                self.config_path,
                name=f"config_{self.wandb_runner.id}",
                type="config",
                aliases=["txt"],
            )

    def on_train_start(self):
        assert self.model is not None

        (
            self.labeled_dataset,
            self.pool_dataset,
            self.valid_dataset,
            self.active_dataset,
            self.valid_dataloader,
        ) = self.get_data()

        self._setup_loss()
        self._setup_active_selector()

        self.current_round = 0

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

        if self.resume is not None:
            self.load_state_dict(self.resume)

        self._print_train_info()
        self._check_data_sanity()

    def _check_data_sanity(self):
        sanity_path = self.work_path / "sanity"
        sanity_path.mkdir(parents=True, exist_ok=True)

        for i in range(50):
            sample = self.labeled_dataset[i % 2]
            image = sample["image"]
            label = sample["label"]
            image_pil = F.to_pil_image(image).convert("RGB")
            mask_overlay = draw_mask(image_pil, label)
            mask_overlay_pil = Image.fromarray(mask_overlay)
            mask_overlay_pil.save(str(sanity_path / f"{i + 1}.png"))

    def on_train_end(self):
        pass

    def on_round_start(self):
        assert self.model is not None

        if self.config.model_ckpt:
            self.load_model_checkpoint(self.config.model_ckpt)

        self.model.train()
        self.model.to(self.device)

        data_list_path = (
            self.work_path / f"round_{self.current_round}/data_list.json"
        )

        new_samples = self.active_selector.select_next_batch(
            self.active_dataset, self.config.budget, self.model, self.device
        )
        self.active_dataset.extend_train_set(new_samples)
        self.active_dataset.save_data_list(data_list_path)
        if self.use_wandb:
            self.wandb_runner.log_artifact(
                data_list_path,
                name=f"data_list_{self.wandb_runner.id}_round_{self.current_round}",
                type="data_list",
            )

        self.train_dataloader = self.get_train_dataloader(self.active_dataset)

        self.current_epoch = 0
        self.current_iter = 0
        self.current_patience = 0

        self.optimizer, self.lr_scheduler = self._get_optimizer(
            self.model,
        )

        default_metric = torch.tensor(
            -torch.inf if self.config.maximum_save_metric else torch.inf
        )
        self._best_valid_metric = default_metric
        self._cur_valid_metric = default_metric

        labeled_size, pool_size = self.active_dataset.get_size()
        self.logger.info("")
        self.logger.info(f"Round {self.current_round}:")
        self.logger.info(f"Labeled size: {labeled_size}")
        self.logger.info(f"Pool size: {pool_size}")

    def on_round_end(self):
        ckpt_path = self.work_path / f"round_{self.current_round}/final_model"
        self.save_state_dict(ckpt_path, True)
        if self.use_wandb:
            self.wandb_runner.log_model(
                ckpt_path,
                name=f"model_{self.wandb_runner.id}_round_{self.current_round}",
                aliases=[f"epoch_{self.current_epoch}", "final"],
            )

        self.load_model_checkpoint(
            self.work_path / f"round_{self.current_round}/best_model/model.pth"
        )
        self.perform_real_test()

        self.logger.info("")
        self.logger.info("")
        self.current_round += 1

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
            ckpt_path = (
                self.work_path
                / f"round_{self.current_round}/epoch_{self.current_epoch}"
            )
            self.save_state_dict(ckpt_path, True)

            if self.use_wandb:
                self.wandb_runner.log_model(
                    ckpt_path,
                    name=f"model_{self.wandb_runner.id}_round_{self.current_round}",
                    aliases=[f"epoch_{self.current_epoch}"],
                )

        train_loss = (
            torch.stack([o["loss"] for o in self.epoch_train_outputs])
            .mean(0)
            .item()
        )
        self.logger.info(f"Loss ({self.config.loss_name}): {train_loss}")

        self._train_end_time = time.time()
        time_elapsed = self._train_end_time - self._train_start_time
        self.logger.info(f"Train time elapsed: {time_elapsed:.3f} seconds")

        if self.use_wandb:
            train_metric = {
                f"round_{self.current_round}/train/epoch/loss": train_loss,
                f"round_{self.current_round}_train_epoch": self.current_epoch,
                f"round_{self.current_round}_train_iter": self.current_iter,
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

        asd_per_class = metric[:, 2]
        avg_asd = asd_per_class.mean()

        jc_per_class = metric[:, 3]
        avg_jc = jc_per_class.nanmean()

        self.logger.info(f"DSC per class: {dsc_per_class.tolist()}")
        self.logger.info(f"HD per class: {hd_per_class.tolist()}")
        self.logger.info(f"ASD per class: {asd_per_class.tolist()}")
        self.logger.info(f"JC per class: {jc_per_class.tolist()}")
        self.logger.info(f"avg DSC: {avg_dsc.item()}")
        self.logger.info(f"avg HD: {avg_hd.item()}")
        self.logger.info(f"avg ASD: {avg_asd.item()}")
        self.logger.info(f"avg JC: {avg_jc.item()}")
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
                f"round_{self.current_round}/valid/metric/dsc": avg_dsc.item(),
                f"round_{self.current_round}/valid/metric/hd95": avg_hd.item(),
                f"round_{self.current_round}/valid/metric/asd": avg_asd.item(),
                f"round_{self.current_round}/valid/metric/jc": avg_jc.item(),
                f"round_{self.current_round}/valid/metric/loss": loss.item(),
                f"round_{self.current_round}_train_epoch": self.current_epoch,
                f"round_{self.current_round}_train_iter": self.current_iter,
                f"round_{self.current_round}_valid_step": self.current_iter,
            }
            for i in range(self.config.num_classes):
                valid_metric[
                    f"round_{self.current_round}/valid/metric_per_cls/dsc/class_{i}"
                ] = dsc_per_class[i].item()
                valid_metric[
                    f"round_{self.current_round}/valid/metric_per_cls/hd95/class_{i}"
                ] = hd_per_class[i].item()
                valid_metric[
                    f"round_{self.current_round}/valid/metric_per_cls/asd/class_{i}"
                ] = asd_per_class[i].item()
                valid_metric[
                    f"round_{self.current_round}/valid/metric_per_cls/jc/class_{i}"
                ] = jc_per_class[i].item()

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
            self.save_state_dict(
                self.work_path / f"round_{self.current_round}/best_model"
            )

            ckpt_path = (
                self.work_path
                / f"round_{self.current_round}/iter_{self.current_iter}_{self._best_valid_metric:.4f}"
            )
            self.save_state_dict(ckpt_path)

            if self.use_wandb:
                self.wandb_runner.log_model(
                    ckpt_path,
                    name=f"best_model_{self.wandb_runner.id}_round_{self.current_round}",
                    aliases=[
                        f"iter_{self.current_iter}",
                        f"{self.config.save_metric_name}_{self._best_valid_metric:.4f}",
                    ],
                )
            is_improved = True

        if is_improved:
            self.current_patience = 0
            if self.use_wandb:
                text_lines = [
                    f"round={self.current_round}",
                    f"iter={self.current_iter}",
                    f"epoch={self.current_epoch}",
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
            if self.use_wandb and self.config.early_stop_max_patience:
                alert_threshold = self.config.early_stop_max_patience * 0.5
                if self.current_patience >= alert_threshold:
                    self.wandb_runner.alert(
                        title="Performance Stagnation",
                        text=f"Round {self.current_round}: performance is not improved for {self.current_patience} step",
                        level="WARN",
                    )

        self._valid_end_time = time.time()
        time_elapsed = self._valid_end_time - self._valid_start_time
        self.logger.info(f"current_patience: {self.current_patience}")
        self.logger.info(f"Valid time elapsed: {time_elapsed:.3f} seconds")

    def train_step(self, sampled_batch):
        self.model.train()

        _train_step_start_time = time.time()

        self.logger.info(f"Iteration {self.current_iter}:")

        if self.lr_scheduler:
            self.lr_scheduler.step(self.current_iter)
        self.logger.info(f"lr: {self.optimizer.param_groups[0]['lr']}")

        image_batch, label_batch = (
            sampled_batch["image"],
            sampled_batch["label"],
        )  #  [b, c, h, w], [b, h, w]

        image_batch, label_batch = image_batch.to(self.device), label_batch.to(
            self.device
        )


        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.model(image_batch)
            loss, _, _ = self.supervised_loss(output, label_batch)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_norm
        )
        self.optimizer.step()

        self.logger.info(f"Loss: {loss.item()}")
        self.epoch_train_outputs.append({"loss": loss})

        if self.use_wandb:
            lr = self.optimizer.param_groups[0]["lr"]
            train_metric = {
                f"round_{self.current_round}/train/iter/lr": lr,
                f"round_{self.current_round}/train/iter/loss": loss,
                f"round_{self.current_round}_train_epoch": self.current_epoch,
                f"round_{self.current_round}_train_iter": self.current_iter,
            }
            self.wandb_runner.log(train_metric)

        _train_step_end_time = time.time()
        time_elapsed = _train_step_end_time - _train_step_start_time
        self.logger.info(f"Iteration time elapsed: {time_elapsed:.3f} seconds")

        self.logger.info(f"")
        self.current_iter += 1

    def valid_step(self, sampled_batch):
        if self.config.valid_mode == "volumn":
            metric, loss = self.valid_volumns(sampled_batch)
        else:
            metric, loss = self.valid_slices(sampled_batch)

        self.epoch_valid_outputs.append(
            {
                "metric": torch.Tensor(metric),
                "loss": loss,
            }
        )

    def valid_slices(self, sampled_batch, spacing=None):
        spacing = (
            sampled_batch["spacing"] if "spacing" in sampled_batch else None
        )

        image_batch, label_batch = (
            sampled_batch["image"],
            sampled_batch["label"],
        )  #  [b, c, h, w], [b, h, w]

        B, C, H, W = image_batch.shape

        image_batch, label_batch = image_batch.to(self.device), label_batch.to(
            self.device
        )

        if self.config.image_size is not None:
            image_batch = F.resize(
                image_batch,
                list(self.config.image_size),
                interpolation=F.InterpolationMode.BILINEAR,
            )

        output = self.model(image_batch)
        prob = output.softmax(1)
        pred = prob.argmax(1)

        if pred.shape[-2:] != label_batch.shape[-2:]:
            pred = F.resize(
                pred,
                size=label_batch.shape[-2:],
                interpolation=F.InterpolationMode.NEAREST,
            )
            loss_label_batch = F.resize(
                label_batch,
                size=output.shape[-2:],
                interpolation=F.InterpolationMode.NEAREST,
            )
        else:
            loss_label_batch = label_batch

        if hasattr(self, "supervised_loss"):
            loss, _, _ = self.supervised_loss(output, loss_label_batch)
        else:
            loss = None

        pred = pred.cpu().numpy()
        label_batch = label_batch.cpu().numpy()

        metric = np.zeros((B, self.config.num_classes, 4))

        for b in range(B):
            for c in range(1, self.config.num_classes + 1):
                metric[b, c - 1] = self.calculate_metric_percase(
                    pred[b] == c, label_batch[b] == c, spacing
                )

        return metric, loss

    def valid_volumns(self, sampled_batch):
        spacing = (
            sampled_batch["spacing"] if "spacing" in sampled_batch else None
        )

        image_batch, label_batch = (
            sampled_batch["image"],
            sampled_batch["label"],
        )  #  [1, c, d, h, w], [1, d, h, w]

        image_batch, label_batch = image_batch.to(self.device), label_batch.to(
            self.device
        )
        image_batch = image_batch.squeeze(0).permute(1, 0, 2, 3)  # D, C, H, W
        label_batch = label_batch.squeeze(0)  # D, H, W

        if self.config.image_size is not None:
            image_batch = F.resize(
                image_batch,
                list(self.config.image_size),
                interpolation=F.InterpolationMode.BILINEAR,
            )

        output = self.model(image_batch)
        prob = output.softmax(1)
        pred = prob.argmax(1)

        if pred.shape[-2:] != label_batch.shape[-2:]:
            pred = F.resize(
                pred,
                size=label_batch.shape[-2:],
                interpolation=F.InterpolationMode.NEAREST,
            )
            loss_label_batch = F.resize(
                label_batch,
                size=output.shape[-2:],
                interpolation=F.InterpolationMode.NEAREST,
            )
        else:
            loss_label_batch = label_batch

        if hasattr(self, "supervised_loss"):
            loss, _, _ = self.supervised_loss(output, loss_label_batch)
        else:
            loss = None

        pred = pred.cpu().numpy()
        label_batch = label_batch.cpu().numpy()

        metric = np.zeros((self.config.num_classes, 4))

        for c in range(1, self.config.num_classes + 1):
            metric[c - 1] = self.calculate_metric_percase(
                pred == c, label_batch == c, spacing
            )

        return metric, loss

    def calculate_metric_percase(
        self, pred: np.ndarray, gt: np.ndarray, spacing=None
    ):
        pred[pred > 0] = 1
        gt[gt > 0] = 1

        dice = 0
        hd95 = np.nan
        asd = np.nan
        jc = 0

        if pred.sum() > 0:
            dice = metric.dc(pred, gt)
            hd95 = metric.hd95(pred, gt, spacing)
            asd = metric.asd(pred, gt, spacing)
            jc = metric.jc(pred, gt)

        return dice, hd95, asd, jc

    def train(self):
        self.on_train_start()

        for round in range(self.config.num_rounds):
            self.on_round_start()
            while not self.is_finished():
                self.on_epoch_start()
                self.on_train_epoch_start()
                for sampled_batch in self.train_tqdm:
                    if self.is_finished():
                        break
                    self.train_step(sampled_batch)
                    self.valid()
                self.on_train_epoch_end()
                self.on_epoch_end()
            self.on_round_end()

        self.on_train_end()

    def valid(self):
        if (self.current_iter) % self.config.valid_freq_iter == 0:
            with torch.no_grad():
                self.on_valid_epoch_start()
                for sampled_batch in self.valid_tqdm:
                    self.valid_step(sampled_batch)
                self.on_valid_epoch_end()

    def is_finished(self):
        if self.config.early_stop_max_patience:
            is_finished = (
                self.current_patience >= self.config.early_stop_max_patience
            )
            if is_finished:
                self.logger.info(
                    "Exceeded maximum patience. Training will be early stopped"
                )
            return is_finished

        return self.current_iter >= self.config.num_iters

    def run_training(self):
        self.train()

    def perform_real_test(self):
        test_dataset = ACDCDataset(
            data_path=self.config.data_path,
            split="test",
            normalize=self._get_valid_normalize(),
            transform=self._get_valid_transform(),
            logger=self.logger,
            image_channels=self.config.in_channels,
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=1,
            pin_memory=True,
        )

        metric_list = []

        for sampled_batch in tqdm(test_dataloader):
            if self.config.valid_mode == "volumn":
                metric, _ = self.valid_volumns(sampled_batch)
            else:
                metric, _ = self.valid_slices(sampled_batch)

            metric_list.append(metric)

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
            self.wandb_runner.log(
                {f"test_performance_round_{self.current_round}": wandb_table}
            )

        avg_metric_per_cls = metric_tensor.mean(0)
        self.logger.info("Real test results:")
        for id in classes.keys():
            if id == 0:
                continue

            self.logger.info(
                f"  {classes[id]}: {avg_metric_per_cls[id-1].tolist()}"
            )

        self.logger.info(f"Average: {avg_metric_per_cls.mean(0).tolist()}")

        if self.use_wandb:
            avg_metric = avg_metric_per_cls.mean(0)
            avg_dsc = avg_metric[0]
            avg_hd = avg_metric[1]
            avg_asd = avg_metric[2]
            avg_jc = avg_metric[3]
            test_metric = {
                f"test/metric/dsc": avg_dsc.item(),
                f"test/metric/hd95": avg_hd.item(),
                f"test/valid/metric/asd": avg_asd.item(),
                f"test/valid/metric/jc": avg_jc.item(),
                f"round_step": self.current_round,
            }
            self.wandb_runner.log(test_metric)
        # save:
        write_csv = self.work_path / f"test_mean_round_{self.current_round}.csv"
        save = pd.DataFrame(dataframe_dict)
        save.to_csv(write_csv, index=False, sep=",")

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer,
            "current_iter": self.current_iter,
            "current_epoch": self.current_epoch,
            "current_round": self.current_round,
            "data_list": self.active_dataset.data_list(),
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
            # Offset by 1 because we save the training state at the end
            self.current_epoch = training_state["current_epoch"] + 1
            self.current_iter = training_state["current_iter"] + 1
            self.current_round = training_state["current_round"] + 1
            self.active_dataset.load_data_list(training_state["data_list"])

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
