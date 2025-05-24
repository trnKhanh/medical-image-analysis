import random
from copy import deepcopy
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

from medpy import metric as medpy_metric

import numpy as np
import pandas as pd

from PIL import Image

from rich.logging import RichHandler
from rich.console import Console

from tqdm import tqdm

from .base_trainer import BaseTrainer
from datasets import (
    ACDCDataset,
    ActiveDataset,
    ExtendableDataset,
    TN3KDataset,
    TG3KDataset,
    FUGCDataset,
)
from losses.compound_losses import DiceAndCELoss
from losses.dice_loss import DiceLoss
from scheduler.lr_scheduler import PolyLRScheduler

from utils import get_path, draw_mask, dummy_context

from models.unet import UNet, UnetProcessor
from models._unet import _UNet

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

from activelearning import (
    ActiveSelector,
    RandomSelector,
    EntropySelector,
    ConfidenceSelector,
    MarginSelector,
    CoresetSelector,
    KMeanSelector,
    BADGESelector,
)

from metric import cal_hd


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
        init_round_path: str | Path | None = None,
        init_data_list: str | Path | None = None,
        # Model parameters
        in_channels: int = 1,
        num_classes: int = 3,
        postprocess_mask: bool = False,
        block_type: Literal["plain"] = "plain",
        block_normalization: Literal["batch", "instance"] = "batch",
        dropout_prob: float = 0.1,
        deep_supervision: bool = False,
        ds_layer: int = 3,
        patch_size: int | tuple[int, int] | None = None,
        image_size: int | tuple[int, int] | None = None,
        model_ckpt: Path | str | None = None,
        # Data parameters
        dataset: Literal["ACDC", "tn3k", "tg3k", "fugc"] = "ACDC",
        data_path: Path | str = "data",
        do_oversample: bool = False,
        do_augment: bool = False,
        do_normalize: bool = False,
        batch_size: int = 32,
        valid_batch_size: int = 1,
        num_workers: int = 1,
        pin_memory: bool = True,
        # Training parameters
        active_learning: bool = True,
        num_rounds: int = 5,
        budget: int = 10,
        persist_model_weight: bool = False,
        active_selector_name: Literal[
            "random",
            "entropy",
            "confidence",
            "margin",
            "coreset-l2",
            "coreset-cosine",
            "kmean-l2",
            "kmean-cosine",
            "badge",
        ] = "random",
        coreset_criteria: Literal["min", "mean"] = "min",
        kmean_sharp_factor: float = 1.0,
        kmean_softmax: bool = False,
        feature_path: Path | str | None = None,
        loaded_feature_weight: float = 0.0,
        loaded_feature_only: bool = False,
        optimizer_name: Literal["adam", "adamw", "sgd"] = "adamw",
        optimizer_kwargs: dict = {},
        grad_norm: float = 10.0,
        num_iters: int = 4000,
        start_lr: float = 1e-3,
        lr_scheduler_name: Literal["poly", "none"] = "poly",
        lr_interval: int = 1,
        lr_warmup_iter: int = 5000,
        save_freq_epoch: int | None = None,
        valid_freq_iter: int = 200,
        valid_mode: Literal["volumn", "slice"] = "volumn",
        save_metric_name: Literal["dice", "hd", "loss"] = "dice",
        maximum_save_metric: bool | None = None,
        loss_name: Literal["dice+ce"] = "dice+ce",
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        early_stop_max_patience: int | None = None,
        # Inference parameters
        stride: int | tuple[int, ...] | list[int] | None = None,
        # Misc parameters
        exp_name: str = "",
        **kwargs,
    ):
        self._config_dict = {}

        self.seed = seed
        self.init_round_path = init_round_path
        self.init_data_list = init_data_list

        # >>> Model parameters
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.postprocess_mask = postprocess_mask
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
        self.do_oversample = do_oversample
        self.do_augment = do_augment
        self.do_normalize = do_normalize
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        # <<< Data parameters

        # >>> Training parameters
        self.active_learning = active_learning
        if self.active_learning:
            self.num_rounds = num_rounds
            self.budget = budget
        else:
            self.num_rounds = 1
            self.budget = -1

        self.persist_model_weight = persist_model_weight

        self.active_selector_name = active_selector_name
        self.coreset_criteria = coreset_criteria
        self.kmean_sharp_factor = kmean_sharp_factor
        self.kmean_softmax = kmean_softmax
        self.feature_path = feature_path
        self.loaded_feature_weight = loaded_feature_weight
        self.loaded_feature_only = loaded_feature_only
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs
        self.grad_norm = grad_norm
        self.num_iters = num_iters
        self.start_lr = start_lr
        self.lr_scheduler_name = lr_scheduler_name
        self.lr_interval = lr_interval
        self.lr_warmup_iter = lr_warmup_iter
        self.save_freq_epoch = save_freq_epoch
        self.valid_freq_iter = valid_freq_iter
        self.valid_mode = valid_mode
        self.save_metric_name = save_metric_name
        self.maximum_save_metric = maximum_save_metric
        self.early_stop_max_patience = early_stop_max_patience
        self.loss_name = loss_name
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
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
    seed = int(os.environ["AL_SEED"] or 0)
    seed = seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class ALTrainer(BaseTrainer):
    def __init__(
        self,
        work_path: Path | str = Path.cwd(),
        deterministic: bool = True,
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

        self.deterministic = deterministic

        if self.deterministic:
            print("Change cudnn backend to deterministic")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

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
            f"{self.config.dataset}",
            f"{current_time_str}",
            f"al-{self.config.active_learning}",
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
            f"ce-{self.config.ce_weight}",
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
        if self.verbose:
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
        self.model_processor = UnetProcessor(image_size=self.config.image_size)

        if self.config.model_ckpt:
            self.load_model_checkpoint(self.config.model_ckpt)

    def load_model_checkpoint(self, ckpt: str | Path):
        if self.model is None:
            raise RuntimeError("Model is not built before loading checkpoint")

        try:
            state_dict = torch.load(ckpt, map_location=self.device)
            if "model" in state_dict:
                self.model.load_state_dict(state_dict["model"])
            else:
                self.model.load_state_dict(state_dict)

            self.logger.info(f"Loaded model checkpoint from {ckpt}")
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

    def get_dataset(
        self,
        split: Literal["train", "valid", "test"],
        include_transform: bool = False,
    ):
        if split == "train":
            _transform = self._get_train_transform()
            _normalize = self._get_train_normalize()
            _image_size = self.config.image_size
        else:
            _transform = self._get_valid_transform()
            _normalize = self._get_valid_normalize()
            _image_size = None

        if not include_transform:
            _transform = None

        if self.config.dataset == "ACDC":
            dataset = ACDCDataset(
                data_path=self.config.data_path,
                split=split,
                normalize=_normalize,
                transform=_transform,
                logger=self.logger,
                image_channels=self.config.in_channels,
                image_size=_image_size,
            )
        elif self.config.dataset == "tn3k":
            dataset = TN3KDataset(
                data_path=self.config.data_path,
                split=split,
                normalize=_normalize,
                transform=_transform,
                logger=self.logger,
                image_channels=self.config.in_channels,
                image_size=_image_size,
            )
        elif self.config.dataset == "tg3k":
            dataset = TG3KDataset(
                data_path=self.config.data_path,
                split=split,
                normalize=_normalize,
                transform=_transform,
                logger=self.logger,
                image_channels=self.config.in_channels,
                image_size=_image_size,
            )
        elif self.config.dataset == "fugc":
            dataset = FUGCDataset(
                data_path=self.config.data_path,
                split=split,
                normalize=_normalize,
                transform=_transform,
                logger=self.logger,
                image_channels=self.config.in_channels,
                image_size=_image_size,
            )
        else:
            raise ValueError(f"{self.config.dataset} dataset is undefined")

        return dataset

    def get_data(self):
        labeled_dataset = self.get_dataset("train", include_transform=True)
        pool_dataset = self.get_dataset("train", include_transform=False)
        valid_dataset = self.get_dataset("valid", include_transform=True)

        ex_labeled_dataset = ExtendableDataset(labeled_dataset, [])
        ex_pool_dataset = ExtendableDataset(pool_dataset)

        active_dataset = ActiveDataset(ex_labeled_dataset, ex_pool_dataset)

        valid_dataloader = DataLoader(
            dataset=valid_dataset,
            batch_size=self.config.valid_batch_size,
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
        train_dataset = active_dataset.get_train_dataset()
        oversampled_dataset = deepcopy(train_dataset)

        if self.config.do_oversample:
            # If dataset is not enough for batch_size, we oversample it
            # For some reasons, this implementation is extremely faster compared
            # to just oversample to batch_size
            total_seen_samples = self.config.num_iters * self.config.batch_size
            num_extended = int(np.ceil(total_seen_samples / len(train_dataset)))
            oversampled_dataset.image_idx = (
                oversampled_dataset.image_idx * num_extended
            )

        train_dataloader = DataLoader(
            dataset=oversampled_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            worker_init_fn=_worker_init_fn,
        )
        return train_dataloader

    def _get_train_transform(self):
        transforms = []
        if self.config.do_augment:
            if self.config.dataset == "fugc":
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
            else:
                transforms.append(
                    RandomTransform(
                        ComposeTransform(
                            [
                                RandomRotation90(),
                                RandomChoiceTransform(
                                    [
                                        MirrorTransform((-2)),
                                        MirrorTransform((-1)),
                                    ]
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

    def _setup_optimizer(
        self,
    ):
        assert self.model is not None

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())

        if self.config.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                parameters,
                betas=(0.9, 0.999),
                **self.config.optimizer_kwargs,
            )
        elif self.config.optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                parameters,
                betas=(0.9, 0.999),
                **self.config.optimizer_kwargs,
            )
        elif self.config.optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(
                parameters,
                momentum=0.9,
                **self.config.optimizer_kwargs,
            )
        else:
            raise ValueError(
                f'Optimizer "{self.config.optimizer_name}" not supported'
            )

        if self.config.lr_scheduler_name == "poly":
            self.lr_scheduler = PolyLRScheduler(
                self.optimizer,
                initial_lr=self.config.start_lr,
                max_steps=self.config.num_iters,
                warmup_steps=self.config.lr_warmup_iter,
                interval=self.config.lr_interval,
            )
        elif self.config.lr_scheduler_name == "none":
            self.lr_scheduler = None
        else:
            raise ValueError(
                f'Learning rate scheduler "{self.config.lr_scheduler_name}" not supported'
            )

    def _setup_loss(self):
        if self.config.loss_name == "dice+ce":
            self.supervised_loss = DiceAndCELoss(
                dice_loss=DiceLoss,
                dice_kwargs={
                    "num_classes": self.config.num_classes,
                    "smooth": 1e-5,
                    "do_bg": True,
                    "softmax": True,
                    "batch": False,
                    "squared": False,
                },
                ce_loss=torch.nn.CrossEntropyLoss,
                ce_kwargs={},
                default_dice_weight=self.config.dice_weight,
                default_ce_weight=self.config.ce_weight,
            )
        else:
            raise ValueError(f"Loss function {self.config.loss_name} not found")

    def _setup_active_selector(self):
        if self.config.active_selector_name == "random":
            self.active_selector = RandomSelector()
        elif self.config.active_selector_name == "entropy":
            self.active_selector = EntropySelector(
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
            )
        elif self.config.active_selector_name == "confidence":
            self.active_selector = ConfidenceSelector(
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
            )
        elif self.config.active_selector_name == "margin":
            self.active_selector = MarginSelector(
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
            )
        elif self.config.active_selector_name == "coreset-l2":
            self.active_selector = CoresetSelector(
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                metric="l2",
                feature_path=self.config.feature_path,
                loaded_feature_weight=self.config.loaded_feature_weight,
                coreset_criteria=self.config.coreset_criteria,
            )
        elif self.config.active_selector_name == "coreset-cosine":
            self.active_selector = CoresetSelector(
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                metric="cosine",
                feature_path=self.config.feature_path,
                loaded_feature_weight=self.config.loaded_feature_weight,
                coreset_criteria=self.config.coreset_criteria,
            )
        elif self.config.active_selector_name == "kmean-l2":
            self.active_selector = KMeanSelector(
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                metric="l2",
                feature_path=self.config.feature_path,
                loaded_feature_weight=self.config.loaded_feature_weight,
                coreset_criteria=self.config.coreset_criteria,
                sharp_factor=self.config.kmean_sharp_factor,
                softmax=self.config.kmean_softmax,
                loaded_feature_only=self.config.loaded_feature_only,
            )
        elif self.config.active_selector_name == "kmean-cosine":
            self.active_selector = KMeanSelector(
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                metric="cosine",
                feature_path=self.config.feature_path,
                loaded_feature_weight=self.config.loaded_feature_weight,
                coreset_criteria=self.config.coreset_criteria,
                sharp_factor=self.config.kmean_sharp_factor,
                softmax=self.config.kmean_softmax,
                loaded_feature_only=self.config.loaded_feature_only,
            )
        elif self.config.active_selector_name == "badge":
            self.active_selector = BADGESelector(
                dice_loss=self.supervised_loss.dice_loss,
                ce_loss=self.supervised_loss.ce_loss,
                batch_size=1,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                multiple_loss="add",
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
        self.logger.info(f"deterministic: {self.deterministic}")
        self.logger.info(f"device: {self.device}")
        self.logger.info(f"seed: {self.seed}")
        self.logger.info(f'log_file: "{self.log_path}"')
        if self.use_wandb:
            self.logger.info(f"wandb_run_id: {self.wandb_runner.id}")

        self.logger.info(f"model: {self.model}")
        self.logger.info(f"  in_channels: {self.config.in_channels}")
        self.logger.info(f"  num_classes: {self.config.num_classes}")
        self.logger.info(f"  postprocess_mask: {self.config.postprocess_mask}")
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
        self.logger.info(f"  do_oversample: {self.config.do_oversample}")
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
        self.logger.info(
            f"persist_model_weight: {self.config.persist_model_weight}"
        )
        self.logger.info(f"active_selector: {self.config.active_selector_name}")
        if self.config.active_selector_name.startswith(
            "coreset"
        ) or self.config.active_selector_name.startswith("kmean"):
            self.logger.info(
                f"coreset_criteria: {self.config.coreset_criteria}"
            )
            self.logger.info(
                f"kmean_sharp_factor: {self.config.kmean_sharp_factor}"
            )
            self.logger.info(f"kmean_softmax: {self.config.kmean_softmax}")
        self.logger.info(f"feature_path: {self.config.feature_path}")
        self.logger.info(
            f"loaded_feature_weight: {self.config.loaded_feature_weight}"
        )
        self.logger.info(f"valid_mode: {self.config.valid_mode}")
        self.logger.info(f"optimizer: {self.config.optimizer_name}")
        self.logger.info(f"  lr_warmup_iter: {self.config.lr_warmup_iter}")
        self.logger.info(f"  lr_scheduler: {self.config.lr_scheduler_name}")
        self.logger.info(f"  start_lr: {self.config.start_lr}")
        self.logger.info(f"  optimizer_kwargs: {self.config.optimizer_kwargs}")
        self.logger.info(f"  instance: {self.optimizer}")
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
        self.logger.info(f"ce_weight: {self.config.ce_weight}")
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

        self._setup_optimizer()
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

        if self.config.init_round_path:
            round_0_path = get_path(self.config.init_round_path)

            self.load_model_checkpoint(round_0_path / "best_model/model.pth")
            self.active_dataset.load_data_list(round_0_path / "data_list.json")

            self.perform_real_test()

            self.current_round = 1

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

        data_list_path = (
            self.work_path / f"round_{self.current_round}/data_list.json"
        )
        # Load model from last round to select label
        if self.current_round > 0:
            last_ckpt = (
                self.work_path
                / f"round_{self.current_round-1}/best_model/model.pth"
            )

            if self.current_round > 1 or self.config.init_round_path is None:
                self.load_model_checkpoint(last_ckpt)

        if self.config.active_learning:
            if self.current_round == 0 and self.config.init_data_list:
                self.active_dataset.load_data_list(self.config.init_data_list)
            else:
                new_samples = self.active_selector.select_next_batch(
                    self.active_dataset,
                    self.config.budget,
                    self.model,
                    self.device,
                )

                self.active_dataset.extend_train_set(new_samples)
        else:
            pool_samples = deepcopy(self.active_dataset.pool_dataset.image_idx)
            self.active_dataset.extend_train_set(pool_samples)

        # Loading model must be placed after selecting samples since we use the
        # model to choose samples
        if self.current_round > 0:
            self._build_model()
            if self.config.persist_model_weight and (
                self.current_round > 1 or self.config.init_round_path is None
            ):
                self.load_model_checkpoint(
                    self.work_path
                    / f"round_{self.current_round-1}/best_model/model.pth"
                )

        self.model.to(self.device)
        self.model.train()

        self.active_dataset.save_data_list(data_list_path)
        if self.use_wandb:
            self.wandb_runner.log_artifact(
                data_list_path,
                name=f"data_list_{self.wandb_runner.id}",
                aliases=[f"round_{self.current_round}"],
                type="data_list",
            )

        self.train_dataloader = self.get_train_dataloader(self.active_dataset)

        self.current_epoch = 0
        self.current_iter = 0
        self.current_patience = 0

        self._setup_optimizer()

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
                name=f"model_{self.wandb_runner.id}",
                aliases=[
                    f"round_{self.current_round}",
                ],
            )
            if self.use_wandb:
                self.wandb_runner.log_model(
                    ckpt_path,
                    name=f"best_model_{self.wandb_runner.id}",
                    aliases=[
                        f"{self.config.save_metric_name}_{self._best_valid_metric:.4f}",
                        f"round_{self.current_round}",
                    ],
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
        if (
            self.config.save_freq_epoch
            and (self.current_epoch + 1) % self.config.save_freq_epoch == 0
        ):
            ckpt_path = (
                self.work_path
                / f"round_{self.current_round}/epoch_{self.current_epoch}"
            )
            self.save_state_dict(ckpt_path, True)

            if self.use_wandb:
                self.wandb_runner.log_model(
                    ckpt_path,
                    name=f"model_{self.wandb_runner.id}",
                    aliases=[
                        f"epoch_{self.current_epoch}",
                        f"round_{self.current_round}",
                    ],
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
        avg_metric_all = torch.cat(
            [o["metric_all"] for o in self.epoch_valid_outputs], dim=0
        ).nanmean(0)
        avg_metric_per_cls = torch.cat(
            [o["metric"] for o in self.epoch_valid_outputs], dim=0
        ).nanmean(0)
        loss = torch.stack(
            [o["loss"] for o in self.epoch_valid_outputs]
        ).nanmean()

        dsc_per_class = avg_metric_per_cls[:, 0]
        avg_dsc = dsc_per_class.mean()

        hd_per_class = avg_metric_per_cls[:, 1]
        avg_hd = hd_per_class.nanmean()

        asd_per_class = avg_metric_per_cls[:, 2]
        avg_asd = asd_per_class.mean()

        jc_per_class = avg_metric_per_cls[:, 3]
        avg_jc = jc_per_class.nanmean()

        classes = self.valid_dataset.CLASSES

        self.logger.info("Valid results (DSC, HD, ASD, JSD):")
        for id in classes.keys():
            if id == 0:
                self.logger.info(f"  all: {avg_metric_all.tolist()}")
            else:
                self.logger.info(
                    f"  {classes[id]}: {avg_metric_per_cls[id-1].tolist()}"
                )

        self.logger.info(f"Average: {avg_metric_per_cls.nanmean(0).tolist()}")
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
                f"round_{self.current_round}/valid/metric/hd": avg_hd.item(),
                f"round_{self.current_round}/valid/metric/loss": loss.item(),
                f"round_{self.current_round}_train_epoch": self.current_epoch,
                f"round_{self.current_round}_train_iter": self.current_iter,
                f"round_{self.current_round}_valid_step": self.current_iter,
            }

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

            is_improved = True

        if is_improved:
            self.current_patience = 0
            # if self.use_wandb:
            #     text_lines = [
            #         f"round={self.current_round}",
            #         f"iter={self.current_iter}",
            #         f"epoch={self.current_epoch}",
            #         f"metric={self._best_valid_metric.item():.4f}",
            #         f"dsc="
            #         + "["
            #         + ", ".join([f"{x:.4f}" for x in dsc_per_class.tolist()])
            #         + "]",
            #         f"average_dsc={avg_dsc.item():.4f}",
            #         f"hd95="
            #         + "["
            #         + ", ".join([f"{x:.4f}" for x in hd_per_class.tolist()])
            #         + "]",
            #         f"average_hd95={avg_hd.item():.4f}",
            #         f"loss={loss.item():.4f}",
            #     ]
            #     self.wandb_runner.alert(
            #         title="Improved Performance",
            #         text="; ".join(text_lines),
            #         level="INFO",
            #     )
        else:
            self.current_patience += 1
            # if self.use_wandb and self.config.early_stop_max_patience:
            #     alert_threshold = self.config.early_stop_max_patience * 0.5
            #     if self.current_patience >= alert_threshold:
            #         self.wandb_runner.alert(
            #             title="Performance Stagnation",
            #             text=f"Round {self.current_round}: performance is not improved for {self.current_patience} step",
            #             level="WARN",
            #         )

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

        image_batch, label_batch = image_batch.to(
            self.device, dtype=torch.float32
        ), label_batch.to(self.device, dtype=torch.long)

        output = self.model(image_batch)

        loss = self.supervised_loss(output, label_batch)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.config.grad_norm
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
            metric_all, metric, loss = self.valid_volumns(sampled_batch)
        else:
            metric_all, metric, loss = self.valid_slices(sampled_batch)

        self.epoch_valid_outputs.append(
            {
                "metric_all": torch.Tensor(metric_all),
                "metric": torch.Tensor(metric),
                "loss": loss,
            }
        )

    def valid_slices(self, sampled_batch):
        image_batch, label_batch = (
            sampled_batch["image"],
            sampled_batch["label"],
        )  #  [b, c, h, w], [b, h, w]

        B, C, H, W = image_batch.shape

        image_batch, label_batch = image_batch.to(self.device), label_batch.to(
            self.device
        )

        image_batch = self.model_processor.preprocess(image_batch)

        output = self.model(image_batch)
        prob = output.softmax(1)
        pred = prob.argmax(1)

        if pred.shape[-2:] != label_batch.shape[-2:]:
            loss_label_batch = F.resize(
                label_batch,
                size=output.shape[-2:],
                interpolation=F.InterpolationMode.NEAREST,
            )
        else:
            loss_label_batch = label_batch

        pred = self.model_processor.postprocess(
            pred,
            label_batch.shape[-2:],
            do_denoise=self.config.postprocess_mask,
        )

        if hasattr(self, "supervised_loss"):
            loss = self.supervised_loss(output, loss_label_batch)
        else:
            loss = None

        pred = pred.cpu().numpy()
        label_batch = label_batch.cpu().numpy()

        metric_all = np.zeros((B, 4))
        metric_per_cls = np.zeros((B, self.config.num_classes, 4))

        if "spacing" in sampled_batch:
            spacing = sampled_batch["spacing"][0]
            spacing = torch.roll(spacing, 1).cpu().numpy()
        else:
            spacing = None

        for b in range(B):
            metric_all[b] = self.calculate_metric_percase(
                pred[b] > 0, label_batch[b] > 0, spacing
            )
            for c in range(1, self.config.num_classes + 1):
                metric_per_cls[b, c - 1] = self.calculate_metric_percase(
                    pred[b] == c, label_batch[b] == c, spacing
                )

        return metric_all, metric_per_cls, loss

    def valid_volumns(self, sampled_batch):
        image_batch, label_batch = (
            sampled_batch["image"],
            sampled_batch["label"],
        )  #  [1, c, d, h, w], [1, d, h, w]

        assert image_batch.shape[0] == 1

        image_batch, label_batch = image_batch.to(self.device), label_batch.to(
            self.device
        )
        image_batch = image_batch.squeeze(0).permute(1, 0, 2, 3)  # D, C, H, W
        label_batch = label_batch.squeeze(0)  # D, H, W

        image_batch = self.model_processor.preprocess(image_batch)

        output = self.model(image_batch)
        prob = output.softmax(1)
        pred = prob.argmax(1)

        if pred.shape[-2:] != label_batch.shape[-2:]:
            loss_label_batch = F.resize(
                label_batch,
                size=output.shape[-2:],
                interpolation=F.InterpolationMode.NEAREST,
            )
        else:
            loss_label_batch = label_batch

        pred = self.model_processor.postprocess(
            pred,
            label_batch.shape[-2:],
            do_denoise=self.config.postprocess_mask,
        )

        if hasattr(self, "supervised_loss"):
            loss = self.supervised_loss(output, loss_label_batch)
        else:
            loss = None

        pred = pred.cpu().numpy()
        label_batch = label_batch.cpu().numpy()

        metric_all = np.zeros((1, 4))
        metric_per_cls = np.zeros((1, self.config.num_classes, 4))

        if "spacing" in sampled_batch:
            spacing = sampled_batch["spacing"][0]
            spacing = torch.roll(spacing, 1).cpu().numpy()
        else:
            spacing = None

        metric_all[0] = self.calculate_metric_percase(
            pred > 0, label_batch > 0, spacing
        )

        for c in range(1, self.config.num_classes + 1):
            metric_per_cls[0, c - 1] = self.calculate_metric_percase(
                pred == c, label_batch == c, spacing
            )

        return metric_all, metric_per_cls, loss

    def calculate_metric_percase(
        self, pred: np.ndarray, gt: np.ndarray, spacing=None
    ):
        pred[pred > 0] = 1
        gt[gt > 0] = 1

        dice = 0
        hd = np.nan
        asd = np.nan
        jc = 0

        if pred.sum() > 0:
            dice = medpy_metric.dc(pred, gt)
            hd = cal_hd(pred, gt, spacing)
            asd = medpy_metric.asd(pred, gt, spacing)
            jc = medpy_metric.jc(pred, gt)

        return dice, hd, asd, jc

    def train(self):
        self.on_train_start()

        while self.current_round < self.config.num_rounds:
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
            if self.current_patience >= self.config.early_stop_max_patience:
                self.logger.info(
                    "Exceeded maximum patience. Training will be early stopped"
                )
                return True

        return self.current_iter >= self.config.num_iters

    def run_training(self):
        self.train()

    @torch.no_grad()
    def perform_real_test(self):
        self.model.eval()
        test_dataset = self.get_dataset("test", include_transform=True)

        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=self.config.valid_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=1,
            pin_memory=True,
        )

        metric_all_list = []
        metric_list = []

        for sampled_batch in tqdm(test_dataloader):
            if self.config.valid_mode == "volumn":
                metric_all, metric, _ = self.valid_volumns(sampled_batch)
            else:
                metric_all, metric, _ = self.valid_slices(sampled_batch)

            metric_all_list.extend(metric_all)
            metric_list.extend(metric)

        metric_all_tensor = torch.from_numpy(
            np.array(metric_all_list)
        )  # N, C, 4 (DSC, HD, ASD, JSD)
        metric_tensor = torch.from_numpy(
            np.array(metric_list)
        )  # N, C, 4 (DSC, HD, ASD, JSD)
        classes = test_dataset.CLASSES
        metric_name = {0: "DSC", 1: "HD", 2: "ASD", 3: "JSD"}

        dataframe_dict = {}
        for class_id in classes.keys():
            if class_id == 0:
                for metric_id in metric_name.keys():
                    field_name = f"all-{metric_name[metric_id]}"
                    dataframe_dict[field_name] = metric_all_tensor[
                        :, metric_id
                    ].tolist()
            else:
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

        avg_metric_all = metric_all_tensor.nanmean(0)
        avg_metric_per_cls = metric_tensor.nanmean(0)
        self.logger.info("Real test results (DSC, HD, ASD, JSD):")
        for id in classes.keys():
            if id == 0:
                self.logger.info(f"  all: {avg_metric_all.tolist()}")
            else:
                self.logger.info(
                    f"  {classes[id]}: {avg_metric_per_cls[id-1].tolist()}"
                )

        self.logger.info(f"Average: {avg_metric_per_cls.nanmean(0).tolist()}")

        if self.use_wandb:
            avg_metric = avg_metric_per_cls.nanmean(0)
            avg_dsc = avg_metric[0]
            avg_hd = avg_metric[1]
            avg_asd = avg_metric[2]
            avg_jc = avg_metric[3]
            test_metric = {
                f"test/metric/dsc_all": avg_metric_all[0].item(),
                f"test/metric/hd_all": avg_metric_all[1].item(),
                f"test/metric/dsc": avg_dsc.item(),
                f"test/metric/hd": avg_hd.item(),
                f"test/metric/asd": avg_asd.item(),
                f"test/metric/jc": avg_jc.item(),
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
