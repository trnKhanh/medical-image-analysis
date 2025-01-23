import os
import shutil
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
from datasets.fugc import FUGCDataset
from losses.compound_losses import DC_and_CE_loss
from losses.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from metric.metric import HD
from scheduler.lr_scheduler import PolyLRScheduler

from utils import get_path, dummy_context
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
)
from transforms.common import RandomTransform, ComposeTransform


class UNetTrainer(BaseTrainer):
    def __init__(
        self,
        work_path: Path | str = Path.cwd(),
        device: torch.device | str = torch.device("cuda"),
        seed: int = 12345,
        # Model parameters
        num_classes: int = 2,
        image_size: int | tuple[int, int] | None = None,
        pretrained_model: Path | str | None = None,
        # Data parameters
        data_path: Path | str | list[Path] | list[str] = "data",
        data_split_dicts: Path | str | dict | None = None,
        data_num_folds: int | None = None,
        data_fold: int | str | None = None,
        data_valid_rate: float = 0.0,
        data_oversample: int = 10,
        data_augment: bool = True,
        data_normalize: bool = True,
        batch_size: int = 32,
        num_workers: int = 1,
        pin_memory: bool = False,
        # Optimizer parameters
        optimizer: Literal["adam", "adamw", "sgd"] = "adamw",
        optimizer_kwargs: dict = {},
        warmup_steps: int = 0,
        start_lr: float = 1e-3,
        lr_scheduler: Literal["poly"] = "poly",
        # Train parameters
        num_epochs: int = 1000,
        save_freq: int = 10,
        patient: int = 200,
        # Log parameters
        verbose: bool = True,
        log_path: Path | str | None = None,
        log_mode: str = "a",
        log_override: bool = False,
    ):
        self.work_path = get_path(work_path)
        self.device = torch.device("cpu")
        self.to(device)

        self.seed = seed
        torch.manual_seed(self.seed)

        # >>> Model parameters
        self.num_classes = num_classes
        self.image_size = image_size
        self.pretrained_model = pretrained_model
        # <<< Model parameters

        # >>> Data parameters
        if not isinstance(data_path, list):
            data_path = [get_path(data_path)]
        self.data_path = data_path
        self.data_split_dicts = data_split_dicts
        self.data_num_folds = data_num_folds
        self.data_fold = data_fold
        self.data_valid_rate = data_valid_rate
        self.data_oversample = data_oversample
        self.data_augment = data_augment
        self.data_normalize = data_normalize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        # <<< Data parameters

        # >>> Optimizer parameters
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.lr_scheduler = lr_scheduler
        # <<< Optimizer parameters

        # >>> Train parameters
        self.current_epoch = 0
        self.num_epochs = num_epochs
        self.save_freq = save_freq
        self.patient = patient
        # <<< Train parameters

        # >>> Log parameters
        self.verbose = verbose
        self.log_path = log_path
        self.log_mode = log_mode
        self.log_override = log_override
        # <<< Log parameters

    def initialize(self):
        self._setup_logger()
        self._setup_split_dict()
        self.model = self._build_model()

        self.metric = HD()

    def _setup_logger(self):
        self.logger = logging.getLogger("MIA.UNetTrainer")
        self.logger.setLevel(logging.DEBUG)

        self._setup_log_file()
        self._setup_log_shell()

    def _setup_log_file(self):
        if self.logger is None:
            raise RuntimeError(
                "UNetTrainer logger is not initialized before setting up log file"
            )

        if not self.log_path:
            return

        self.log_path = get_path(self.log_path)

        if self.log_path.exists() and not self.log_override:
            current_time_str = datetime.now().strftime("%d%m%Y_%H%M%S")
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

    def _setup_log_shell(self):
        if self.logger is None:
            raise RuntimeError(
                "UNetTrainer logger is not initialized before setting up log file"
            )

        if self.verbose:
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

    def _build_model(self, pretrained_model: str | Path | None = None):
        self.model = UNet(3)
        if pretrained_model:
            self.load_model_checkpoint(pretrained_model)
        self.model.init_head(self.num_classes)

    def _setup_split_dict(self):
        self.cur_split_dict_id = 0
        default_split_dict_path = self.work_path / "split_dicts.json"

        try:
            if isinstance(self.data_split_dicts, (str, Path)):
                with open(self.data_split_dicts, "r") as f:
                    self.data_split_dicts = json.load(f)
        except:
            self.data_split_dicts = None

        if self.data_split_dicts:
            if not isinstance(self.data_split_dicts, list):
                self.data_split_dicts = [self.data_split_dicts]
        elif self.data_num_folds:
            self.data_split_dicts = self._get_cross_split_dicts(
                self.data_num_folds
            )
            if self.data_fold is not None and isinstance(self.data_fold, int):
                self.data_split_dicts = [self.data_split_dicts[self.data_fold]]

        else:
            self.data_split_dicts = [
                self._get_random_split_dict(self.data_valid_rate)
            ]
        with open(default_split_dict_path, "w") as f:
            json.dump(self.data_split_dicts, f)
        self._assert_no_data_leak()

    def _assert_no_data_leak(self):
        assert isinstance(
            self.data_split_dicts, list
        ), "split_dicts must be a list"

        for id, split_dict in enumerate(self.data_split_dicts):
            for subset in split_dict.values():
                samples = set(subset["train"] + subset["valid"])
                assert len(samples) == len(subset["train"]) + len(
                    subset["valid"]
                ), f"data leaking in fold {id}"

    def _setup_seed(self, seed: int):
        torch.manual_seed(seed)

    def get_data(self, id: int = 0):
        assert isinstance(
            self.data_split_dicts, list
        ), "data_split_dict is not initialized"
        split_dict = self.data_split_dicts[id]

        train_datasets = []
        for data_path in self.data_path:
            train_datasets.append(
                FUGCDataset(
                    data_path=data_path,
                    transform=self._get_train_transform(),
                    normalize=self._get_train_normalize(),
                    split="train",
                    split_dict=split_dict[str(data_path)],
                    logger=self.logger,
                    oversample=self.data_oversample,
                )
            )
        train_dataset = ConcatDataset(train_datasets)

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        valid_datasets = []
        for data_path in self.data_path:
            valid_datasets.append(
                FUGCDataset(
                    data_path=data_path,
                    transform=self._get_valid_transform(),
                    normalize=self._get_valid_normalize(),
                    split="valid",
                    split_dict=split_dict[str(data_path)],
                    logger=self.logger,
                    oversample=self.data_oversample,
                )
            )
        valid_dataset = ConcatDataset(valid_datasets)

        valid_dataloader = DataLoader(
            dataset=valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        return train_dataloader, valid_dataloader, train_dataset, valid_dataset

    def _get_train_transform(self):
        transforms = []
        if self.data_augment:
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
        if self.data_normalize:
            return ZScoreNormalize()
        else:
            return None

    def _get_valid_transform(self):
        transforms = []
        if self.image_size:
            transforms.append(JointResize(self.image_size))
        return ComposeTransform(transforms)

    def _get_valid_normalize(self):
        if self.data_normalize:
            return ZScoreNormalize()
        else:
            return None

    def _get_random_split_dict(
        self,
        valid_rate: float = 0.0,
    ):
        assert (
            valid_rate >= 0
        ), f"valid_rate must be a non-negative float (>=0). Found {valid_rate}"
        split_dicts = {}
        for data_path in self.data_path:
            self.logger.info(f"Setting up split dict for {str(data_path)}")
            split_dict = {"train": [], "valid": []}
            samples = []
            samples.extend(FUGCDataset.get_samples(data_path))

            perm_ids = torch.randperm(len(samples))

            valid_size = int(len(samples) * valid_rate)
            valid_ids = perm_ids[:valid_size]

            for sample_id in range(len(samples)):
                if sample_id in valid_ids:
                    split_dict["valid"].append(samples[sample_id])
                else:
                    split_dict["train"].append(samples[sample_id])

            split_dicts[str(data_path)] = split_dict

        return split_dicts

    def _get_cross_split_dicts(
        self,
        num_folds: int = 5,
    ):
        assert (
            num_folds >= 2
        ), f"num_folds must be a postive integer >= 2. Found {num_folds}"
        split_dicts = [{} for _ in range(num_folds)]
        for data_path in self.data_path:
            self.logger.info(f"Setting up split dict for {str(data_path)}")
            samples = []
            samples.extend(FUGCDataset.get_samples(data_path))

            perm_ids = torch.randperm(len(samples))
            samples_per_split = int(len(samples) / num_folds)

            for i in range(num_folds):
                split_dict = {"train": [], "valid": []}
                start_id = i * samples_per_split
                end_id = (i + 1) * samples_per_split
                valid_ids = perm_ids[start_id:end_id]

                for sample_id in range(len(samples)):
                    if sample_id in valid_ids:
                        split_dict["valid"].append(samples[sample_id])
                    else:
                        split_dict["train"].append(samples[sample_id])
                split_dicts[i][str(data_path)] = split_dict
        return split_dicts

    def _get_optimizer(
        self, optimizer: str, lr_scheduler: str, model: nn.Module, **kwargs
    ):
        if optimizer == "adam":
            _optimizer = torch.optim.Adam(model.parameters(), **kwargs)
        elif optimizer == "adamw":
            _optimizer = torch.optim.AdamW(model.parameters(), **kwargs)
        elif optimizer == "sgd":
            _optimizer = torch.optim.SGD(model.parameters(), **kwargs)
        else:
            raise ValueError(f'Optimizer "{optimizer}" not supported')

        if lr_scheduler == "poly":
            _lr_scheduler = PolyLRScheduler(
                _optimizer, self.start_lr, self.num_epochs, self.warmup_steps
            )
        else:
            raise ValueError(
                f'Learning rate scheduler "{lr_scheduler}" not supported'
            )

        return _optimizer, _lr_scheduler

    def _get_loss(self):
        loss = DC_and_CE_loss(
            {"smooth": 1e-5, "do_bg": False},
            {},
            weight_ce=1,
            weight_dice=1,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        return loss

    def _print_train_info(self):
        self.logger.info(f"Training summary")
        self.logger.info(f"seed: {self.seed}")
        self.logger.info(f"device: {self.device}")
        self.logger.info(f"start_epoch: {self.current_epoch}")
        self.logger.info(f"num_epochs: {self.num_epochs}")
        self.logger.info(f'log_file: "{self.log_path}"')

        self.logger.info(f"model: {self.model}")
        self.logger.info(f"  pretrained_model: {self.pretrained_model}")
        self.logger.info(f"  image_size: {self.image_size}")

        self.logger.info(f'data: "{self.data_path}"')
        if self.data_num_folds:
            self.logger.info(f"  num_folds: {self.data_num_folds}")
            self.logger.info(f"  fold: {self.cur_split_dict_id}")
        else:
            self.logger.info(f"  valid_rate: {self.data_valid_rate}")
        self.logger.info(f"  train_samples: {len(self.train_dataset)}")
        self.logger.info(f"  valid_samples: {len(self.valid_dataset)}")
        self.logger.info(f"  oversample: {self.data_oversample}")
        self.logger.info(f"  augment: {self.data_augment}")
        if self.data_augment:
            self.logger.info(
                f"{json.dumps(self._get_train_transform().get_params_dict(), indent=1)}"
            )
        self.logger.info(f"  normalize: {self.data_normalize}")
        self.logger.info(f"  batch_size: {self.batch_size}")
        self.logger.info(f"  num_workers: {self.num_workers}")
        self.logger.info(f"  pin_memory: {self.pin_memory}")

        self.logger.info(f"optimizer: {self.optimizer}")
        self.logger.info(f"  warmup_steps: {self.warmup_steps}")
        self.logger.info(f"  lr_scheduler: {self.lr_scheduler}")
        self.logger.info(f"  start_lr: {self.start_lr}")
        self.logger.info(f"  optimizer_kwargs: {self.optimizer_kwargs}")

    def on_train_start(self):
        self._build_model(self.pretrained_model)

        self.current_epoch = 0
        self.current_patient = 0

        self._optimizer, self._lr_scheduler = self._get_optimizer(
            self.optimizer,
            self.lr_scheduler,
            self.model,
            **self.optimizer_kwargs,
        )
        self.loss = self._get_loss()

        self.model = self.model.to(self.device)

        self._best_valid_metric = torch.inf
        self._cur_valid_metric = torch.inf

        (
            self.train_dataloader,
            self.valid_dataloader,
            self.train_dataset,
            self.valid_dataset,
        ) = self.get_data(self.cur_split_dict_id)

        self._print_train_info()
        self._check_data_sanity()

    def _check_data_sanity(self):
        current_time_str = datetime.now().strftime("%d%m%Y_%H%M%S")
        sanity_path = self.work_path / "sanity" / current_time_str
        sanity_path.mkdir(parents=True, exist_ok=True)

        for i in range(50):
            image, _ = self.train_dataset[0]
            image_pil: Image.Image = F.to_pil_image(image)
            image_pil.save(str(sanity_path / f"{i + 1}.png"))

    def on_train_end(self):
        self.save_state_dict(
            self.work_path / f"fold_{self.cur_split_dict_id}" / "checkpoint.pth"
        )
        self.logger.info("")
        self.logger.info("")

    def on_epoch_start(self):
        self._epoch_start_time = time.time()

        self.logger.info("")
        self.logger.info(
            f"Epoch {self.current_epoch} (fold {self.cur_split_dict_id}):"
        )

    def on_epoch_end(self):
        self.current_epoch += 1

        self._epoch_end_time = time.time()
        time_elapsed = self._epoch_end_time - self._epoch_start_time
        self.logger.info(f"Epoch time elapsed: {time_elapsed:.3f} seconds")

    def on_train_epoch_start(self):
        self._train_start_time = time.time()
        self.logger.info("Train")

        self._lr_scheduler.step(self.current_epoch)
        self.logger.info(f"LR: {self._lr_scheduler.get_last_lr()}")
        self.epoch_train_outputs = []
        self.train_tqdm = tqdm(self.train_dataloader)

        self.model.train()

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % self.save_freq == 0:
            self.save_state_dict(
                self.work_path
                / f"fold_{self.cur_split_dict_id}"
                / "checkpoint.pth"
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
        self.logger.info(f"DICE per class: {global_dc_per_class}")
        self.logger.info(
            f"Mean DICE: {torch.Tensor(global_dc_per_class).mean()}"
        )
        train_loss = torch.stack(
            [o["loss"] for o in self.epoch_train_outputs]
        ).mean()
        self.logger.info(f"Loss: {train_loss.item()}")

        train_metric = (
            torch.stack([o["metric"] for o in self.epoch_train_outputs])
            .mean()
            .item()
        )
        self.logger.info(f"Metric ({self.metric}): {train_metric}")

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

        self.logger.info(f"DICE per class: {global_dc_per_class}")
        self.logger.info(
            f"Mean DICE: {torch.Tensor(global_dc_per_class).mean()}"
        )

        valid_loss = (
            torch.stack([o["loss"] for o in self.epoch_valid_outputs])
            .mean()
            .item()
        )
        self.logger.info(f"Loss: {valid_loss}")

        valid_metric = (
            torch.stack([o["metric"] for o in self.epoch_valid_outputs])
            .mean()
            .item()
        )
        self.logger.info(f"Metric ({self.metric}): {valid_metric}")

        self._cur_valid_metric = valid_metric

        if self._cur_valid_metric < self._best_valid_metric:
            self._best_valid_metric = self._cur_valid_metric
            self.logger.info(
                f"New best metric ({self.metric}): {self._cur_valid_metric}"
            )
            self.save_state_dict(
                self.work_path
                / f"fold_{self.cur_split_dict_id}"
                / "checkpoint_best.pth"
            )
            self.current_patient = 0
        else:
            self.current_patient += 1

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

    def train_step(self, data: torch.Tensor, target: torch.Tensor):
        data = data.to(self.device)
        target = target.to(self.device)

        self._optimizer.zero_grad()

        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.model(data)
            l: torch.Tensor = self.loss(output, target)

        l.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
        self._optimizer.step()

        axes = [0] + list(range(2, output.ndim))

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            self._get_one_hot_output(output), target, axes=axes
        )
        tp_hard = tp.detach().cpu()
        fp_hard = fp.detach().cpu()
        fn_hard = fn.detach().cpu()
        metric = torch.Tensor([self.metric(output, target[:, 0])])

        self.epoch_train_outputs.append(
            {
                "loss": l.detach().cpu(),
                "tp_hard": tp_hard,
                "fp_hard": fp_hard,
                "fn_hard": fn_hard,
                "metric": metric,
            }
        )

        train_loss = torch.stack(
            [o["loss"] for o in self.epoch_train_outputs]
        ).mean()

        self.train_tqdm.set_postfix(dict(train_loss=train_loss.item()))

    def valid_step(self, data: torch.Tensor, target: torch.Tensor):
        data = data.to(self.device)
        target = target.to(self.device)

        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.model(data)
            l: torch.Tensor = self.loss(output, target)

        axes = [0] + list(range(2, output.ndim))

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            self._get_one_hot_output(output), target, axes=axes
        )
        tp_hard = tp.detach().cpu()
        fp_hard = fp.detach().cpu()
        fn_hard = fn.detach().cpu()

        metric = torch.Tensor([self.metric(output, target[:, 0])])

        self.epoch_valid_outputs.append(
            {
                "loss": l.detach().cpu(),
                "tp_hard": tp_hard,
                "fp_hard": fp_hard,
                "fn_hard": fn_hard,
                "metric": metric,
            }
        )

        valid_loss = torch.stack(
            [o["loss"] for o in self.epoch_valid_outputs]
        ).mean()

        self.valid_tqdm.set_postfix(dict(valid_loss=valid_loss.item()))

    def train(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            if self.is_finished():
                break
            self.on_epoch_start()
            self.on_train_epoch_start()
            for data, target in self.train_tqdm:
                self.train_step(data, target)
            self.on_train_epoch_end()

            with torch.no_grad():
                self.on_valid_epoch_start()
                for data, target in self.valid_tqdm:
                    self.valid_step(data, target)
                self.on_valid_epoch_end()
            self.on_epoch_end()

        self.on_train_end()

    def is_finished(self):
        if isinstance(self.patient, int) and self.patient > 0:
            return self.current_patient >= self.patient

        return True

    def run_training(self):
        assert isinstance(
            self.data_split_dicts, list
        ), "data_split_dict is not initialized"

        while self.cur_split_dict_id < len(self.data_split_dicts):
            self.train()
            self.cur_split_dict_id += 1

    def valid(self):
        pass

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
        }

    def load_model_checkpoint(self, pretrained_model: str | Path):
        state_dict = torch.load(pretrained_model, map_location="cpu", weights_only=True)
        try:
            if "model" in state_dict:
                self.model.load_state_dict(state_dict["model"], strict=False)
            else:
                self.model.load_state_dict(state_dict, strict=False)
            self.logger.info(f"Load model checkpoint from {pretrained_model}")
        except Exception as e:
            self.logger.warn("Load model checkpoint failed")
            self.logger.exception(e)


    def load_state_dict(self, save_path: str | Path):
        state_dict = torch.load(save_path, map_location="cpu")
        try:
            if "model" in state_dict:
                self.model.load_state_dict(state_dict["model"])
            else:
                self.model.load_state_dict(state_dict)
        except:
            self.logger.warn("Load model checkpoint failed")
        # TODO: load state dict

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
