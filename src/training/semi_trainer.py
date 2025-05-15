import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Sequence

import torch
import torchvision.transforms.functional as F
from PIL import Image
from rich.console import Console
from rich.logging import RichHandler
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from datasets import LA2018Dataset
from losses.compound_losses import DC_and_CE_loss
from losses.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from metric.metric import HD
from models.unet import UNet
from scheduler.lr_scheduler import PolyLRScheduler
from transforms.common import ComposeTransform, RandomTransform
from transforms.image_transform import (RandomBrightness, RandomContrast,
                                        RandomGamma, RandomGaussianBlur,
                                        RandomGaussianNoise, SimulateLowRes)
from transforms.joint_transform import (JointResize, MirrorTransform,
                                        RandomAffine, RandomCrop2D,
                                        RandomRotation)
from transforms.normalization import ZScoreNormalize
from utils import dummy_context, get_path

from .base_trainer import BaseTrainer


class SemiTrainer(BaseTrainer):
    def __init__(
        self,
        work_path: Path | str = Path.cwd(),
        device: torch.device | str = torch.device("cuda"),
        seed: int = 12345,
        # Model parameters
        model_cls=UNet,
        in_channels: int = 3,
        num_classes: int = 2,
        patch_size: int | tuple[int, int] | None = None,
        image_size: int | tuple[int, int] | None = None,
        pretrained_model: Path | str | None = None,
        # Data parameters
        data_path: Path | str = "data",
        split_dict: Path | str | dict | None = None,
        num_folds: int | None = None,
        current_fold: int | str | None = None,
        labeled_ratio: float = 1.0,
        valid_ratio: float = 0.0,
        oversample_factor: int = 1,
        do_augment: bool = False,
        do_normalize: bool = False,
        batch_size: int = 32,
        num_workers: int = 1,
        pin_memory: bool = True,
        # Training parameters
        optimizer: Literal["adam", "adamw", "sgd"] = "adamw",
        optimizer_kwargs: dict = {},
        num_epochs: int = 1000,
        warmup_epochs: int = 0,
        start_lr: float = 1e-3,
        lr_scheduler: Literal["poly"] = "poly",
        save_freq: int = 10,
        save_metric: Literal["hd", "loss"] = "loss",
        early_stop_patient: int = 200,
        # Inference parameters
        stride: int | tuple[int, ...] | list[int] | None = None,
        # Log parameters
        verbose: bool = True,
        log_path: Path | str | None = None,
        log_mode: str = "a",
        log_override: bool = False,
    ):
        self.work_path = get_path(work_path)
        self.device = torch.device("cpu")
        self.to(device)

        self._set_seed(seed)

        # >>> Model parameters
        self.model_cls = model_cls
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.image_size = image_size
        self.pretrained_model = pretrained_model
        # <<< Model parameters

        # >>> Data parameters
        self.data_path = data_path
        self.split_dicts = split_dict
        self.num_folds = num_folds
        self.current_fold = current_fold
        self.labeled_ratio = labeled_ratio
        self.valid_ratio = valid_ratio
        self.oversample_factor = oversample_factor
        self.do_augment = do_augment
        self.do_normalize = do_normalize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        # <<< Data parameters

        # >>> Training parameters
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.start_lr = start_lr
        self.lr_scheduler = lr_scheduler
        self.save_freq = save_freq
        self.save_metric = save_metric
        self.early_stop_patient = early_stop_patient

        self.current_epoch = 0
        # <<< Training parameters

        # >>> Inference parameters
        self.stride = stride
        # <<< Inference parameters

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

    def _set_seed(self, seed: int):
        self.seed = seed
        torch.manual_seed(seed)

    def _setup_logger(self):
        self.logger = logging.getLogger("MIA.SemiTrainer")
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

    def _build_model(self, pretrained_model: str | Path | None = None):
        self.model = self.model_cls(self.in_channels)
        if pretrained_model:
            self.load_model_checkpoint(pretrained_model)
        self.model.init_head(self.num_classes)

    def _setup_split_dict(self):
        self.cur_split_dict_id = 0
        default_split_dict_path = self.work_path / "split_dicts.json"
        split_dicts = None

        try:
            if isinstance(self.split_dicts, (str, Path)):
                with open(self.split_dicts, "r") as f:
                    split_dicts = json.load(f)
        except:
            split_dicts = None

        if split_dicts:
            if not isinstance(split_dicts, list):
                split_dicts = [split_dicts]
        elif self.num_folds:
            split_dicts = self._get_cross_split_dicts(self.num_folds)
            if self.current_fold is not None and isinstance(
                self.current_fold, int
            ):
                split_dicts = [split_dicts[self.current_fold]]

        else:
            split_dicts = [self._get_random_split_dict()]

        self._assert_mutual_exclusive(split_dicts)

        with open(default_split_dict_path, "w") as f:
            json.dump(split_dicts, f, indent=2)

        self.split_dicts = split_dicts

    def _get_random_split_dict(self):
        assert self.valid_ratio >= 0
        assert self.labeled_ratio >= 0

        split_dict = {"labeled": [], "unlabeled": [], "valid": []}
        samples = []
        samples.extend(
            LA2018Dataset.find_samples(self.data_path, require_label=False)
        )
        samples_with_gt = [s for s in samples if s["labeled"]]
        samples_without_gt = [s for s in samples if not s["labeled"]]

        # Valid set is a part of samples set with ground truth
        valid_size = int(len(samples_with_gt) * self.valid_ratio)
        # Train set is the remaining samples
        train_size = len(samples) - valid_size
        # The labeled set of train set is limited to samples with ground truth
        labeled_size = min(
            train_size * self.labeled_ratio, len(samples_with_gt) - valid_size
        )

        # Shuffle the samples with ground truth
        perm_ids = torch.randperm(len(samples_with_gt))
        valid_ids = perm_ids[:valid_size]
        labeled_ids = perm_ids[valid_size : valid_size + labeled_size]

        for sample_id in range(len(samples_with_gt)):
            if sample_id in valid_ids:
                split_dict["valid"].append(samples_with_gt[sample_id])
            elif sample_id in labeled_ids:
                split_dict["labeled"].append(samples_with_gt[sample_id])
            else:
                split_dict["unlabeled"].append(samples_with_gt[sample_id])

        # The samples without ground truth is in the unlabeled set
        split_dict["unlabeled"].extend(samples_without_gt)

        return split_dict

    def _get_cross_split_dicts(
        self,
    ):
        assert self.num_folds and self.num_folds >= 2

        split_dicts = [{} for _ in range(self.num_folds)]
        samples = []
        samples.extend(
            LA2018Dataset.find_samples(self.data_path, require_label=False)
        )
        samples_with_gt = [s for s in samples if s["labeled"]]
        samples_without_gt = [s for s in samples if not s["labeled"]]

        # Valid set is a part of samples set with ground truth
        valid_size = int(len(samples_with_gt) / self.num_folds)
        # Train set is the remaining samples
        train_size = len(samples) - valid_size
        # The labeled set of train set is limited to samples with ground truth
        labeled_size = min(
            train_size * self.labeled_ratio, len(samples_with_gt) - valid_size
        )

        perm_ids = torch.randperm(len(samples_without_gt))

        for i in range(self.num_folds):
            split_dict = {"labeled": [], "unlabeled": [], "valid": []}
            start_id = i * valid_size
            end_id = (i + 1) * valid_size
            valid_ids = perm_ids[start_id:end_id]
            labeled_ids = torch.cat([perm_ids[:start_id], perm_ids[end_id]])[
                :labeled_size
            ]

            for sample_id in range(len(samples)):
                if sample_id in valid_ids:
                    split_dict["valid"].append(samples_with_gt[sample_id])
                elif sample_id in labeled_ids:
                    split_dict["labeled"].append(samples_with_gt[sample_id])
                else:
                    split_dict["unlabeled"].append(samples_with_gt[sample_id])
            split_dicts[i] = split_dict

        return split_dicts

    def _assert_mutual_exclusive(self, split_dicts: list[dict]):
        assert isinstance(split_dicts, list), "split_dict must be a list"

        for id, split_dict in enumerate(split_dicts):
            for dataset_name, subsets in split_dict.items():
                intersect = len(set(subsets["train"] + subsets["valid"]))
                total = len(subsets["train"]) + len(subsets["valid"])

                assert (
                    intersect == total
                ), f"Mutual samples in split_dict[{id}]/{dataset_name}"

    def _setup_seed(self, seed: int):
        torch.manual_seed(seed)

    def get_data(self, id: int = 0):
        assert isinstance(self.split_dicts, list)
        split_dict = self.split_dicts[id]

        labeled_dataset = LA2018Dataset(
            data_path=self.data_path,
            require_label=False,
            transform=self._get_train_transform(),
            normalize=self._get_train_normalize(),
            sample_ids=split_dict["labeled"],
            logger=self.logger,
        )

        labeled_dataloader = DataLoader(
            dataset=labeled_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        unlabeled_dataset = LA2018Dataset(
            data_path=self.data_path,
            require_label=False,
            transform=self._get_train_transform(),
            normalize=self._get_train_normalize(),
            sample_ids=split_dict["unlabeled"],
            logger=self.logger,
        )

        unlabeled_dataloader = DataLoader(
            dataset=unlabeled_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        valid_dataset = LA2018Dataset(
            data_path=self.data_path,
            require_label=False,
            transform=self._get_valid_transform(),
            normalize=self._get_valid_transform(),
            sample_ids=split_dict["valid"],
            logger=self.logger,
        )

        valid_dataloader = DataLoader(
            dataset=valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        return (
            labeled_dataset,
            labeled_dataloader,
            unlabeled_dataset,
            unlabeled_dataloader,
            valid_dataset,
            valid_dataloader,
        )

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
        state_dict = torch.load(
            pretrained_model, map_location="cpu", weights_only=True
        )
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
