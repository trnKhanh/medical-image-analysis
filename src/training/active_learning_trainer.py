__all__ = ["ActiveLearningTrainer"]

import json
import logging
import random
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image
from rich.console import Console
from rich.logging import RichHandler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import ACDCDataset, TwoStreamBatchSampler
from losses.compound_losses import DC_and_CE_loss
from losses.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from models import UNet
from scheduler.lr_scheduler import PolyLRScheduler
from transforms.common import (ComposeTransform, RandomChoiceTransform,
                               RandomTransform)
from transforms.joint_transform import (JointResize, MirrorTransform,
                                        RandomAffine, RandomRotation90)
from transforms.normalization import ZScoreNormalize
from utils import draw_mask, dummy_context, get_path
from metric.metric import HD

from .base_trainer import BaseTrainer


class ActiveLearningTrainer(BaseTrainer):
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
        model_ckpt: Path | str | None = None,
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
        num_cycles: int = 10,
        budget: int = 50,
        initial_size: int = 200,
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
        query_strategy: Literal["uncertainty", "entropy", "margin", "bald"] = "uncertainty",
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
        self.model_ckpt = model_ckpt
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
        self.num_cycles = num_cycles
        self.budget = budget
        self.initial_size = initial_size
        self.min_iter = min_iter
        self.warmup_iter = warmup_iter
        self.start_lr = start_lr
        self.lr_scheduler_name = lr_scheduler_name
        self.lr_warmup_iter = lr_warmup_iter
        self.save_freq_epoch = save_freq_epoch
        self.valid_freq_iter = valid_freq_iter
        self.save_metric_name = save_metric_name
        self.dice_weight = dice_weight
        self.query_strategy = query_strategy
        self.early_stop_max_patience = early_stop_max_patience

        self.current_epoch = 0
        self.current_round = 0
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

        self.metric = HD()

    def _set_snapshot_work_dir(self):
        current_time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        snapshot_list = [
            f"ACDC",
            f"{current_time_str}",
            f"patchsize-{self.patch_size}",
            f"imagesize-{self.image_size}",
            f"labeled-{self.labeled_num}",
            f"batchsize-{self.batch_size}",
            f"optimizer-{self.optimizer_name}",
            f"lrscheduler-{self.lr_scheduler_name}",
            f"lrwarmup-{self.lr_warmup_iter}",
            f"startlr-{self.start_lr}",
            f"dice-{self.dice_weight}",
        ]
        if self.exp_name:
            snapshot_list.append(self.exp_name)
        snapshot_str = "_".join(snapshot_list)
        self.work_path = self.work_path / snapshot_str
        self.work_path.mkdir(parents=True, exist_ok=True)

    def _set_seed(self, seed: int) -> None:
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def _setup_logger(self) -> None:
        self.logger = logging.getLogger("MIA.ActiveLearningTrainer")
        self.logger.setLevel(logging.DEBUG)

        self._setup_log_file()
        self._setup_log_shell()

    def _setup_log_file(self) -> None:
        assert self.logger is not None

        if not self.log_path:
            self.log_path = self.work_path / "log.txt"

        self.log_path = get_path(self.log_path)

        if self.log_path.exists() and not self.log_override:
            current_time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
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

    def _add_config_file(self) -> None:
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

    def _build_model(self) -> None:
        self.model = UNet(n_channels=1, n_classes=self.num_classes).to(self.device)
        if self.model_ckpt:
            self.model.load_state_dict(torch.load(self.model_ckpt, map_location=self.device))

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
        labeled_dataset = ACDCDataset(
            data_path=self.data_path,
            split="train",
            normalize=self._get_train_normalize(),
            transform=self._get_train_transform(),
            logger=self.logger,
        )
        pool_dataset = ACDCDataset(
            data_path=self.data_path,
            split="train",
            normalize=self._get_train_normalize(),
            transform=None,
            logger=self.logger,
            image_channels=self.in_channels,
            image_size=self.image_size
        )



    def get_unlabeled_indices(self) -> list:
        total_slices = len(self.train_dataset)
        labeled_slices = self.patients_to_slices("ACDC", self.labeled_num)
        labeled_indices = list(range(0, labeled_slices))
        unlabeled_indices = list(range(labeled_slices, total_slices))
        return unlabeled_indices

    def query_new_samples(self):
        self.model.eval()
        uncertainties = []

        total_slice = len(self.train_dataset)
        labeled_slice = self.patients_to_slices("ACDC", self.labeled_num)

        with torch.no_grad():
            for idx in unlabeled_indices:
                sample = self.train_dataset[idx]
                input_tensor = sample['image'].unsqueeze(0).to(self.device)  # Adjust based on your dataset

                output = self.model(input_tensor)  # logits shape: (1, num_classes, H, W) or (1, num_classes)
                probs = torch.softmax(output, dim=1)  # shape: (1, num_classes, ...)

                # Flatten spatial dims if segmentation, else directly use probs
                probs_flat = probs.view(probs.size(0), probs.size(1), -1)  # (1, C, N)
                probs_mean = probs_flat.mean(dim=2).squeeze(0)  # (num_classes,)

                if self.query_strategy == "uncertainty":
                    # Uncertainty = 1 - max class probability
                    uncertainty = 1 - probs_mean.max().item()

                elif self.query_strategy == "entropy":
                    # Entropy = -sum(p * log p)
                    entropy = -torch.sum(probs_mean * torch.log(probs_mean + 1e-8))
                    uncertainty = entropy.item()

                elif self.query_strategy == "margin":
                    # Margin = difference between top two probs (smaller margin = higher uncertainty)
                    top2 = torch.topk(probs_mean, 2).values
                    margin = top2[0] - top2[1]
                    uncertainty = -margin.item()  # negative margin so higher uncertainty = bigger value

                elif self.query_strategy == "bald":
                    # BALD requires MC Dropout or ensembles; here's a dummy placeholder:
                    # Replace with proper BALD implementation if you have MC Dropout or ensembles.
                    uncertainty = entropy.item()  # fallback to entropy for now

                else:
                    raise ValueError(f"Unknown query strategy: {self.query_strategy}")

                uncertainties.append((idx, uncertainty))

        # Sort descending by uncertainty
        uncertainties.sort(key=lambda x: x[1], reverse=True)

        queried = [idx for idx, _ in uncertainties[:budget]]

        new_labeled_indices = labeled_indices + queried
        new_unlabeled_indices = [idx for idx in unlabeled_indices if idx not in queried]

        return new_labeled_indices, new_unlabeled_indices


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
        return None

    def _get_valid_transform(self):
        return None

    def _get_valid_normalize(self):
        if self.do_normalize:
            return ZScoreNormalize()
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
        loss = DC_and_CE_loss(
            {"smooth": 1e-5, "do_bg": False},
            {},
            weight_ce=1,
            weight_dice=1,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        return loss

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
        self.logger.info(f"  model_ckpt: {self.model_ckpt}")

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
        self.logger.info(f"save_metric: {self.save_metric_name}")
        self.logger.info(f"start_epoch: {self.current_epoch}")
        self.logger.info(f"num_epochs: {self.num_epochs}")
        self.logger.info(f"num_cycles: {self.num_cycles}")
        self.logger.info(f"warmup_iter: {self.warmup_iter}")
        self.logger.info(f"save_freq_epoch: {self.save_freq_epoch}")
        self.logger.info(f"valid_freq_iter: {self.valid_freq_iter}")
        self.logger.info(f"dice_weight: {self.dice_weight}")
        self.logger.info(
            f"early_stop_max_patience: {self.early_stop_max_patience}"
        )

        self._remove_config_file()

    def on_train_start(self):
        assert self.model is not None

        self.model.train()
        self.model.to(self.device)

        (
            self.train_dataset,
            self.valid_dataset,
            self.train_dataloader,
            self.valid_dataloader,
        ) = self.get_data()

        self.max_iterations = self.num_epochs * len(self.train_dataloader)

        self.current_round = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.current_patient = 0

        self._optimizer, self._lr_scheduler = self._get_optimizer(self.model)

        self._best_valid_metric = torch.inf
        self._cur_valid_metric = torch.inf

        self._best_valid_prompt_metric = torch.inf
        self._cur_valid_prompt_metric = torch.inf

        self._print_train_info()
        self._check_data_sanity()

    def _check_data_sanity(self) -> None:
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

    def on_train_end(self) -> None:
            self.save_state_dict(self.work_path / "ckpt/final_model")
            self.logger.info("")
            self.logger.info("")

    def on_epoch_start(self) -> None:
        self._epoch_start_time = time.time()

        self.logger.info("")
        self.logger.info(f"Epoch {self.current_epoch + 1}/{self.num_epochs}")

    def on_epoch_end(self) -> None:
        self._lr_scheduler.step()
        self.logger.info(f"lr: {self._lr_scheduler.get_lr()[0]:.6f}")
        self.logger.info(f"time: {(time.time() - self._epoch_start_time):.2f}s")

    def on_train_epoch_start(self) -> None:
        self._train_epoch_start_time = time.time()
        self.logger.info("Train")

        self._lr_scheduler.step(self.current_epoch)
        self.logger.info(f"LR: {self._lr_scheduler.get_last_lr()}")
        self.epoch_train_outputs = []
        self.train_tqdm = tqdm(self.train_dataloader)

        self.model.train()

    def on_train_epoch_end(self) -> None:
        if (self.current_epoch + 1) % self.save_freq_epoch == 0:
            self.save_state_dict(self.work_path / f"ckpt/epoch_{self.current_epoch + 1}")

        self._train_end_time = time.time()
        time_elapsed = self._train_end_time - self._train_epoch_start_time
        self.logger.info(f"Train time elapsed: {time_elapsed:.2f}s")

    def on_valid_epoch_start(self) -> None:
        self._valid_start_time = time.time()
        self.logger.info("Valid")

        self.model.eval()
        self.valid_tqdm = tqdm(self.valid_dataloader)
        self.epoch_valid_outputs = []

    def on_valid_epoch_end(self) -> None:
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

    def valid_step(self, sampled_batch):
        pass

    def cycle_step(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            if self.is_finished():
                break
            self.on_epoch_start()
            self.on_train_epoch_start()
            for sampled_batch in self.train_tqdm:
                data = sampled_batch["image"]
                target = sampled_batch["label"]
                self.train_step(data, target)
            self.on_train_epoch_end()

            with torch.no_grad():
                self.on_valid_epoch_start()
                for sampled_batch in self.valid_tqdm:
                    data = sampled_batch["image"]
                    target = sampled_batch["label"]
                    self.valid_step(data, target)
                self.on_valid_epoch_end()
            self.on_epoch_end()

        self.on_train_end()

    def train(self):
        for i in range(self.num_cycles):
            self.cycle_step()

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

    def save_model_checkpoint(self, ckpt: Union[str, Path]) -> None:
        if self.model is None:
            raise RuntimeError("Model is not initialized")

        try:
            torch.save(self.model.state_dict(), ckpt)
            self.logger.info(f"Model checkpoint saved to {ckpt}")
        except Exception as e:
            self.logger.error(f"Error saving model checkpoint: {e}")
            self.logger.error(traceback.format_exc())

    def save_state_dict(self, save_path: Union[str, Path]):
        save_path = get_path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        self.save_model

    def to(self, device: Union[torch.device | str]) -> None:
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda" and torch.cuda.is_available():
            self.device = device
        elif device.type == "mps" and torch.backends.mps.is_available():
            self.device = device
        else:
            self.device = torch.device("cpu")