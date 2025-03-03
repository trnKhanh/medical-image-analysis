from pathlib import Path
from abc import ABC, abstractmethod

import torch

class BaseTrainer(ABC):
    @abstractmethod
    def on_train_start(self):
        pass

    @abstractmethod
    def on_train_end(self):
        pass

    @abstractmethod
    def on_train_epoch_start(self):
        pass

    @abstractmethod
    def on_train_epoch_end(self):
        pass

    @abstractmethod
    def on_valid_epoch_start(self):
        pass

    @abstractmethod
    def on_valid_epoch_end(self):
        pass

    @abstractmethod
    def train_step(self, data: torch.Tensor, target: torch.Tensor):
        pass

    @abstractmethod
    def valid_step(self, data: torch.Tensor, target: torch.Tensor):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def valid(self):
        pass

    @abstractmethod
    def state_dict(self) -> dict:
        pass

    @abstractmethod
    def load_state_dict(self, save_path: str | Path):
        pass

    @abstractmethod
    def save_state_dict(self, save_path: str | Path):
        pass

    @abstractmethod
    def to(self, device: torch.device | str):
        pass

