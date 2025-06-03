from typing import Optional
import torch
from torch.optim.lr_scheduler import _LRScheduler


class PolyLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        initial_lr: float,
        max_steps: int,
        warmup_steps: int,
        exponent: float = 0.9,
        current_step: int | None = None,
        interval: int = 1,
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.exponent = exponent
        self.interval = interval
        self.ctr = 0

        self._adjusted_warmup_steps = self.warmup_steps // self.interval
        self._adjusted_max_steps = self.max_steps // self.interval
        super().__init__(
            optimizer, current_step if current_step is not None else -1
        )

    def step(self, epoch: Optional[int] = None):
        if epoch is None or epoch == -1:
            epoch = self.ctr
            self.ctr += 1

        step_index = epoch // self.interval

        if step_index < self._adjusted_warmup_steps:
            new_lr = self.initial_lr * (step_index + 1) / self._adjusted_warmup_steps
        else:
            step_index = step_index - self._adjusted_warmup_steps
            real_max_steps = self._adjusted_max_steps - self._adjusted_warmup_steps

            new_lr = (
                self.initial_lr
                * (1.0 - step_index / real_max_steps) ** self.exponent
            )

        for param_group in self.optimizer.param_groups:
            if isinstance(param_group["lr"], torch.Tensor):
                param_group["lr"].fill_(new_lr)
            else:
                param_group["lr"] = new_lr

        self._last_lr = [p["lr"] for p in self.optimizer.param_groups]
