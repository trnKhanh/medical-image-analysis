from typing import Optional
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
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(
            optimizer, current_step if current_step is not None else -1
        )

    def step(self, epoch: Optional[int] = None):
        if epoch is None or epoch == -1:
            epoch = self.ctr
            self.ctr += 1

        if epoch < self.warmup_steps:
            new_lr = self.initial_lr * (epoch + 1) / (self.warmup_steps + 1)
        else:
            real_epoch = epoch - self.warmup_steps
            real_max_steps = self.max_steps - self.warmup_steps

            new_lr = (
                self.initial_lr
                * (1 - real_epoch / real_max_steps) ** self.exponent
            )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        self._last_lr = [p["lr"] for p in self.optimizer.param_groups]
