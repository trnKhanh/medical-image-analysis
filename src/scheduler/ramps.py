from abc import ABC, abstractmethod
from typing import Any

import torch
import numpy as np


class BaseRampUp(ABC):
    @abstractmethod
    def step(self, step_index: int | None = None) -> Any:
        pass


class SigmoidRampUp(BaseRampUp):
    def __init__(
        self,
        final_value: float,
        max_steps: int,
        interval: int = 1,
        exponent: float = 5.0,
    ):
        self.final_value = final_value
        self.max_steps = max_steps
        self.interval = interval
        self.exponent = exponent
        self.ctr = 0

        self._adjusted_max_steps = self.max_steps // self.interval

    def step(self, step_index: int | None = None):
        if step_index is None:
            step_index = self.ctr
            self.ctr += 1

        step_index = step_index // self.interval

        if self._adjusted_max_steps == 0:
            return self.final_value
        else:
            step_index = int(np.clip(step_index, 0, self._adjusted_max_steps))
            phase = 1.0 - step_index / self._adjusted_max_steps
            return self.final_value * float(np.exp(-self.exponent * phase**2))


class LinearRampUp(BaseRampUp):
    def __init__(
        self,
        final_value: float,
        max_steps: int,
        interval: int = 1,
    ):
        self.final_value = final_value
        self.max_steps = max_steps
        self.interval = interval
        self.ctr = 0

        self._adjusted_max_steps = self.max_steps // self.interval

    def step(self, step_index: int | None = None):
        if step_index is None:
            step_index = self.ctr
            self.ctr += 1

        step_index = step_index // self.interval

        if self._adjusted_max_steps == 0:
            return self.final_value
        else:
            step_index = int(np.clip(step_index, 0, self._adjusted_max_steps))
            return self.final_value * step_index / self._adjusted_max_steps
