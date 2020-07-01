import torch
from typing import Optional
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class DummyLRScheduler(_LRScheduler):
    def __init__(self, network: Optimizer):
        super().__init__(network)

    def step(self, epoch: Optional[int] = None) -> None:
        pass
