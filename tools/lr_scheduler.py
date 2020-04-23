import torch
from torch.optim import Optimizer
from typing import Callable


class LRScheduler:
    def __init__(self, optimizer: Optimizer, initial_lr: float, lambda_fn: Callable, method: str = 'lambda_lr', last_epoch: int = -1):
        self.optimizer = optimizer
        self.initial_lr = initial_lr

        if method == 'lambda_lr':
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda_fn, last_epoch=last_epoch)
        else:
            raise ValueError("Method {} not implemented")

        self.scheduler = scheduler

    def step(self):
        self.scheduler.step()
