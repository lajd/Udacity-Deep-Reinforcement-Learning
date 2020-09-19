from tools.misc import set_seed
import torch


class BaseComponent(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def set_seed(seed):
        if seed:
            set_seed(seed)
