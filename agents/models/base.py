import torch
from tools.misc import set_seed
from abc import abstractmethod


class BaseModel(torch.nn.Module):
    """" Base model class """
    def __init__(self, **kwargs):
        super().__init__()

    def set_seed(self, seed: int):
        """ Set seed of model for consistency """
        set_seed(seed)

    @abstractmethod
    def forward(self, state: torch.Tensor, act: bool = True) -> torch.Tensor:
        """ Generate the Q(state, a) for all possible actions a """
        return state

    @abstractmethod
    def dist(self, state: torch.Tensor, act: bool = True) -> torch.Tensor:
        """ Generate a distribution of Q(state, a) """
        return state

    def preprocess_state(self, state: torch.Tensor) -> torch.Tensor:
        """ Intercept the state from the environment and perform
        preprocessing to it before it reaches the model
        """
        return state

    def step(self):
        """ Perform internal updates after a back-propagation step """
        pass

    def step_episode(self, i_episode):
        """ Perform additional internal updates after an end-of-episode"""
        pass
