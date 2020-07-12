from abc import abstractmethod
import torch
import numpy as np
from typing import Tuple
from agents.policies.base_policy import Policy
from torch.optim.lr_scheduler import _LRScheduler
from tools.rl_constants import Experience, Action
from tools.parameter_capture import ParameterCapture


class Agent(torch.nn.Module):
    """ An agent which received state & reward from, and interacts with, and environment"""
    def __init__(self, state_shape: Tuple[int, ...], action_size: int, policy: Policy, optimizer: torch.optim.Optimizer, lr_scheduler: _LRScheduler):
        super().__init__()
        self.state_shape = state_shape
        self.action_size = action_size

        self.policy: Policy = policy
        self.optimizer: optimizer = optimizer
        self.lr_scheduler: _LRScheduler = lr_scheduler

        self.param_capture = ParameterCapture()

    def set_mode(self, mode: str):
        """ Set the mode of the agent """
        if mode == 'train':
            self.train()
            self.policy.train = True
        elif mode.startswith('eval'):
            self.eval()
            self.policy.eval()  # Make the policy greedy
        else:
            raise ValueError("only modes `train`, `evaluate` are supported")

    def preprocess_state(self, state: torch.Tensor):
        return state

    @abstractmethod
    def load(self, *args, **kwargs):
        """ Load the agent model """
        pass

    @abstractmethod
    def get_action(self, state: np.array) -> Action:
        """Determine an action given an environment state"""
        pass

    @abstractmethod
    def get_random_action(self, *args) -> Action:
        pass

    @abstractmethod
    def step(self, experience: Experience, **kwargs) -> None:
        """Take a step in the environment, encompassing model learning and memory population"""
        pass

    @abstractmethod
    def step_episode(self, episode: int) -> None:
        """Perform any end-of-episode updates"""
        pass
