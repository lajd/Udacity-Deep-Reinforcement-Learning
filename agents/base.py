from abc import abstractmethod
import torch
import numpy as np
from typing import Tuple, Union
from tools.rl_constants import Experience
from tools.parameter_capture import ParameterCapture


class Agent:
    """ An agent which received state & reward from, and interacts with, and environment"""
    def __init__(self, state_shape: Union[Tuple[int, ...], int], action_size: int, num_agents: int):
        self.state_shape = state_shape
        self.action_size = action_size
        self.num_agents = num_agents

        self.warmup = False
        self.t_step = 0
        self.episode_counter = 0
        self.param_capture = ParameterCapture()
        self.training = True

    def set_warmup(self, warmup: bool):
        self.warmup = warmup

    @abstractmethod
    def set_mode(self, mode: str):
        pass

    def preprocess_state(self, state):
        """ Perform any state preprocessing """
        return state

    @abstractmethod
    def get_action(self, state: np.array) -> np.ndarray:
        """Determine an action given an environment state"""
        pass

    @abstractmethod
    def get_random_action(self, *args) -> np.ndarray:
        pass

    @abstractmethod
    def step(self, experience: Experience, **kwargs) -> None:
        """Take a step in the environment, encompassing model learning and memory population"""
        pass

    @abstractmethod
    def step_episode(self, episode: int) -> None:
        """Perform any end-of-episode updates"""
        pass
