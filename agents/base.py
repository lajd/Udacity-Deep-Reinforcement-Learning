from abc import abstractmethod
import numpy as np
from typing import Tuple, Union
from tools.rl_constants import Experience, ExperienceBatch, BrainSet, Action
from tools.parameter_capture import ParameterCapture


class Agent:
    """ An agent which received state & reward from, and interacts with, and environment"""
    def __init__(self, state_shape: Union[Tuple[int, ...], int], action_size: int):
        self.state_shape = state_shape
        self.action_size = action_size

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
    def get_action(self, state: np.array, *args, **kwargs) -> Action:
        """Determine an action given an environment state"""
        raise NotImplementedError

    @abstractmethod
    def get_random_action(self, *args, **kwargs) -> Action:
        raise NotImplementedError

    @abstractmethod
    def step(self, experience: Experience, **kwargs) -> None:
        """Take a step in the environment, encompassing model learning and memory population"""
        raise NotImplementedError

    @abstractmethod
    def step_episode(self, episode: int, *args) -> None:
        """Perform any end-of-episode updates"""
        raise NotImplementedError

    @abstractmethod
    def learn(self, experience_batch: ExperienceBatch):
        pass
