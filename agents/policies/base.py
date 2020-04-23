
import numpy as np
from abc import abstractmethod


class Policy:
    def __init__(self, action_size: int, train: bool = True):
        self.action_size = action_size
        self.actions = np.arange(self.action_size)

        self.train = train

    def train(self):
        self.train = True

    def eval(self):
        self.train = False

    @abstractmethod
    def step_episode(self, episode_number: int):
        pass

    @abstractmethod
    def get_action(self, action_values: np.array):
        pass

    @abstractmethod
    def get_greedy_action(self, action_values: np.array):
        pass
