import torch
from collections import namedtuple


class Experience:
    def __init__(self, state: torch.Tensor, action:int, reward: float, next_state: torch.Tensor, done: torch.Tensor):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

    def as_tuple(self):
        experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        experience.state = self.state
        experience.action = self.action
        experience.reward = self.reward
        experience.next_state = self.next_state
        experience.done = self.done
