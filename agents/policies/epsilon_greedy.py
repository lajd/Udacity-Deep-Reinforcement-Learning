import numpy as np
import torch
from tools.parameter_decay import ParameterDecay
from agents.policies.base import Policy
from typing import Callable, Optional


class EpsilonGreedyPolicy(Policy):
    def __init__(self, action_size: int, epsilon_decay_fn: Callable[[int], float], initial_eps: float = 1, final_eps: Optional[float] = None):
        super().__init__(action_size=action_size)
        self.epsilon_decayer = ParameterDecay(initial=initial_eps, lambda_fn=epsilon_decay_fn, final=final_eps)

        self.action_size = action_size

        # Initialize epsilon
        self.epsilon = self.epsilon_decayer.initial

    def step_episode(self, episode_number: int):
        self.epsilon = self.epsilon_decayer.get_param(episode_number)
        return True

    def get_action(self, action_values: np.array):
        """ Implement this function for speed"""
        if self.train:
            if torch.rand((1,)) <= self.epsilon:
                return int(torch.randint(0, self.action_size, (1,)))
            else:
                return int(torch.argmax(action_values).data)
        else:
            greedy_action = np.argmax(action_values)
            return greedy_action

    def get_deterministic_policy(self, state_action_values_dict: dict):
        deterministic_policy = {}
        for state in state_action_values_dict:
            deterministic_policy[state] = np.argmax(state_action_values_dict[state])
        return deterministic_policy
