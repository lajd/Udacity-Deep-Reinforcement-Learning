import numpy as np
import torch
from tools.parameter_decay import ParameterScheduler
from agents.policies.base_policy import Policy
import random


class EpsilonGreedySoftmaxPolicy(Policy):
    def __init__(self, action_size: int, epsilon_scheduler: ParameterScheduler, seed: int = None):
        super().__init__(action_size=action_size)
        self.epsilon_scheduler = epsilon_scheduler

        self.action_size = action_size

        # Initialize epsilon
        self.epsilon = self.epsilon_scheduler.initial

        if seed:
            self.set_seed(seed)

    def step(self, episode_number: int):
        self.epsilon = self.epsilon_scheduler.get_param(episode_number)
        return True

    def get_action(self, state: np.array, model: torch.nn.Module) -> np.ndarray:
        """ Implement this function for speed"""

        def _get_action_values():
            model.eval()
            with torch.no_grad():
                action_values = model.forward(state, act=True)
            model.train()
            return action_values

        if self.train:
            action_values_ = _get_action_values()
            if random.random() > self.epsilon:
                action = action_values_.max(1)[1].data[0]
                return action
            else:
                probs = torch.nn.functional.softmax(action_values_)
                action = np.random.choice(np.arange(0, self.action_size), p=probs.view(-1).numpy())
                return action
        else:
            action_values_ = _get_action_values()
            action = action_values_.max(1)[1].data[0]
            return action

    def get_deterministic_policy(self, state_action_values_dict: dict):
        deterministic_policy = {}
        for state in state_action_values_dict:
            deterministic_policy[state] = np.argmax(state_action_values_dict[state])
        return deterministic_policy
