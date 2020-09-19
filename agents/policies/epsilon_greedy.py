import numpy as np
import torch
from tools.parameter_scheduler import ParameterScheduler
from agents.policies.base_policy import Policy
from tools.rl_constants import Action
import random


class EpsilonGreedyPolicy(Policy):
    """ Traditional epsilon-greedy policy

    Epsilon is annealed according to the `epsilon_scheduler`, which is
    updated at each invocation of `step`.

    The selected action is random with probability epsilon, and argmax(Q(s, a)) otherwise
    """
    def __init__(self, action_size: int, epsilon_scheduler: ParameterScheduler, seed: int = None):
        super().__init__(action_size=action_size, seed=seed)
        self.epsilon_scheduler = epsilon_scheduler

        self.action_size = action_size

        # Initialize epsilon
        self.epsilon = self.epsilon_scheduler.initial

    def step_episode(self, episode_number: int):
        self.epsilon = self.epsilon_scheduler.get_param(episode_number)
        return True

    def get_action(self, state: np.array, model: torch.nn.Module) -> Action:
        def _get_greedy_action():
            model.eval()
            with torch.no_grad():
                action_values = model.forward(state, act=True)
                action_ = action_values.max(1)[1].cpu().numpy()
            model.train()
            return action_

        if self.training:
            if random.random() > self.epsilon:
                action = _get_greedy_action()
            else:
                action = torch.randint(0, self.action_size, (1,)).numpy()
        else:
            action = _get_greedy_action()

        return Action(value=action)

    def get_deterministic_policy(self, state_action_values_dict: dict):
        deterministic_policy = {}
        for state in state_action_values_dict:
            deterministic_policy[state] = np.argmax(state_action_values_dict[state])
        return deterministic_policy
