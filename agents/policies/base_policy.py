import torch
import numpy as np
from abc import abstractmethod
import torch.nn.functional as F
from typing import Optional
from tools.misc import set_seed
from tools.rl_constants import ExperienceBatch


class Policy:
    def __init__(self, action_size: int, train: bool = True, seed: Optional[int] = None):
        self.action_size = action_size
        self.actions = np.arange(self.action_size)
        self.train = train

        if seed:
            self.set_seed(seed)

    @staticmethod
    def set_seed(seed: int):
        set_seed(seed)

    def train(self):
        self.train = True

    def eval(self):
        self.train = False

    def step(self, episode_number: int):
        pass

    @abstractmethod
    def get_action(self, state: np.array, model: torch.nn.Module) -> np.ndarray:
        pass

    @abstractmethod
    def get_random_action(self, *args) -> np.ndarray:
        pass

    # Acts with an ε-greedy policy (used for evaluation)
    def get_action_e_greedy(self, state: np.array, model: torch.nn.Module, epsilon=0.001) -> np.ndarray:  # High ε can reduce evaluation scores drastically
        if np.random.random() < epsilon:
            action = np.array(np.random.choice(self.actions))
        else:
            action = self.get_action(state, model)
        return action

    def compute_errors(self, online_model, target_model, experience_batch: ExperienceBatch, gamma: float = 0.99) -> tuple:
        q = online_model(experience_batch.states).squeeze()
        q_next = online_model(experience_batch.next_states)
        next_q_target = target_model(experience_batch.next_states)

        qa = q.gather(1, experience_batch.actions)
        qa_next = next_q_target.gather(1, torch.max(q_next, 1)[1].unsqueeze(1))
        expected_q_value = experience_batch.rewards + gamma * qa_next * (1 - experience_batch.dones)

        errors = F.mse_loss(qa, torch.autograd.Variable(expected_q_value.data), reduction='none')

        # Get Loss and TD Errors
        if experience_batch.is_weights is not None:
            errors = errors * experience_batch.get_norm_is_weights().reshape_as(errors)

        loss = errors.mean()
        return loss, errors
