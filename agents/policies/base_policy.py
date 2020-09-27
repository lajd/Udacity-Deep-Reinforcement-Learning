import torch
import numpy as np
from abc import abstractmethod
import torch.nn.functional as F
from typing import Optional
from tools.misc import set_seed
from tools.rl_constants import ExperienceBatch, Action


class Policy:
    def __init__(self, action_size: int, training: bool = True, seed: Optional[int] = None):
        self.action_size = action_size
        self.actions = np.arange(self.action_size)
        self.training = training

        if seed:
            self.set_seed(seed)

    @staticmethod
    def set_seed(seed: int):
        set_seed(seed)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def step_episode(self, episode_number: int):
        pass

    @abstractmethod
    def get_action(self, state: np.array, model: torch.nn.Module) -> Action:
        pass

    @abstractmethod
    def get_random_action(self, *args) -> Action:
        pass

    def compute_errors(self, online_model, target_model, experience_batch: ExperienceBatch, gamma: float = 0.99) -> tuple:
        q = online_model(experience_batch.states)
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

    def compute_critic_errors(self, *args, **kwargs):
        raise NotImplementedError

    def compute_actor_errors(self, *args, **kwargs):
        raise NotImplementedError
