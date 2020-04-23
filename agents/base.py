from abc import abstractmethod
import torch
import numpy as np
import datetime
from agents.policies.base import Policy
from tools.lr_scheduler import LRScheduler


class Agent(torch.nn.Module):
    def __init__(self, state_size: int, action_size: int, policy: Policy, optimizer: torch.optim.Optimizer, lr_scheduler: LRScheduler):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        # Define the policy
        self.policy = policy

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def set_mode(self, mode: str):
        if mode == 'train':
            self.train()
        elif mode == 'evaluate':
            self.eval()
            self.policy.eval()  # Make the policy greedy
        else:
            raise ValueError("only modes `train`, `evaluate` are supported")

    @abstractmethod
    def get_agent_networks(self):
        pass

    @abstractmethod
    def act(self, state: np.array):
        pass

    @abstractmethod
    def step(self, state: np.array, action: np.array, reward: np.array, next_state: np.array, done: np.array, **kwargs):
        pass

    @abstractmethod
    def step_episode(self, episode: int):
        pass

    def checkpoint(self, tag: str, checkpoint_dir: str):
        print('Creating checkpoint for tag {}'.format(tag))
        agent_networks = self.get_agent_networks()
        for parameter_name, network in agent_networks.items():
            torch.save(network.state_dict(), '{}/{}_{}_checkpoint_{}.pth'.format(checkpoint_dir, tag, parameter_name, str(datetime.datetime.utcnow())))
        return True
