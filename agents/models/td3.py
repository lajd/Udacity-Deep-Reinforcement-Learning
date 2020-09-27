import torch
import torch.nn as nn
from typing import Callable
from agents.models.components.critics import Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TD3Critic(nn.Module):
    def __init__(self, critic_model_factory: Callable[[], Critic], seed=123):
        super().__init__()

        self.q_network_a = critic_model_factory()
        self.q_network_b = critic_model_factory()

        self.q_network_a.set_seed(seed)
        self.q_network_b.set_seed(seed + 1)

    def forward(self, state, action):
        q1 = self.q_network_a(state, action)
        q2 = self.q_network_b(state, action)
        return q1, q2

    def qa(self, state, action):
        """ Forward pass through only one stream"""
        return self.q_network_a(state, action)


class MATD3Critic(nn.Module):
    def __init__(self, critic_model_factory: Callable[[], Critic], seed=123):
        super().__init__()

        self.q_network_a = critic_model_factory()
        self.q_network_b = critic_model_factory()

        self.q_network_a.set_seed(seed)
        self.q_network_b.set_seed(seed + 1)

    def forward(self, state, other_agent_states, other_agent_actions, action):
        q1 = self.q_network_a(state, other_agent_states, other_agent_actions, action)
        q2 = self.q_network_b(state, other_agent_states, other_agent_actions, action)
        return q1, q2

    def qa(self, state, other_agent_states, other_agent_actions, action):
        """ Forward pass through only one stream"""
        return self.q_network_a(state, other_agent_states, other_agent_actions, action)
