import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np
from tools.misc import set_seed
from agents.models.components import BaseComponent
from tools.misc import ensure_batch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Critic(BaseComponent):
    """Critic (Value) Model."""

    def __init__(self, output_module: nn.Module, state_featurizer: Optional[nn.Module], action_featurizer: Optional[nn.Module] = None, seed: Optional[int] = None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super().__init__()
        self.state_featurizer = state_featurizer
        self.action_featurizer = action_featurizer
        self.output_module = output_module
        self.set_seed(seed)

    @staticmethod
    def set_seed(seed):
        if seed:
            set_seed(seed)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if self.state_featurizer:
            state = self.state_featurizer(state)
        if self.action_featurizer:
            action = self.action_featurizer(action)
        x = torch.cat((state, action), dim=1)
        return self.output_module(x)


class MACritic(BaseComponent):
    """Multi agent Critic (Value) Model."""

    def __init__(self, output_module: nn.Module, state_featurizer: Optional[nn.Module], action_featurizer: Optional[nn.Module] = None, seed: Optional[int] = None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super().__init__()
        self.state_featurizer = state_featurizer
        self.action_featurizer = action_featurizer
        self.output_module = output_module
        self.set_seed(seed)

    @staticmethod
    def set_seed(seed):
        if seed:
            set_seed(seed)

    def forward(self, agent_state, other_agent_states, other_agent_actions, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        agent_state, other_agent_states, action, other_agent_actions = ensure_batch(agent_state, other_agent_states,
                                                                                    action, other_agent_actions)
        bsize = len(agent_state)

        other_agent_states = other_agent_states.view(bsize, -1).to(device)
        other_agent_actions = other_agent_actions.view(bsize, -1).to(device)

        agent_state = agent_state.to(device)

        action = action.float().to(device)

        state = torch.cat((agent_state, other_agent_states, other_agent_actions.float()), dim=1).view(bsize, -1)

        if self.state_featurizer:
            state = self.state_featurizer(state)
        if self.action_featurizer:
            action = self.action_featurizer(action)

        action = action.view(bsize, -1)
        x = torch.cat((state, action), dim=1)
        return self.output_module(x)
