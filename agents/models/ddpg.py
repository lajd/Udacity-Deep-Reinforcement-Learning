import torch
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_AGENTS = 20
NUM_EPISODES = 200

SEED = 0
BATCH_SIZE = 256 # 128
NUM_STACKED_FRAMES = 1
REPLAY_BUFFER_SIZE = int(1e6)
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
# LR_CRITIC = 1e-3        # learning rate of the critic
N_LEARNING_ITERATIONS = 10     # number of learning updates
UPDATE_FREQUENCY = 20       # every n time step do update
WARMUP_STEPS = int(1e4)
MAX_T = 1000
CRITIC_WEIGHT_DECAY = 0.0#1e-2
ACTOR_WEIGHT_DECAY = 0.0
MIN_PRIORITY = 1e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hidden_init(layer):
    """ Initialize hidden layers """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


def output_layer_initialization(*args):
    """  We need the initialization for last layer of the Actor to be between -0.003 and 0.003 as this prevents us
     from getting 1 or -1 output values in the initial stages, which would squash our gradients to zero,
    as we use the tanh activation.
    """
    return -3e-3, 3e-3


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128, activation=F.leaky_relu):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*output_layer_initialization())

    def forward(self, state: torch.Tensor):
        """Build an actor (policy) network that maps states -> actions."""
        print(state.shape)
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128, activation=F.leaky_relu):
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
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*output_layer_initialization())

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.activation(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = self.activation(self.fc2(x))
        return self.fc3(x)


class TD3Actor(Actor):
    def __init__(self, state_dim, action_dim, seed: int, activation=F.leaky_relu):
        super().__init__(state_dim, action_dim, seed=seed, activation=activation)


class TD3Critic(nn.Module):
    def __init__(self, state_dim, action_dim, seed: int, activation=F.leaky_relu):
        super().__init__()

        self.seed = torch.manual_seed(seed)
        self.q_network_a = Critic(state_dim, action_dim, seed=1, activation=activation)
        self.q_network_b = Critic(state_dim, action_dim, seed=2, activation=activation)
        self.activation = activation

    def forward(self, state, action):
        q1 = self.q_network_a(state, action)
        q2 = self.q_network_b(state, action)
        return q1, q2

    def qa(self, state, action):
        """ Forward pass through only one stream"""
        return self.q_network_a(state, action)
