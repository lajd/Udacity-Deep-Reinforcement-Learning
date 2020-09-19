import os
import torch
from agents.ddpg_agent import DDPGAgent
from agents.policies.td3_policy import TD3Policy
from agents.memory.prioritized_memory import PrioritizedMemory
from agents.memory.memory import Memory
from tasks.crawler.solutions.utils import get_simulator, STATE_SIZE, ACTION_SIZE, BRAIN_NAME, NUM_AGENTS
from tasks.crawler.solutions.ddpg import SOLUTIONS_CHECKPOINT_DIR
from tools.lr_schedulers import DummyLRScheduler
from tools.parameter_scheduler import ParameterScheduler
import pickle
from tools.rl_constants import BrainSet, Brain, RandomBrainAction
from agents.models.components.mlp import MLP
from tools.layer_initializations import init_layer_inverse_root_fan_in, init_layer_within_range
from agents.models.components.critics import Critic
from agents.models.td3 import TD3Critic

from torch import nn
from torch.nn import functional as F
import numpy as np

NUM_EPISODES = 3000
SEED = 0
BATCH_SIZE = 1024
REPLAY_BUFFER_SIZE = int(1e6)
GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters
N_LEARNING_ITERATIONS = 10     # number of learning updates
UPDATE_FREQUENCY = 20       # every n time step do update
MAX_T = 1000
CRITIC_WEIGHT_DECAY = 1e-2
ACTOR_WEIGHT_DECAY = 1e-4

POLICY_UPDATE_FREQUENCY = 2
WARMUP_STEPS = int(1e4)
MIN_PRIORITY = 1e-3

LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-4  # learning rate of the critic
BATCHNORM = False
DROPOUT = None

SOLVE_SCORE = 1600
SAVE_TAG = 'per_td3_baseline'
ACTOR_CHECKPOINT_PATH = os.path.join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_actor_checkpoint.pth')
CRITIC_CHECKPOINT_PATH = os.path.join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_critic_checkpoint.pth')
TRAINING_SCORES_SAVE_PATH = os.path.join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_training_scores.pkl')


def output_layer_initialization(*args):
    """  We need the initialization for last layer of the Actor to be between -0.003 and 0.003 as this prevents us
     from getting 1 or -1 output values in the initial stages, which would squash our gradients to zero,
    as we use the tanh activation.
    """
    return -3e-3, 3e-3


def hidden_init(layer):
    """ Initialize hidden layers """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


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

    def set_seed(self, seed):
        torch.manual_seed(seed)

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

def get_solution_agent(memory_, seed):
    return DDPGAgent(
        state_shape=STATE_SIZE,
        action_size=ACTION_SIZE,
        random_seed=seed,
        memory_factory=lambda: memory_,
        # actor_model_factory= lambda: Actor(STATE_SIZE, ACTION_SIZE, SEED),
        actor_model_factory=lambda: MLP(
            layer_sizes=(STATE_SIZE, 128, 128, ACTION_SIZE),
            seed=seed, with_batchnorm=BATCHNORM, dropout=DROPOUT,
            output_function=torch.nn.Tanh(), output_layer_initialization_fn=init_layer_within_range,
            activation_function=torch.nn.LeakyReLU(True)
        ),
        critic_model_factory=lambda: TD3Critic(
            # critic_model_factory=lambda: Critic(STATE_SIZE, ACTION_SIZE, SEED),
            critic_model_factory=lambda: Critic(
                state_featurizer=MLP(
                    layer_sizes=(STATE_SIZE, 128), dropout=DROPOUT, with_batchnorm=BATCHNORM,
                    hidden_layer_initialization_fn=init_layer_inverse_root_fan_in,
                    activation_function=torch.nn.LeakyReLU()),
                output_module=MLP(
                    layer_sizes=(128 + ACTION_SIZE, 1), dropout=DROPOUT, with_batchnorm=BATCHNORM,
                    hidden_layer_initialization_fn=init_layer_inverse_root_fan_in,
                    activation_function=torch.nn.LeakyReLU()),
                seed=seed,
            ),
            seed=seed
        ),
        actor_optimizer_factory=lambda params: torch.optim.AdamW(params, lr=LR_ACTOR, weight_decay=ACTOR_WEIGHT_DECAY),
        critic_optimizer_factory=lambda params: torch.optim.AdamW(params, lr=LR_CRITIC, weight_decay=CRITIC_WEIGHT_DECAY),
        critic_optimizer_scheduler=lambda x: DummyLRScheduler(x),
        actor_optimizer_scheduler=lambda x: DummyLRScheduler(x),
        policy_factory=lambda: TD3Policy(
            action_dim=ACTION_SIZE,
            noise=None,
            seed=seed,
            random_brain_action_factory=lambda: RandomBrainAction(
                ACTION_SIZE,
                1,
                continuous_actions=True,
                continuous_action_range=(-1, 1),
            ),
            epsilon_scheduler=ParameterScheduler(initial=1, final=0, lambda_fn=lambda episode: 1 - episode/NUM_EPISODES)
        ),
        update_frequency=UPDATE_FREQUENCY,
        n_learning_iterations=N_LEARNING_ITERATIONS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        policy_update_frequency=POLICY_UPDATE_FREQUENCY,
        shared_agent_brain=True,
        td3=True
    )


def get_solution_brain_set():
    memory = Memory(buffer_size=REPLAY_BUFFER_SIZE, seed=SEED)

    crawler_brain = Brain(
        brain_name=BRAIN_NAME,
        action_size=ACTION_SIZE,
        state_shape=STATE_SIZE,
        observation_type='vector',
        agents=[get_solution_agent(memory, seed) for seed in range(NUM_AGENTS)],
    )

    brain_set = BrainSet(brains=[crawler_brain])

    return brain_set


if __name__ == "__main__":
    simulator = get_simulator()

    brain_set = get_solution_brain_set()

    simulator.warmup(
        brain_set,
        n_episodes=int(WARMUP_STEPS / MAX_T),
        max_t=MAX_T
    )
    agents, training_scores, i_episode, training_time = simulator.train(
        brain_set,
        n_episodes=NUM_EPISODES,
        max_t=MAX_T,
        solved_score=SOLVE_SCORE
    )

    if training_scores.get_mean_sliding_scores() > SOLVE_SCORE:
        brain = brain_set[BRAIN_NAME]
        trained_agent = brain.agents[0]
        torch.save(trained_agent.online_actor.state_dict(), ACTOR_CHECKPOINT_PATH)
        torch.save(trained_agent.online_critic.state_dict(), CRITIC_CHECKPOINT_PATH)
        with open(TRAINING_SCORES_SAVE_PATH, 'wb') as f:
            pickle.dump(training_scores, f)
