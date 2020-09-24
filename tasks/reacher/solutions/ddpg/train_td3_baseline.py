import os
import torch
from agents.ddpg_agent import DDPGAgent
from agents.policies.td3_policy import TD3Policy
from agents.models.components.noise import GaussianProcess
from agents.memory.prioritized_memory import PrioritizedMemory
from agents.models.components.mlp import MLP
from agents.models.td3 import TD3Critic
from tasks.reacher.solutions.utils import get_simulator, STATE_SIZE, ACTION_SIZE, BRAIN_NAME
from tasks.reacher.solutions.ddpg import SOLUTIONS_CHECKPOINT_DIR
from agents.models.components.critics import Critic
from tools.lr_schedulers import DummyLRScheduler
from tools.parameter_scheduler import ParameterScheduler
import pickle
from tools.rl_constants import BrainSet, Brain
from tools.rl_constants import RandomBrainAction
from tools.parameter_scheduler import LinearDecaySchedule
from tools.layer_initializations import init_layer_within_range, init_layer_inverse_root_fan_in
from torch import nn
import numpy as np
import torch.nn.functional as F

NUM_AGENTS = 20
NUM_EPISODES = 200
SEED = 0
BATCH_SIZE = 512
REPLAY_BUFFER_SIZE = int(1e6)
GAMMA = 0.99            # discount factor
TAU = 5e-3              # for soft update of target parameters
N_LEARNING_ITERATIONS = 10     # number of learning updates
UPDATE_FREQUENCY = 20       # every n time step do update
MAX_T = 1000
CRITIC_WEIGHT_DECAY = 0.0  # 1e-2
ACTOR_WEIGHT_DECAY = 0.0
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-4  # learning rate of the critic
POLICY_UPDATE_FREQUENCY = 2
WARMUP_STEPS = int(5e3)
MIN_PRIORITY = 1e-3

DROPOUT = None
BATCHNORM = False

SOLVE_SCORE = 30
SAVE_TAG = 'per_td3'
ACTOR_CHECKPOINT_PATH = os.path.join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_actor_checkpoint.pth')
CRITIC_CHECKPOINT_PATH = os.path.join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_critic_checkpoint.pth')
TRAINING_SCORES_PLOT_SAVE_PATH = os.path.join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_training_scores.png')
TRAINING_SCORES_SAVE_PATH = os.path.join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_training_scores.pkl')


# def output_layer_initialization(*args):
#     """  We need the initialization for last layer of the Actor to be between -0.003 and 0.003 as this prevents us
#      from getting 1 or -1 output values in the initial stages, which would squash our gradients to zero,
#     as we use the tanh activation.
#     """
#     return -3e-3, 3e-3
#
#
# def hidden_init(layer):
#     """ Initialize hidden layers """
#     fan_in = layer.weight.data.size()[0]
#     lim = 1. / np.sqrt(fan_in)
#     return -lim, lim
#
#
# class Actor(nn.Module):
#     """Actor (Policy) Model."""
#
#     def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128, activation=F.leaky_relu):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#             fc1_units (int): Number of nodes in first hidden layer
#             fc2_units (int): Number of nodes in second hidden layer
#         """
#         super().__init__()
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(state_size, fc1_units)
#         self.fc2 = nn.Linear(fc1_units, fc2_units)
#         self.fc3 = nn.Linear(fc2_units, action_size)
#         self.reset_parameters()
#         self.activation = activation
#
#     def reset_parameters(self):
#         self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
#         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
#         self.fc3.weight.data.uniform_(*output_layer_initialization())
#
#     def forward(self, state: torch.Tensor):
#         """Build an actor (policy) network that maps states -> actions."""
#         x = self.activation(self.fc1(state))
#         x = self.activation(self.fc2(x))
#         return torch.tanh(self.fc3(x))
#
# class Critic(nn.Module):
#     """Critic (Value) Model."""
#
#     def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128, activation=F.leaky_relu):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#             fc1_units (int): Number of nodes in the first hidden layer
#             fc2_units (int): Number of nodes in the second hidden layer
#         """
#         super().__init__()
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(state_size, fc1_units)
#         self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
#         self.fc3 = nn.Linear(fc2_units, 1)
#         self.reset_parameters()
#         self.activation = activation
#
#     def set_seed(self, seed):
#         torch.manual_seed(seed)
#
#     def reset_parameters(self):
#         self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
#         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
#         self.fc3.weight.data.uniform_(*output_layer_initialization())
#
#     def forward(self, state, action):
#         """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
#         xs = self.activation(self.fc1(state))
#         x = torch.cat((xs, action), dim=1)
#         x = self.activation(self.fc2(x))
#         return self.fc3(x)


def get_agent(memory_):
    return DDPGAgent(
        state_shape=STATE_SIZE,
        action_size=ACTION_SIZE,
        random_seed=SEED,
        memory_factory=lambda: memory_,
        actor_model_factory=lambda: MLP(
            layer_sizes=(STATE_SIZE, 256, 128, ACTION_SIZE),
            seed=SEED, with_batchnorm=BATCHNORM, dropout=DROPOUT,
            output_function=torch.nn.Tanh(),
            output_layer_initialization_fn=init_layer_within_range,
            activation_function=torch.nn.LeakyReLU()
        ),
        # actor_model_factory=lambda: Actor(STATE_SIZE, ACTION_SIZE, SEED),
        critic_model_factory=lambda: TD3Critic(
            # critic_model_factory=lambda: Critic(STATE_SIZE, ACTION_SIZE, SEED),
            critic_model_factory=lambda: Critic(
                state_featurizer=MLP(
                    layer_sizes=(STATE_SIZE, 256),
                    dropout=DROPOUT,
                    with_batchnorm=BATCHNORM,
                    activation_function=torch.nn.LeakyReLU(),
                    output_function=torch.nn.LeakyReLU(),
                ),
                output_module=MLP(
                    layer_sizes=(256 + ACTION_SIZE, 128, 1),
                    dropout=DROPOUT,
                    with_batchnorm=BATCHNORM,
                    activation_function=torch.nn.LeakyReLU(),
                    output_layer_initialization_fn=init_layer_within_range,
                ),
                seed=SEED,
            ),
            seed=SEED
        ),
        actor_optimizer_factory=lambda params: torch.optim.Adam(params, lr=LR_ACTOR, weight_decay=ACTOR_WEIGHT_DECAY),
        critic_optimizer_factory=lambda params: torch.optim.Adam(params, lr=LR_CRITIC, weight_decay=CRITIC_WEIGHT_DECAY),
        critic_optimizer_scheduler=lambda x: DummyLRScheduler(x),
        actor_optimizer_scheduler=lambda x: DummyLRScheduler(x),
        policy_factory=lambda: TD3Policy(
            action_dim=ACTION_SIZE,
            noise=GaussianProcess(std_fn=LinearDecaySchedule(start=0.3, end=0, steps=NUM_EPISODES)),
            seed=SEED,
            random_brain_action_factory=lambda: RandomBrainAction(
                ACTION_SIZE,
                1,
                continuous_actions=True,
                continuous_action_range=(-1, 1),
            )
        ),
        update_frequency=UPDATE_FREQUENCY,
        n_learning_iterations=N_LEARNING_ITERATIONS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        policy_update_frequency=POLICY_UPDATE_FREQUENCY,
        shared_agent_brain=True
    )


if __name__ == "__main__":
    simulator = get_simulator()

    memory = PrioritizedMemory(
        capacity=REPLAY_BUFFER_SIZE,
        state_shape=(1, STATE_SIZE),
        # Anneal alpha linearly
        alpha_scheduler=ParameterScheduler(initial=0.6, lambda_fn=lambda i: 0.6 - 0.6 * i / NUM_EPISODES, final=0.),
        # Anneal beta linearly
        beta_scheduler=ParameterScheduler(initial=0.4, final=1,
                                          lambda_fn=lambda i: 0.4 + 0.6 * i / NUM_EPISODES),  # Anneal beta linearly
        seed=SEED,
        continuous_actions=True,
        min_priority=MIN_PRIORITY
    )

    reacher_brain = Brain(
        brain_name=BRAIN_NAME,
        action_size=ACTION_SIZE,
        state_shape=STATE_SIZE,
        observation_type='vector',
        agents=[get_agent(memory) for _ in range(NUM_AGENTS)],
    )

    brain_set = BrainSet(brains=[reacher_brain])

    simulator.warmup(brain_set, n_episodes=int(WARMUP_STEPS / MAX_T), max_t=MAX_T)
    agents, training_scores, i_episode, training_time = simulator.train(brain_set, n_episodes=NUM_EPISODES, max_t=MAX_T, solved_score=SOLVE_SCORE)

    if training_scores.get_mean_sliding_scores() > SOLVE_SCORE:
        brain = brain_set[BRAIN_NAME]
        trained_agent = brain.agents[0]
        torch.save(trained_agent.online_actor.state_dict(), ACTOR_CHECKPOINT_PATH)
        torch.save(trained_agent.online_critic.state_dict(), CRITIC_CHECKPOINT_PATH)

        training_scores.save_scores_plot(TRAINING_SCORES_PLOT_SAVE_PATH)
        with open(TRAINING_SCORES_SAVE_PATH, 'wb') as f:
            pickle.dump(training_scores, f)
