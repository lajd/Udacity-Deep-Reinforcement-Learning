import os
import torch
from agents.ddpg_agent import DDPGAgent
from agents.policies.ddpg_policy import DDPGPolicy
from agents.models.components.noise import OUNoise, GaussianProcess
from agents.memory.memory import Memory
from tasks.reacher.solutions.utils import get_simulator, STATE_SIZE, ACTION_SIZE, BRAIN_NAME
from tasks.reacher.solutions.ddpg import SOLUTIONS_CHECKPOINT_DIR
from tools.lr_schedulers import DummyLRScheduler
import pickle
from tools.rl_constants import BrainSet, Brain, ExperienceBatch, RandomBrainAction
from agents.models.components.mlp import MLP
from agents.models.components.critics import Critic
from tools.layer_initializations import init_layer_within_range, init_layer_inverse_root_fan_in
from tools.parameter_scheduler import LinearDecaySchedule

NUM_AGENTS = 20
NUM_EPISODES = 200
SEED = 0
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = int(1e6)
GAMMA = 0.99            # discount factor
TAU = 5e-3              # for soft update of target parameters
N_LEARNING_ITERATIONS = 10     # number of learning updates
UPDATE_FREQUENCY = 20       # every n time step do update
MAX_T = 1000
CRITIC_WEIGHT_DECAY = 0.0  # 1e-2
ACTOR_WEIGHT_DECAY = 0.0
MIN_PRIORITY = 1e-4
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-4  # learning rate of the critic
POLICY_UPDATE_FREQUENCY = 1
DROPOUT = None
BATCHNORM = False

SOLVE_SCORE = 1
SAVE_TAG = 'ddpg'
ACTOR_CHECKPOINT = os.path.join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_actor_checkpoint.pth')
CRITIC_CHECKPOINT = os.path.join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_critic_checkpoint.pth')
TRAINING_SCORES_PLOT_SAVE_PATH = os.path.join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_training_scores.png')
TRAINING_SCORES_SAVE_PATH = os.path.join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_training_scores.pkl')


def get_agent(memory_):
    """ Return an agent with shared memory, actor, critic and optimizers"""
    return DDPGAgent(
        state_shape=STATE_SIZE,
        action_size=ACTION_SIZE,
        random_seed=SEED,
        memory_factory=lambda: memory_,
        actor_model_factory=lambda: MLP(
            layer_sizes=(STATE_SIZE, 256, 128, ACTION_SIZE),
            seed=SEED,
            output_function=torch.nn.Tanh(),
            with_batchnorm=True,
            output_layer_initialization_fn=lambda l: init_layer_within_range(l),
            hidden_layer_initialization_fn=lambda l: init_layer_inverse_root_fan_in(l),
            activation_function=torch.nn.LeakyReLU(True),
            dropout=None
        ),
        critic_model_factory=lambda: Critic(
            state_featurizer=MLP(
                layer_sizes=(STATE_SIZE, 128),
                dropout=DROPOUT,
                with_batchnorm=BATCHNORM,
                activation_function=torch.nn.LeakyReLU(True)
            ),
            output_module=MLP(
                layer_sizes=(128 + ACTION_SIZE, 1),
                dropout=DROPOUT,
                with_batchnorm=BATCHNORM,
                activation_function=torch.nn.LeakyReLU(True)
            ),
            seed=SEED,
        ),
        actor_optimizer_factory=lambda params: torch.optim.Adam(params, lr=LR_ACTOR, weight_decay=ACTOR_WEIGHT_DECAY),
        critic_optimizer_factory=lambda params: torch.optim.Adam(params, lr=LR_CRITIC, weight_decay=CRITIC_WEIGHT_DECAY),
        critic_optimizer_scheduler=lambda x: DummyLRScheduler(x),
        actor_optimizer_scheduler=lambda x: DummyLRScheduler(x),
        policy_factory=lambda: DDPGPolicy(
            action_dim=ACTION_SIZE,
            noise=GaussianProcess(std_fn=LinearDecaySchedule(start=1, end=0, steps=NUM_EPISODES)),
            seed=SEED,
            random_brain_action_factory=lambda: RandomBrainAction(
                action_dim=ACTION_SIZE,
                num_agents=NUM_AGENTS,
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


if __name__ == '__main__':
    simulator = get_simulator()

    # Standard replay memory
    memory = Memory(buffer_size=REPLAY_BUFFER_SIZE, seed=SEED)

    reacher_brain = Brain(
        brain_name=BRAIN_NAME,
        action_size=ACTION_SIZE,
        state_shape=STATE_SIZE,
        observation_type='vector',
        agents=[get_agent(memory) for _ in range(NUM_AGENTS)],
    )

    brain_set = BrainSet(brains=[reacher_brain])

    brain_set, training_scores, i_episode, training_time = simulator.train(
        brain_set,
        n_episodes=NUM_EPISODES,
        max_t=MAX_T,
        solved_score=SOLVE_SCORE
    )

    if training_scores.get_mean_sliding_scores() > SOLVE_SCORE:
        brain = brain_set[BRAIN_NAME]
        trained_agent = brain.agents[0]
        torch.save(trained_agent.online_actor.state_dict(), ACTOR_CHECKPOINT)
        torch.save(trained_agent.online_critic.state_dict(), CRITIC_CHECKPOINT)

        training_scores.save_scores_plot(TRAINING_SCORES_PLOT_SAVE_PATH)
        with open(TRAINING_SCORES_SAVE_PATH, 'wb') as f:
            pickle.dump(training_scores, f)
