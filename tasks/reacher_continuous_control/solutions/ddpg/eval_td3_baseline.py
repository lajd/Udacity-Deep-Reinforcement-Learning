import os
import torch
from agents.ddpg_agent import DDPGAgent
from agents.policies.td3_policy import TD3Policy
from agents.models.components.noise import GaussianNoise
from agents.memory.memory import Memory
from agents.models.ddpg import TD3Actor, TD3Critic
from tasks.reacher_continuous_control.solutions.utils import get_simulator, STATE_SIZE, ACTION_SIZE
from tools.lr_schedulers import DummyLRScheduler
from tasks.reacher_continuous_control.solutions.ddpg.train_td3_baseline import SAVE_TAG
from tasks.reacher_continuous_control.solutions.ddpg import SOLUTIONS_CHECKPOINT_DIR
from tools.parameter_decay import ParameterScheduler

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
LR_ACTOR = 3e-4  # learning rate of the actor
LR_CRITIC = 3e-4  # learning rate of the critic
POLICY_UPDATE_FREQUENCY = 2
WARMUP_STEPS = int(5e3)
MIN_PRIORITY = 1e-3

ACTOR_CHECKPOINT = os.path.join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_actor_checkpoint.pth')
CRITIC_CHECKPOINT = os.path.join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_critic_checkpoint.pth')


def get_solution_agent(actor_network: torch.nn.Module, critic_network: torch.nn.Module):
    return DDPGAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        random_seed=SEED,
        memory_factory=lambda: memory,
        actor_model_factory=lambda: actor_network,
        critic_model_factory=lambda: critic_network,
        actor_optimizer_factory=lambda actor: torch.optim.Adam(actor.parameters(), lr=LR_ACTOR, weight_decay=ACTOR_WEIGHT_DECAY),
        critic_optimizer_factory=lambda critic: torch.optim.Adam(critic.parameters(), lr=LR_CRITIC, weight_decay=CRITIC_WEIGHT_DECAY),
        critic_optimizer_scheduler=lambda x: DummyLRScheduler(x),
        actor_optimizer_scheduler=lambda x: DummyLRScheduler(x),
        policy_factory=lambda: TD3Policy(action_dim=ACTION_SIZE, noise=GaussianNoise(), seed=SEED),
        update_frequency=UPDATE_FREQUENCY,
        n_learning_iterations=N_LEARNING_ITERATIONS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        policy_update_frequency=POLICY_UPDATE_FREQUENCY,
    )


if __name__ == '__main__':
    actor = TD3Actor(STATE_SIZE, ACTION_SIZE, seed=SEED)
    actor.load_state_dict(torch.load(ACTOR_CHECKPOINT))
    critic = TD3Critic(STATE_SIZE, ACTION_SIZE, seed=SEED)
    critic.load_state_dict(torch.load(CRITIC_CHECKPOINT))

    memory = Memory(buffer_size=REPLAY_BUFFER_SIZE, seed=SEED)

    simulator = get_simulator()

    agents = [get_solution_agent(actor, critic) for _ in range(NUM_AGENTS)]

    agents, average_score = simulator.evaluate(agents, n_episodes=1, max_t=MAX_T)
