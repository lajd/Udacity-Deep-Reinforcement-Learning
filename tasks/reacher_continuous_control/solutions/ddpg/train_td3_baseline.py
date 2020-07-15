import os
import torch
from agents.ddpg_agent import DDPGAgent
from agents.policies.td3_policy import TD3Policy
from agents.models.components.noise import GaussianNoise
from agents.memory.prioritized_memory import PrioritizedMemory
from agents.models.ddpg import TD3Actor, TD3Critic
from tasks.reacher_continuous_control.solutions.utils import get_simulator, STATE_SIZE, ACTION_SIZE, BRAIN_NAME
from tasks.reacher_continuous_control.solutions.ddpg import SOLUTIONS_CHECKPOINT_DIR
from tools.lr_schedulers import DummyLRScheduler
from tools.parameter_decay import ParameterScheduler
import pickle
from tools.rl_constants import BrainSet, Brain

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
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-4  # learning rate of the critic
POLICY_UPDATE_FREQUENCY = 2
WARMUP_STEPS = int(5e3)
MIN_PRIORITY = 1e-3

SOLVE_SCORE = 30
SAVE_TAG = 'per_td3_baseline'
ACTOR_CHECKPOINT_PATH = os.path.join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_actor_checkpoint.pth')
CRITIC_CHECKPOINT_PATH = os.path.join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_critic_checkpoint.pth')
TRAINING_SCORES_SAVE_PATH = os.path.join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_training_scores.pkl')


def get_agent(memory_):
    return DDPGAgent(
        state_shape=STATE_SIZE,
        action_size=ACTION_SIZE,
        random_seed=SEED,
        num_agents=NUM_AGENTS,
        memory_factory=lambda: memory_,
        actor_model_factory=lambda: TD3Actor(STATE_SIZE, ACTION_SIZE, seed=SEED),
        critic_model_factory=lambda: TD3Critic(STATE_SIZE, ACTION_SIZE, seed=SEED),
        actor_optimizer_factory=lambda actor: torch.optim.Adam(actor.parameters(), lr=LR_ACTOR, weight_decay=ACTOR_WEIGHT_DECAY),
        critic_optimizer_factory=lambda critic: torch.optim.Adam(critic.parameters(), lr=LR_CRITIC, weight_decay=CRITIC_WEIGHT_DECAY),
        critic_optimizer_scheduler=lambda x: DummyLRScheduler(x),
        actor_optimizer_scheduler=lambda x: DummyLRScheduler(x),
        policy_factory=lambda: TD3Policy(action_dim=ACTION_SIZE, num_agents=NUM_AGENTS, noise=GaussianNoise(), seed=SEED),
        # policy_factory=lambda: TD3Policy(action_dim=ACTION_SIZE, noise=OUNoise(size=ACTION_SIZE, seed=SEED), seed=SEED),
        update_frequency=UPDATE_FREQUENCY,
        n_learning_iterations=N_LEARNING_ITERATIONS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        policy_update_frequency=POLICY_UPDATE_FREQUENCY,
    )


if __name__ == "__main__":
    simulator = get_simulator()

    memory = PrioritizedMemory(
        capacity=REPLAY_BUFFER_SIZE,
        state_shape=(1, STATE_SIZE),
        # Anneal alpha linearly
        alpha_scheduler=ParameterScheduler(initial=0.6, lambda_fn=lambda i: 0.6 - 0.6 * i / NUM_EPISODES, final=0.),
        # Anneal alpha linearly
        beta_scheduler=ParameterScheduler(initial=0.4, final=1,
                                          lambda_fn=lambda i: 0.4 + 0.6 * i / NUM_EPISODES),  # Anneal beta linearly
        seed=SEED,
        continuous_actions=True,
        min_priority=MIN_PRIORITY
    )
    # memory = Memory(buffer_size=REPLAY_BUFFER_SIZE, seed=SEED)

    agent = get_agent(memory)

    reacher_brain = Brain(
        brain_name=BRAIN_NAME,
        action_size=ACTION_SIZE,
        state_shape=STATE_SIZE,
        observation_type='vector',
        agent=agent,
        num_agents=NUM_AGENTS
    )

    brain_set = BrainSet(brains=[reacher_brain])

    simulator.warmup(brain_set, n_episodes=int(WARMUP_STEPS / MAX_T), max_t=MAX_T)
    agents, training_scores, i_episode, training_time = simulator.train(brain_set, n_episodes=NUM_EPISODES, max_t=MAX_T, solved_score=SOLVE_SCORE)

    if training_scores.get_mean_sliding_scores() > SOLVE_SCORE:
        brain = brain_set[BRAIN_NAME]
        trained_agent = brain.agent
        torch.save(trained_agent.online_actor.state_dict(), ACTOR_CHECKPOINT_PATH)
        torch.save(trained_agent.online_critic.state_dict(), CRITIC_CHECKPOINT_PATH)
        with open(TRAINING_SCORES_SAVE_PATH, 'wb') as f:
            pickle.dump(training_scores, f)
