import os
import pickle

from agents.models.components.mlp import MLP
from agents.models.ppo import PPO_Actor_Critic
from agents.ppo_agent import PPOAgent
from tasks.crawler.solutions.ppo import SOLUTIONS_CHECKPOINT_DIR
from tasks.crawler.solutions.utils import get_simulator, STATE_SIZE, ACTION_SIZE, BRAIN_NAME
from tools.rl_constants import BrainSet, Brain
from tools.rl_constants import Experience
import torch
from tools.layer_initializations import init_layer_within_range, init_layer_inverse_root_fan_in
from simulation.utils import single_agent_step_agents_fn


NUM_EPISODES = 3000
SEED = 8
MAX_T = 2000
WEIGHT_DECAY = 1e-4
EPSILON = 1e-5  # epsilon of Adam
LR = 1e-4  # learning rate of the actor-critic
BATCH_SIZE = 1024
DROPOUT = None
BATCHNORM = True
SOLVE_SCORE = 1600
SAVE_TAG = 'ppo'
ACTOR_CRITIC_CHECKPOINT_PATH = os.path.join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_actor_checkpoint.pth')
TRAINING_SCORES_PLOT_SAVE_PATH = os.path.join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_training_scores.png')
TRAINING_SCORES_SAVE_PATH = os.path.join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_training_scores.pkl')


def get_solution_brain_set():
    agent = PPOAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        seed=SEED,
        actor_critic_factory=lambda: PPO_Actor_Critic(
            actor_model=MLP(
                layer_sizes=(STATE_SIZE, 128, 128, ACTION_SIZE),
                seed=SEED,
                output_function=torch.nn.Tanh(),
                with_batchnorm=BATCHNORM,
                output_layer_initialization_fn=lambda l: init_layer_within_range(l),
                hidden_layer_initialization_fn=lambda l: init_layer_inverse_root_fan_in(l),
                activation_function=torch.nn.LeakyReLU(True),
                dropout=DROPOUT
            ),
            critic_model=MLP(
                layer_sizes=(STATE_SIZE, 128, 128, 1),
                seed=SEED,
                output_function=torch.nn.Tanh(),
                with_batchnorm=BATCHNORM,
                output_layer_initialization_fn=lambda l: init_layer_within_range(l),
                hidden_layer_initialization_fn=lambda l: init_layer_inverse_root_fan_in(l),
                activation_function=torch.nn.LeakyReLU(True),
                dropout=DROPOUT
            ),
            action_size=ACTION_SIZE,
            continuous_actions=True,
        ),
        optimizer_factory=lambda params: torch.optim.Adam(
            params, lr=LR, weight_decay=WEIGHT_DECAY, eps=EPSILON
        ),
        batch_size=BATCH_SIZE,
    )

    crawler_brain = Brain(
        brain_name=BRAIN_NAME,
        action_size=ACTION_SIZE,
        state_shape=STATE_SIZE,
        observation_type='vector',
        agents=[agent],
    )
    brain_set = BrainSet(brains=[crawler_brain])
    return brain_set


if __name__ == "__main__":
    simulator = get_simulator()

    brain_set = get_solution_brain_set()

    agents, training_scores, i_episode, training_time = simulator.train(
        brain_set,
        n_episodes=NUM_EPISODES,
        max_t=MAX_T,
        solved_score=SOLVE_SCORE,
        step_agents_fn=single_agent_step_agents_fn,
    )

    if training_scores.get_mean_sliding_scores() >= SOLVE_SCORE:
        for brain_name, brain in brain_set:
            trained_agent = brain.agents[0]
            # Only AI agent
            torch.save(trained_agent.online_actor_critic.state_dict(), ACTOR_CRITIC_CHECKPOINT_PATH)
        training_scores.save_scores_plot(TRAINING_SCORES_PLOT_SAVE_PATH)
        with open(TRAINING_SCORES_SAVE_PATH, 'wb') as f:
            pickle.dump(training_scores, f)
