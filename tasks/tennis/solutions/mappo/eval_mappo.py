import os
import torch
import numpy as np
from tasks.tennis.solutions.utils import get_simulator
from tasks.tennis.solutions.mappo.train_mappo import get_solution_brain_set, MAX_T, SOLUTIONS_CHECKPOINT_DIR
from simulation.utils import multi_agent_step_episode_agents_fn, multi_agent_step_agents_fn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SAVED_AGENT_0_FP = os.path.join(SOLUTIONS_CHECKPOINT_DIR, 'TennisBrain_agent_0_mappo_actor_critic_checkpoint.pth')
SAVED_AGENT_1_FP = os.path.join(SOLUTIONS_CHECKPOINT_DIR, 'TennisBrain_agent_1_mappo_actor_critic_checkpoint.pth')


if __name__ == '__main__':

    simulator = get_simulator()

    brain_set = get_solution_brain_set()

    # Load the agents
    brain_set['TennisBrain'].agents[0].target_actor_critic.load_state_dict(torch.load(SAVED_AGENT_0_FP))
    brain_set['TennisBrain'].agents[1].target_actor_critic.load_state_dict(torch.load(SAVED_AGENT_1_FP))

    brain_set, average_score = simulator.evaluate(
        brain_set,
        n_episodes=10,
        max_t=MAX_T,
        brain_reward_accumulation_fn=lambda rewards: np.max(rewards),
        end_episode_critieria=np.all
    )
