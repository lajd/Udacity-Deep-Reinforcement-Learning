import os
import torch
import numpy as np
from tasks.crawler.solutions.utils import get_simulator
from tasks.crawler.solutions.ppo.train_ppo import get_solution_brain_set, MAX_T, SOLUTIONS_CHECKPOINT_DIR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SAVED_AGENT_FP = os.path.join(SOLUTIONS_CHECKPOINT_DIR, 'ppo_actor_checkpoint.pth')


if __name__ == '__main__':

    simulator = get_simulator()

    brain_set = get_solution_brain_set()

    # Load the agents
    brain_set['CrawlerBrain'].agents[0].target_actor_critic.load_state_dict(torch.load(SAVED_AGENT_FP))

    brain_set, average_score = simulator.evaluate(
        brain_set,
        n_episodes=10,
        max_t=MAX_T,
        end_episode_criteria=np.all
    )
