import os
import torch
from tasks.reacher.solutions.utils import get_simulator, BRAIN_NAME
from tasks.reacher.solutions.ddpg import SOLUTIONS_CHECKPOINT_DIR
from tasks.reacher.solutions.ddpg.train_td3_baseline import get_solution_brain_set, MAX_T

SAVE_TAG = 'per_td3'
ACTOR_CHECKPOINT = os.path.join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_actor_checkpoint.pth')


if __name__ == '__main__':

    brain_set = get_solution_brain_set()

    brain_set[BRAIN_NAME].agents[0].online_actor.load_state_dict(torch.load(ACTOR_CHECKPOINT))

    simulator = get_simulator()

    agents, average_score = simulator.evaluate(brain_set, n_episodes=1, max_t=MAX_T)
