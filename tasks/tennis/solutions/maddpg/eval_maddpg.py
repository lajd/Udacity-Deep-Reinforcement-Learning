import os
import torch
import numpy as np
from tasks.tennis.solutions.utils import get_simulator
from tasks.tennis.solutions.maddpg.train_maddpg import get_solution_brain_set, MAX_T, SOLUTIONS_CHECKPOINT_DIR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SAVED_AGENT_0_ACTOR_FP = os.path.join(SOLUTIONS_CHECKPOINT_DIR, 'TennisBrain_agent_0_independent_madtd3_actor_checkpoint.pth')
SAVED_AGENT_1_ACTOR_FP = os.path.join(SOLUTIONS_CHECKPOINT_DIR, 'TennisBrain_agent_1_independent_madtd3_actor_checkpoint.pth')

if __name__ == '__main__':

    simulator = get_simulator()

    brain_set = get_solution_brain_set()

    # Load the agent actors
    brain_set['TennisBrain'].agents[0].online_actor.load_state_dict(torch.load(SAVED_AGENT_0_ACTOR_FP))
    brain_set['TennisBrain'].agents[1].online_actor.load_state_dict(torch.load(SAVED_AGENT_1_ACTOR_FP))

    brain_set, average_score = simulator.evaluate(
        brain_set,
        n_episodes=10,
        max_t=MAX_T,
        brain_reward_accumulation_fn=lambda rewards: np.max(rewards),
        end_episode_critieria=np.all
    )
