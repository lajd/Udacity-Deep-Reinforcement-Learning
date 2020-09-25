import os
import torch
import numpy as np
from tasks.soccer.solutions.utils import get_simulator
from tasks.soccer.solutions.mappo.train_mappo import get_solution_brain_set, MAX_T,\
    SOLUTIONS_CHECKPOINT_DIR, end_of_episode_score_display_fn, episode_reward_fn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SAVED_AGENT_GOALIE_FP = os.path.join(SOLUTIONS_CHECKPOINT_DIR, 'GoalieBrain_agent_0_mappo_100_consecutive_wins_actor_critic_checkpoint.pth')
SAVED_AGENT_STRIKER_FP = os.path.join(SOLUTIONS_CHECKPOINT_DIR, 'StrikerBrain_agent_0_mappo_100_consecutive_wins_actor_critic_checkpoint.pth')


if __name__ == '__main__':

    simulator = get_simulator()

    brain_set = get_solution_brain_set()

    # Load the agents
    brain_set['GoalieBrain'].agents[0].target_actor_critic.load_state_dict(torch.load(SAVED_AGENT_GOALIE_FP))
    brain_set['StrikerBrain'].agents[0].target_actor_critic.load_state_dict(torch.load(SAVED_AGENT_STRIKER_FP))

    brain_set, average_score = simulator.evaluate(
        brain_set,
        n_episodes=20,
        max_t=MAX_T,
        end_episode_criteria=np.all,
        end_of_episode_score_display_fn=end_of_episode_score_display_fn,
        episode_reward_accumulation_fn=lambda brain_episode_scores: episode_reward_fn(brain_episode_scores),
    )
