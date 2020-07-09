import numpy as np
# from tasks.reacher_continuous_control.solution_checkpoint.utils import get_simulator, STATE_SIZE, ACTION_SIZE
# %matplotlib inline
import torch
from tools.rl_constants import Experience
from tasks.tennis.solutions.utils import STATE_SIZE, ACTION_SIZE, NUM_AGENTS, get_simulator
from agents.maddpg_agent import HomogeneousMADDPGAgent as MA


NUM_EPISODES = 1000
MAX_T = 1000
SOLVE_SCORE = 2


if __name__ == '__main__':
    simulator = get_simulator()

    agents_ = [MA(STATE_SIZE, ACTION_SIZE, fc1=400, fc2=300, seed=0, update_times=10) for _ in range(NUM_AGENTS)]

    def get_experience(states: np.ndarray, actions_list: list, rewards: np.ndarray,
                               next_states: np.ndarray, dones: np.ndarray, time_step: int, num_agents: int):
        """ Prepare experiences for the agents """

        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)
        experiences = []
        for i in range(num_agents):
            all_states = torch.cat((states[i], states[1 - i]))
            all_actions = np.concatenate((actions_list[i].value, actions_list[1 - i].value))
            all_next_state = np.concatenate((next_states[i], next_states[1 - i]))

            experiences.append(Experience(
                state=states[i], joint_state=all_states, action=actions_list[i].value, joint_action=all_actions, reward=rewards[i],
                next_state=next_states[i], joint_next_state=all_next_state, done=dones[i], t_step=time_step
            ))

        return experiences

    agents, training_scores, i_episode, training_time = simulator.train(
        agents_, n_episodes=NUM_EPISODES, max_t=MAX_T,
        solved_score=SOLVE_SCORE, get_experiences_fn=get_experience
    )

    # if training_scores.get_mean_sliding_scores() > SOLVE_SCORE:
    #     torch.save(DDPGAgent.online_actor.state_dict(), ACTOR_CHECKPOINT)
    #     torch.save(DDPGAgent.online_critic.state_dict(), CRITIC_CHECKPOINT)
    #     with open(TRAINING_SCORES_SAVE_PATH, 'wb') as f:
    #         pickle.dump(training_scores, f)
