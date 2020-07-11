import os
import pickle
import numpy as np
import torch
from tools.rl_constants import Experience
from tasks.tennis.solutions.utils import STATE_SIZE, ACTION_SIZE, NUM_AGENTS, get_simulator
from agents.maddpg_agent import HomogeneousMADDPGAgent
from tasks.tennis.solutions.maddpg import SOLUTIONS_CHECKPOINT_DIR
from agents.policies.maddpg_policy import MADDPGPolicy
from tools.misc import LinearSchedule
from agents.models.components import noise as rm

SAVE_TAG = 'multi_homogeneous_maddpg_baseline'
ACTOR_CHECKPOINT = os.path.join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_actor_checkpoint.pth')
CRITIC_CHECKPOINT = os.path.join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_critic_checkpoint.pth')
TRAINING_SCORES_SAVE_PATH = os.path.join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_training_scores.pkl')


NUM_EPISODES = 1000
MAX_T = 1000
SOLVE_SCORE = 2


def step_agents_fn(states: np.ndarray, actions_list: list, rewards: np.ndarray,
                   next_states: np.ndarray, dones: np.ndarray, time_step: int, agents):
    """ Prepare experiences for the agents """

    if isinstance(states, np.ndarray):
        states = torch.from_numpy(states)

    for i in range(NUM_AGENTS):
        all_states = torch.cat((states[i], states[1 - i]))
        all_actions = np.concatenate((actions_list[0].value[i], actions_list[0].value[1 - i]))
        all_next_state = np.concatenate((next_states[i], next_states[1 - i]))

        experience = Experience(
            state=states[i], joint_state=all_states, action=actions_list[0].value[i], joint_action=all_actions,
            reward=rewards[i],
            next_state=next_states[i], joint_next_state=all_next_state, done=dones[i], t_step=time_step
        )

        agents[0].step(experience)


if __name__ == '__main__':
    simulator = get_simulator()

    noise_factory = lambda: rm.OrnsteinUhlenbeckProcess(size=(ACTION_SIZE,), std=LinearSchedule(0.4, 0, 2000))


    policy = MADDPGPolicy(
        noise_factory=noise_factory,
        action_dim=ACTION_SIZE,
        num_agents=NUM_AGENTS
    )

    homogeneous_maddpg_agent = HomogeneousMADDPGAgent(
        policy, STATE_SIZE, ACTION_SIZE, num_homogeneous_agents=NUM_AGENTS,
        fc1=400, fc2=300, seed=0,
    )

    agents, training_scores, i_episode, training_time = simulator.train(
        [homogeneous_maddpg_agent], n_episodes=NUM_EPISODES, max_t=MAX_T,
        solved_score=SOLVE_SCORE, step_agents_fn=step_agents_fn,
        reward_accumulation_fn=lambda rewards: np.max(rewards),
        preprocess_actions_fn=lambda actions: actions.reshape(1, -1),
        get_actions_list_fn=lambda agents, states: [agents[0].get_action(states)]
    )
    
    if training_scores.get_mean_sliding_scores() > SOLVE_SCORE:
        trained_agent = agents[0]
        torch.save(trained_agent.online_actor.state_dict(), ACTOR_CHECKPOINT)
        torch.save(trained_agent.online_critic.state_dict(), CRITIC_CHECKPOINT)
        with open(TRAINING_SCORES_SAVE_PATH, 'wb') as f:
            pickle.dump(training_scores, f)
