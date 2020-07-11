import os
import pickle
import numpy as np
import torch
from tools.rl_constants import Experience, Brain, BrainSet
from tasks.tennis.solutions.utils import STATE_SIZE, ACTION_SIZE, NUM_AGENTS, BRAIN_NAME, get_simulator
from agents.maddpg_agent import HomogeneousMADDPGAgent
from tasks.tennis.solutions.maddpg import SOLUTIONS_CHECKPOINT_DIR
from agents.policies.maddpg_policy import MADDPGPolicy
from tools.misc import LinearSchedule
from agents.models.components import noise as rm

SAVE_TAG = 'homogeneous_maddpg_baseline'
ACTOR_CHECKPOINT = os.path.join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_actor_checkpoint.pth')
CRITIC_CHECKPOINT = os.path.join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_critic_checkpoint.pth')
TRAINING_SCORES_SAVE_PATH = os.path.join(SOLUTIONS_CHECKPOINT_DIR, f'{SAVE_TAG}_training_scores.pkl')


NUM_EPISODES = 1000
MAX_T = 1000
SOLVE_SCORE = 2


# def step_agents_fn(states: np.ndarray, actions_list: list, rewards: np.ndarray,
#                    next_states: np.ndarray, dones: np.ndarray, time_step: int, agents):
#     """ Prepare experiences for the agents """
#
#     if isinstance(states, np.ndarray):
#         states = torch.from_numpy(states)
#
#     for i in range(NUM_AGENTS):
#         all_states = torch.cat((states[i], states[1 - i]))
#         all_actions = np.concatenate((actions_list[0].value[i], actions_list[0].value[1 - i]))
#         all_next_state = np.concatenate((next_states[i], next_states[1 - i]))
#
#         experience = Experience(
#             state=states[i], joint_state=all_states, action=actions_list[0].value[i], joint_action=all_actions,
#             reward=rewards[i],
#             next_state=next_states[i], joint_next_state=all_next_state, done=dones[i], t_step=time_step
#         )
#
#         agents[0].step(experience)

def step_agents_fn(brain_set: BrainSet, next_brain_environment: dict, t: int):
    for brain_name, brain_environment in next_brain_environment.items():
        for i in range(brain_set[brain_name].num_agents):
            joint_state = torch.cat((brain_environment['states'][i], brain_environment['states'][1 - i]))
            joint_action = np.concatenate((brain_environment['actions'][i], brain_environment['actions'][1 - i]))
            join_next_state = np.concatenate((brain_environment['next_states'][i], brain_environment['next_states'][1 - i]))

            brain_agent_experience = Experience(
                state=brain_environment['states'][i],
                action=brain_environment['actions'][i],
                reward=brain_environment['rewards'][i],
                next_state=brain_environment['next_states'][i],
                done=brain_environment['dones'][i],
                t_step=t,
                joint_state=joint_state,
                joint_action=torch.from_numpy(joint_action),
                joint_next_state=torch.from_numpy(join_next_state)
            )
            brain_set[brain_name].agent.step(brain_agent_experience)


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

    tennis_brain = Brain(
        brain_name=BRAIN_NAME,
        action_size=ACTION_SIZE,
        state_shape=STATE_SIZE,
        observation_type='vector',
        agent=homogeneous_maddpg_agent,
        num_agents=NUM_AGENTS
    )

    brain_set = BrainSet(brains=[tennis_brain])

    brain_set, training_scores, i_episode, training_time = simulator.train(
        brain_set,
        n_episodes=NUM_EPISODES,
        max_t=MAX_T,
        solved_score=SOLVE_SCORE,
        step_agents_fn=step_agents_fn,
        reward_accumulation_fn=lambda rewards: np.max(rewards),
        preprocess_actions_fn=lambda actions: actions.reshape(1, -1),
        get_actions_list_fn=lambda agents, states: [agents[0].get_action(states)]
    )

    # if training_scores.get_mean_sliding_scores() > SOLVE_SCORE:
    #     trained_agent = agents[0]
    #     torch.save(trained_agent.online_actor.state_dict(), ACTOR_CHECKPOINT)
    #     torch.save(trained_agent.online_critic.state_dict(), CRITIC_CHECKPOINT)
    #     with open(TRAINING_SCORES_SAVE_PATH, 'wb') as f:
    #         pickle.dump(training_scores, f)
