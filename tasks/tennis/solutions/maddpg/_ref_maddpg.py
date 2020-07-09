from unityagents import UnityEnvironment
import torch
import numpy as np
from collections import deque
import os
import matplotlib.pyplot as plt
# %matplotlib inline


from agents._ref_maddpg import HomogeneousMADDPGAgent as MA
# from agents.maddpg_agent import HomogeneousMADDPGAgent as MA
from agents.ref_ddpg_agent import Agent as DA
env = UnityEnvironment(file_name="/home/jon/PycharmProjects/drl_toolbox/tasks/tennis/environments/Tennis_Linux/Tennis.x86_64", no_graphics=False)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

agent = MA(state_size, action_size, num_agents, fc1=400, fc2=300, seed=0, update_times=10)

scores = []


def solve_environment(n_episodes=6000):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    global scores
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        agent.reset_random()  # reset noise object
        state = env_info.vector_observations

        score = 0
        t = 0
        reward_this_episode_1 = 0
        reward_this_episode_2 = 0
        # raise RuntimeError(state.shape)
        while True:
            t = t + 1
            action = agent.act(state)
            env_info = env.step(np.array(action))[brain_name]
            next_state = env_info.vector_observations  # get the next state
            reward = env_info.rewards  # get the reward

            done = env_info.local_done
            # print(state[0])
            agent.step(state, action, reward, next_state, done)
            state = next_state
            # print(reward)
            reward_this_episode_1 += reward[0]
            reward_this_episode_2 += reward[1]

            if np.any(done):
                break

        score = max(reward_this_episode_1, reward_this_episode_2)
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 2:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.critic_local.state_dict(), 'trained_weights/checkpoint_critic.pth')
            torch.save(agent.actor_local.state_dict(), 'trained_weights/checkpoint_actor.pth')
            break
    return


solve_environment()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
