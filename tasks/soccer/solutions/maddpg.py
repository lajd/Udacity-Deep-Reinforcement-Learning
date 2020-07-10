from unityagents import UnityEnvironment
import numpy as np

env = UnityEnvironment(file_name="/home/jon/PycharmProjects/drl_toolbox/tasks/soccer/environments/Soccer_Linux/Soccer.x86_64")

# print the brain names
print(env.brain_names)

# set the goalie brain
g_brain_name = env.brain_names[0]
g_brain = env.brains[g_brain_name]

# set the striker brain
s_brain_name = env.brain_names[1]
s_brain = env.brains[s_brain_name]


# reset the environment
env_info = env.reset(train_mode=True)

# number of agents
num_g_agents = len(env_info[g_brain_name].agents)
print('Number of goalie agents:', num_g_agents)
num_s_agents = len(env_info[s_brain_name].agents)
print('Number of striker agents:', num_s_agents)

# number of actions
g_action_size = g_brain.vector_action_space_size
print('Number of goalie actions:', g_action_size)
s_action_size = s_brain.vector_action_space_size
print('Number of striker actions:', s_action_size)

# examine the state space
g_states = env_info[g_brain_name].vector_observations
g_state_size = g_states.shape[1]
print('There are {} goalie agents. Each receives a state with length: {}'.format(g_states.shape[0], g_state_size))
s_states = env_info[s_brain_name].vector_observations
s_state_size = s_states.shape[1]
print('There are {} striker agents. Each receives a state with length: {}'.format(s_states.shape[0], s_state_size))

for i in range(2):  # play game for 2 episodes
    env_info = env.reset(train_mode=False)  # reset the environment
    g_states = env_info[g_brain_name].vector_observations  # get initial state (goalies)
    s_states = env_info[s_brain_name].vector_observations  # get initial state (strikers)
    g_scores = np.zeros(num_g_agents)  # initialize the score (goalies)
    s_scores = np.zeros(num_s_agents)  # initialize the score (strikers)
    while True:
        # select actions and send to environment
        g_actions = np.random.randint(g_action_size, size=num_g_agents)
        s_actions = np.random.randint(s_action_size, size=num_s_agents)
        actions = dict(zip([g_brain_name, s_brain_name],
                           [g_actions, s_actions]))
        env_info = env.step(actions)

        # get next states
        g_next_states = env_info[g_brain_name].vector_observations
        s_next_states = env_info[s_brain_name].vector_observations

        # get reward and update scores
        g_rewards = env_info[g_brain_name].rewards
        s_rewards = env_info[s_brain_name].rewards
        g_scores += g_rewards
        s_scores += s_rewards

        # check if episode finished
        done = np.any(env_info[g_brain_name].local_done)

        # roll over states to next time step
        g_states = g_next_states
        s_states = s_next_states

        # exit loop if episode finished
        if done:
            break
    print('Scores from episode {}: {} (goalies), {} (strikers)'.format(i + 1, g_scores, s_scores))

    env.close()