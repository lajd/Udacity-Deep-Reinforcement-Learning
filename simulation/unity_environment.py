from collections import Counter

from unityagents import UnityEnvironment
import numpy as np
from agents.base import Agent
from typing import Tuple, Optional, Union, List, Callable
from copy import deepcopy
import time
import torch
import random
import matplotlib.pyplot as plt
from unityagents.brain import BrainInfo
from tools.rl_constants import Experience, Environment, Brain, BrainSet
from tools.rl_constants import Action
from tools.scores import Scores
plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def default_step_agents_fn(brain_set: BrainSet, next_brain_environment: dict, t: int):
    for brain_name, brain_environment in next_brain_environment.items():
        for i in range(brain_set[brain_name].num_agents):
            brain_agent_experience = Experience(
                state=brain_environment['states'][i],
                action=brain_environment['actions'][i],
                reward=brain_environment['rewards'][i],
                next_state=brain_environment['next_states'][i],
                done=brain_environment['dones'][i],
                t_step=t,
            )
            brain_set[brain_name].agent.step(brain_agent_experience)

# def default_step_agents_fn(states: np.ndarray, actions_list: list, rewards: np.ndarray,
#                               next_states: np.ndarray, dones: np.ndarray, time_step: int, agents: List[Agent], **kwargs):
#     """ Prepare experiences for the agents """
#     experiences = [Experience(
#             state=states[i], action=actions_list[i].value, reward=rewards[i],
#             next_state=next_states[i], done=dones[i], t_step=time_step
#         ) for i in range(len(agents))]
#
#     for i, e in enumerate(experiences):
#         agents[i].step(e)
#
#     return experiences


class UnityEnvironmentSimulator:
    """ Helper class for training an agent in a Unity ML-Agents environment """
    def __init__(self, task_name: str, env: UnityEnvironment, observation_type: str, seed: int):
        torch.manual_seed(seed)
        random.seed(seed)
        self.env = env
        self.task_name = task_name

        self.env_info = None
        self.training_scores = None

        # Use the default brain
        # self.brain_name = self.env.brain_names[0]
        # self.brain = self.env.brains[self.brain_name]


        # # Get discrete state/action sizes
        # self.observation_type = observation_type
        # if observation_type == 'vector':
        #     self.state_shape = self.brain.vector_observation_space_size
        #     if isinstance(self.state_shape, int):
        #         self.state_shape = (1, self.state_shape)
        #     self.action_size = self.brain.vector_action_space_size
        # elif observation_type == 'visual':
        #     brain_info = self.env.reset()[self.brain_name]
        #     self.state_shape = brain_info.visual_observations[0].shape
        #     self.action_size = self.brain.vector_action_space_size
        # else:
        #     raise ValueError("Invalid observation_type {}".format(observation_type))

    def reset_env(self, train_mode: bool):
        env_info = self.env.reset(train_mode=train_mode)
        self.env_info = env_info

    def get_next_states(self, brain_set: BrainSet):
        brain_states = {}
        for brain_name, brain in brain_set:
            brain_info = self.env_info[brain_name]
            if brain.observation_type == 'vector':
                states = np.array(brain_info.vector_observations)
            elif brain.observation_type == 'visual':
                states = np.array(brain_info.visual_observations)
            else:
                raise ValueError("Invalid observation_type {}".format(brain.observation_type))

            states = brain.preprocess_state_fn(states)
            states = torch.from_numpy(states).float().to(device)

            brain_states[brain_name] = states
        return brain_states

    # def reset(self, preprocess_state_fn: Callable, train_mode: bool = True) -> torch.Tensor:
    #     """Reset the environment in training/evaluation mode
    #
    #     Args:
    #         train_mode (str): Reset the environment in train mode
    #     Returns:
    #         state (np.array): The initial environment state
    #     """
    #     brain_info = self.env.reset(train_mode=train_mode)[self.brain_name]
    #     states = self._get_state(brain_info)
    #     states = np.array([preprocess_state_fn(i) for i in states])
    #     return torch.from_numpy(states).float().to(device)

    def step(self, brain_set: BrainSet, brain_states: dict):
        brain_actions = brain_set.get_actions(brain_states)

        actions = deepcopy(brain_actions)
        self.env_info = self.env.step(actions)

        next_brain_states = self.get_next_states(brain_set)

        output = {}
        for brain_name in brain_set.names():
            output[brain_name] = {
                'states': brain_states[brain_name],
                'actions': brain_actions[brain_name],
                'next_states': next_brain_states[brain_name],
                'rewards': self.env_info[brain_name].rewards,
                'dones': self.env_info[brain_name].local_done
            }
        return output

    # def step(self, preprocess_state_fn: Calla ble, actions: List[Action], preprocess_actions_fn = lambda actions: actions) -> Union[Environment, List[Environment]]:
    #     """Push an action to the environment, receiving the next environment frame
    #
    #     Args:
    #         preprocess_state_fn (Callable): Function to perform preprocessing on a state
    #         actions (Agent or List[Agent]): Agent(s) interacting with the environment
    #
    #     Returns:
    #         environment (Environment): The next frame of the environment
    #     """
    #     actions = np.vstack([a.value for a in actions])
    #     actions = preprocess_actions_fn(actions)
    #
    #     brain_info = self.env.step(actions)[self.brain_name]
    #     next_states = self._get_state(brain_info)
    #     next_states = np.array([preprocess_state_fn(i) for i in next_states])
    #     rewards = brain_info.rewards
    #     dones = brain_info.local_done
    #     return next_states, rewards, dones

    def train(self, brain_set: BrainSet, solved_score: Optional[float] = None,
              n_episodes=2000, max_t=1000, sliding_window_size: int = 100,
              step_agents_fn=default_step_agents_fn,
              reward_accumulation_fn=lambda rewards: rewards,
              preprocess_actions_fn = lambda actions: actions,
              get_actions_list_fn = lambda agents, states: [agent.get_action(state) for agent, state in zip(agents, states)]
              ) -> Tuple[BrainSet, Scores, int, float]:
        """Train the agent in the environment

        Args:
            brain_set (BrainSet): The agent brains to undergo training
            solved_score (float): The score required to be obtained (calculated as the mean
                of the sliding window scores) in order to mark the task as solved
            n_episodes (int): The number of episodes to train over
            max_t (int): The maximum number of timesteps allowed in each episode
            sliding_window_size (int): The number of historical scores to average over
            step_agents_fn (Callable): Method for preparing experiences
            reward_accumulation_fn (Callable):
        Returns:
            agent (Agent): The trained agent
            scores (Scores): Scores object containing all historic and sliding-window scores
        """
        for brain in brain_set.brains():
            brain.agent.set_mode('train')

        self.training_scores = Scores(window_size=sliding_window_size)

        t_start = time.time()
        for i_episode in range(1, n_episodes + 1):
            self.reset_env(train_mode=True)
            brain_states = self.get_next_states(brain_set)
            # states = self.reset(preprocess_state_fn=pre, train_mode=True)
            # episode_scores = np.zeros(len(agents))
            brain_episode_scores = {brain_name:  np.zeros(brain.num_agents) for brain_name, brain in brain_set}
            # for brain_name in brain_set.brains:
            #     episode_scores[brain_name] =

            for t in range(max_t):
                # actions_list = get_actions_list_fn(agents, states)
                # actions_list = [agent.get_action(state) for agent, state in zip(agents, states)]

                # next_states, rewards, dones = self.step(preprocess_state_fn=preprocess_function, actions=actions_list, preprocess_actions_fn=preprocess_actions_fn)
                next_brain_environment = self.step(brain_set=brain_set, brain_states=brain_states)
                # next_states, rewards, dones = self.step(brain_set=brain_set, brain_actions_dict=brain_actions)

                step_agents_fn(brain_set, next_brain_environment, t)

                # step_agents_fn(states, actions_list, rewards, next_states, dones, t, agents=agents)
                # for brain_name, brain_environment in next_brain_environment.items():
                #     for i in range(brain_set[brain_name].num_agents):
                #         brain_agent_experience = Experience(
                #                 state=brain_environment['states'][i],
                #                 action=brain_environment['actions'][i],
                #                 reward=brain_environment['rewards'][i],
                #                 next_state=brain_environment['next_states'][i],
                #                 done=brain_environment['dones'][i],
                #                 t_step=t,
                #         )
                #         brain_set[brain_name].agent.step(brain_agent_experience)

                brain_states = {
                    brain_name: next_brain_environment[brain_name]['next_states']
                    for brain_name in brain_states
                }
                # next_brain_states
                # next_states = torch.from_numpy(next_states).float().to(device)
                # states = next_states

                for brain_name in brain_episode_scores:
                    brain_episode_scores[brain_name] += reward_accumulation_fn(next_brain_environment[brain_name]['rewards'])

                # episode_scores += reward_accumulation_fn(rewards)
                end_episode = False
                for brain_name in brain_set.names():
                    if np.any(next_brain_environment[brain_name]['dones']):
                        end_episode = True

                if end_episode:
                    break

            for _, brain in brain_set:
                brain.agent.step_episode(i_episode)

            # # Step the episode
            # for agent in agents:
            #     agent.step_episode(i_episode)
            # for brain_name in brain_episode_scores:
            episode_aggregated_score = float(np.mean([np.mean(brain_episode_scores[brain_name]) for brain_name in brain_episode_scores]))
            self.training_scores.add(episode_aggregated_score)

            # episode_score = float(np.mean(episode_scores))
            # self.training_scores.add(episode_score)

            if i_episode % 100 == 0:
                end = '\n'
            else:
                end = ""
            print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, episode_aggregated_score, self.training_scores.get_mean_sliding_scores()), end=end)
            if solved_score and self.training_scores.get_mean_sliding_scores() >= solved_score:
                print("\nTotal Training time = {:.1f} min".format((time.time() - t_start) / 60))
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, self.training_scores.get_mean_sliding_scores()))
                break
        training_time = round(time.time() - t_start)

        return brain_set, self.training_scores, i_episode, training_time

    def warmup(self, brain_set: BrainSet, n_episodes: int, max_t: int):
        print("Performing warmup with {} episodes and max_t={}".format(n_episodes, max_t))
        # for agent in agents:
        #     agent.set_mode('eval')
        for brain in brain_set.brains():
            brain.agent.set_mode('train')

        t1 = time.time()
        for i_episode in range(1, n_episodes + 1):
            # preprocess_state_fn = agents[0].preprocess_state
            # states = self.reset(preprocess_state_fn=preprocess_state_fn, train_mode=True)  # Train mode --> faster
            self.reset_env(train_mode=True)
            brain_states = self.get_next_states(brain_set)

            for t in range(max_t):
                actions_list = [agent.get_random_action(state) for agent, state in zip(agents, states)]
                next_states, _, dones = self.step(preprocess_state_fn=preprocess_state_fn, actions=actions_list)
                states = torch.from_numpy(next_states).float().to(device)
                if np.any(dones):
                    break
                print('\rEpisode {}\tTimestep: {:.2f}'.format(i_episode, t), end="")
        print("Finished warmup in {}s".format(round(time.time() - t1)))

    def evaluate(self, agents: Union[Agent, List[Agent]], n_episodes=5, max_t=1000) -> Tuple[List[Agent], float]:
        """ Evaluate a trained agent in an environment

        Used particularly for visualizing a trained agent

        Args:
            agents (List[Agent]): The (trained) agent to evaluate
            n_episodes (int): The number of episodes to train over
            max_t (int): The maximum number of timesteps allowed in each episode

        Returns:
            agent (Agent): Same as input agent
            scores (Scores): Scores object containing all historic and sliding-window scores
        """
        if not isinstance(agents, list):
            agents = [agents]
        t_start = time.time()
        print('\n----------Starting evaluation----------------')
        cumulative_score = 0
        preprocess_state_fn = agents[0].preprocess_state
        for i_episode in range(1, n_episodes + 1):
            for agent in agents:
                agent.set_mode('eval')

            states = self.reset(preprocess_state_fn=preprocess_state_fn, train_mode=False)
            episode_scores = np.zeros(len(agents))
            for t in range(max_t):
                actions_list = [agent.get_action(state) for agent, state in zip(agents, states)]
                next_states, rewards, dones = self.step(preprocess_state_fn=preprocess_state_fn, actions=actions_list)
                next_states = torch.from_numpy(next_states).float().to(device)
                states = next_states
                episode_scores += rewards
                if np.any(dones):
                    break
                print("\rEpisode {}: Episode score: {}".format(i_episode, np.mean(episode_scores)), end="")

            cumulative_score += np.mean(episode_scores)
            for agent in agents:
                agent.step_episode(i_episode)

        average_score = cumulative_score / i_episode
        print('\n-----------Finished evaluation in {:.2f}s with average score {}-----------'.format(time.time() - t_start, average_score))
        return agents, average_score
    #
    # def get_agent_performance(self, agents: Union[Agent, List[Agent]], n_train_episodes: int = 100, with_eval: bool = False, n_eval_episodes=10, sliding_window_size: int = 100, max_t: int = 1000) -> tuple:
    #     """
    #     :param agent: Agent model
    #     :param n_train_episodes: Number of episodes to train agent over
    #     :param with_eval:  Whether to obtain the performance metric by averaging agent performance in eval-mode over n_eval_episodes
    #     :param n_eval_episodes: Number of evaluation episodes to average over
    #     :param sliding_window_size:
    #     :param max_t:
    #     :return:
    #     """
    #     if not isinstance(agents, list):
    #         agents = [agents]
    #     t1 = time.time()
    #     # TODO: Allow for early stopping if performance is poor
    #     agent, training_scores, i_episode, training_time = self.train(
    #         agents=agents,
    #         solved_score=None,
    #         n_episodes=n_train_episodes,
    #         max_t=max_t,
    #         sliding_window_size=sliding_window_size,
    #     )
    #     t2 = time.time()
    #
    #     info = {
    #         "train_scores": training_scores,
    #         "train_time": round(t2-t1),
    #         "n_train_episodes": n_train_episodes,
    #         "n_eval_episodes": n_eval_episodes,
    #         "sliding_window_size": sliding_window_size,
    #         "max_t": max_t,
    #     }
    #
    #     if with_eval:
    #         eval_agent, average_score = self.evaluate(agents=agents, n_episodes=n_eval_episodes, max_t=max_t)
    #         info.update({'average_eval_score': average_score})
    #         # Performance is mean over all training episodes
    #         performance = average_score
    #     else:
    #         performance = float(training_scores.get_mean_sliding_scores())
    #
    #     return performance, info

    def get_rollout(self):
        pass

    def close(self):
        self.env.close()
