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


class UnityEnvironmentSimulator:
    """ Helper class for training an agent in a Unity ML-Agents environment """
    def __init__(self, task_name: str, env: UnityEnvironment, observation_type: str, seed: int):
        torch.manual_seed(seed)
        random.seed(seed)
        self.env = env
        self.task_name = task_name

        self.env_info = None
        self.training_scores = None

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
            states = torch.from_numpy(states).to(device).float()

            brain_states[brain_name] = states
        return brain_states

    def step(self, brain_set: BrainSet, brain_states: dict, random_actions: bool = False):
        if random_actions:
            brain_actions = brain_set.get_random_actions(brain_states)
        else:
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

    def train(self, brain_set: BrainSet, solved_score: Optional[float] = None,
              n_episodes=2000, max_t=1000, sliding_window_size: int = 100,
              step_agents_fn=default_step_agents_fn,
              reward_accumulation_fn=lambda rewards: rewards,
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
            # brain.agent.set_warmup(False)

        self.training_scores = Scores(window_size=sliding_window_size)

        t_start = time.time()
        for i_episode in range(1, n_episodes + 1):
            self.reset_env(train_mode=True)
            brain_states = self.get_next_states(brain_set)

            brain_episode_scores = {brain_name:  np.zeros(brain.num_agents) for brain_name, brain in brain_set}

            for t in range(max_t):
                next_brain_environment = self.step(brain_set=brain_set, brain_states=brain_states)
                step_agents_fn(brain_set, next_brain_environment, t)
                brain_states = {
                    brain_name: next_brain_environment[brain_name]['next_states']
                    for brain_name in brain_states
                }
                for brain_name in brain_episode_scores:
                    brain_episode_scores[brain_name] += reward_accumulation_fn(next_brain_environment[brain_name]['rewards'])

                end_episode = False
                for brain_name in brain_set.names():
                    if np.any(next_brain_environment[brain_name]['dones']):
                        end_episode = True

                if end_episode:
                    break

            for _, brain in brain_set:
                brain.agent.step_episode(i_episode)

            episode_aggregated_score = float(np.mean([np.mean(brain_episode_scores[brain_name]) for brain_name in brain_episode_scores]))
            self.training_scores.add(episode_aggregated_score)

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

    def warmup(self, brain_set: BrainSet, n_episodes: int, max_t: int, step_agents_fn: Callable = default_step_agents_fn):
        print("Performing warmup with {} episodes and max_t={}".format(n_episodes, max_t))
        for brain in brain_set.brains():
            brain.agent.set_mode('train')
            # brain.agent.set_warmup(True)

        t1 = time.time()
        for i_episode in range(1, n_episodes + 1):
            self.reset_env(train_mode=True)
            brain_states = self.get_next_states(brain_set)
            for t in range(max_t):
                next_brain_environment = self.step(brain_set=brain_set, brain_states=brain_states, random_actions=True)
                step_agents_fn(brain_set, next_brain_environment, t)
                brain_states = {
                    brain_name: next_brain_environment[brain_name]['next_states']
                    for brain_name in brain_states
                }

                end_episode = False
                for brain_name in brain_set.names():
                    if np.any(next_brain_environment[brain_name]['dones']):
                        end_episode = True

                if end_episode:
                    break

                print('\rEpisode {}\tTimestep: {:.2f}'.format(i_episode, t), end="")
        print("Finished warmup in {}s".format(round(time.time() - t1)))

    def evaluate(self, brain_set: BrainSet,
              n_episodes=5, max_t=1000,
              reward_accumulation_fn=lambda rewards: rewards,
              ) -> Tuple[BrainSet, float]:
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
            brain.agent.set_mode('eval')

        average_score = 0
        for i_episode in range(1, n_episodes + 1):
            self.reset_env(train_mode=False)
            brain_states = self.get_next_states(brain_set)

            brain_episode_scores = {brain_name:  np.zeros(brain.num_agents) for brain_name, brain in brain_set}

            for t in range(max_t):
                next_brain_environment = self.step(brain_set=brain_set, brain_states=brain_states)

                brain_states = {
                    brain_name: next_brain_environment[brain_name]['next_states']
                    for brain_name in brain_states
                }
                for brain_name in brain_episode_scores:
                    brain_episode_scores[brain_name] += reward_accumulation_fn(next_brain_environment[brain_name]['rewards'])

                end_episode = False
                for brain_name in brain_set.names():
                    if np.any(next_brain_environment[brain_name]['dones']):
                        end_episode = True

                if end_episode:
                    break

            episode_aggregated_score = float(np.mean([np.mean(brain_episode_scores[brain_name]) for brain_name in brain_episode_scores]))
            average_score += episode_aggregated_score
            print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, episode_aggregated_score, average_score), end='\n')
        average_score /= n_episodes

        return brain_set, average_score

    def get_agent_performance(self, brain_set: BrainSet, n_train_episodes: int = 100, n_eval_episodes=10, sliding_window_size: int = 100, max_t: int = 1000) -> tuple:
        """
        :param agent: Agent model
        :param n_train_episodes: Number of episodes to train agent over
        :param with_eval:  Whether to obtain the performance metric by averaging agent performance in eval-mode over n_eval_episodes
        :param n_eval_episodes: Number of evaluation episodes to average over
        :param sliding_window_size:
        :param max_t:
        :return:
        """
        t1 = time.time()
        # TODO: Allow for early stopping if performance is poor
        brain_set, training_scores, i_episode, training_time = self.train(
            brain_set=brain_set,
            solved_score=None,
            n_episodes=n_train_episodes,
            max_t=max_t,
            sliding_window_size=sliding_window_size,
        )
        t2 = time.time()

        info = {
            "train_scores": training_scores,
            "train_time": round(t2-t1),
            "n_train_episodes": n_train_episodes,
            "n_eval_episodes": n_eval_episodes,
            "sliding_window_size": sliding_window_size,
            "max_t": max_t,
        }

        performance = float(training_scores.get_mean_sliding_scores())

        return performance, info

    def close(self):
        self.env.close()
