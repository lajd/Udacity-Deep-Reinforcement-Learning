from collections import Counter

from unityagents import UnityEnvironment
import numpy as np
from agents.base import Agent
from typing import Tuple, Optional
import time
import torch
import random
import matplotlib.pyplot as plt
from unityagents.brain import BrainInfo
from tools.rl_constants import Experience, Environment
from tools.rl_constants import Action
from tools.scores import Scores
plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)
import warnings
warnings.filterwarnings("ignore")


class UnityEnvironmentSimulator:
    """ Helper class for training an agent in a Unity ML-Agents environment """
    def __init__(self, task_name: str, env: UnityEnvironment, observation_type: str, seed: int):
        torch.manual_seed(seed)
        random.seed(seed)
        self.env = env
        self.task_name = task_name

        # Use the default brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        self.training_scores = None

        # Get discrete state/action sizes
        self.observation_type = observation_type
        if observation_type == 'vector':
            self.state_shape = self.brain.vector_observation_space_size
            if isinstance(self.state_shape, int):
                self.state_shape = (1, self.state_shape)
            self.action_size = self.brain.vector_action_space_size
        elif observation_type == 'visual':
            brain_info = self.env.reset()[self.brain_name]
            self.state_shape = brain_info.visual_observations[0].shape
            self.action_size = self.brain.vector_action_space_size
        else:
            raise ValueError("Invalid observation_type {}".format(observation_type))

    def _get_state(self, brain_info: BrainInfo):
        if self.observation_type == 'vector':
            state = brain_info.vector_observations[0]
        elif self.observation_type == 'visual':
            state = brain_info.visual_observations[0]
        else:
            raise ValueError("Invalid observation_type {}".format(self.observation_type))
        return state

    def reset(self, train_mode: bool = True) -> np.array:
        """Reset the environment in training/evaluation mode

        Args:
            train_mode (str): Reset the environment in train mode
        Returns:
            state (np.array): The initial environment state
        """
        brain_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        state = self._get_state(brain_info)
        return torch.from_numpy(state).float()

    def step(self, agent: Agent, action: int) -> Environment:
        """Push an action to the environment, receiving the next environment frame

        Args:
            action (int): The action to perform
            agent (Agent): Agent interacting with the environment

        Returns:
            environment (Environment): The next frame of the environment
        """
        brain_info = self.env.step(action)[self.brain_name]
        next_state = torch.from_numpy(self._get_state(brain_info)).float()

        next_state = agent.preprocess_state(next_state)
        reward = brain_info.rewards[0]  # get the reward
        done = brain_info.local_done[0]

        return Environment(next_state=next_state, reward=reward, done=done)

    def train(self, agent: Agent, model_save_path: Optional[str] = None, solved_score: Optional[float] = None,
              n_episodes=2000, max_t=1000, sliding_window_size: int = 100,
              eval_every: Optional[int] = 100) -> Tuple[Agent, Scores, int, float]:
        """Train the agent in the environment

        Args:
            agent (Agent): The agent to undergo training
            solved_score (float): The score required to be obtained (calculated as the mean
                of the sliding window scores) in order to mark the task as solved
            n_episodes (int): The number of episodes to train over
            max_t (int): The maximum number of timesteps allowed in each episode
            sliding_window_size (int): The number of historical scores to average over
            model_save_path (str): Path to save the agent if it solves the task
            eval_every (int): Frequency of performing evaluation

        Returns:
            agent (Agent): The trained agent
            scores (Scores): Scores object containing all historic and sliding-window scores
        """
        assert n_episodes > 0
        t_start = time.time()
        self.training_scores = Scores(window_size=sliding_window_size)

        training_time, i_episode = np.nan, np.nan
        for i_episode in range(1, n_episodes + 1):
            agent.set_mode('train')
            state = self.reset(train_mode=True)
            state = agent.preprocess_state(state)
            episode_score = 0
            for t in range(max_t):
                action: Action = agent.act(state)
                environment = self.step(agent, action.value)
                experience = Experience(state=state, action=action.value, reward=environment.reward,
                                        next_state=environment.next_state, done=environment.done, t_step=t)
                agent.step(experience)

                state = environment.next_state
                episode_score += environment.reward
                if environment.done:
                    break

            agent.step_episode(i_episode)
            self.training_scores.add(episode_score)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, self.training_scores.get_mean_sliding_scores()), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, self.training_scores.get_mean_sliding_scores()))
            if eval_every and i_episode % eval_every == 0:
                _, _ = self.evaluate(agent)
            if solved_score and self.training_scores.get_mean_sliding_scores() >= solved_score:
                print("\nTotal Training time = {:.1f} min".format((time.time() - t_start) / 60))
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, self.training_scores.get_mean_sliding_scores()))
                if model_save_path:
                    agent.save(model_save_path)
                training_time = round(time.time() - t_start)
                break
        return agent, self.training_scores, i_episode, training_time

    def evaluate(self, agent: Agent, n_episodes=5, max_t=1000) -> Tuple[Agent, float]:
        """ Evaluate a trained agent in an environment

        Used particularly for visualizing a trained agent

        Args:
            agent (Agent): The (trained) agent to evaluate
            n_episodes (int): The number of episodes to train over
            max_t (int): The maximum number of timesteps allowed in each episode

        Returns:
            agent (Agent): Same as input agent
            scores (Scores): Scores object containing all historic and sliding-window scores
        """
        t_start = time.time()
        print('\n----------Starting evaluation----------------')
        cumulative_score = 0
        for i_episode in range(1, n_episodes + 1):
            agent.set_mode('evaluate')
            state = self.reset(train_mode=False)
            state = agent.preprocess_state(state)
            episode_score = 0
            for t in range(max_t):
                action: Action = agent.act(state)
                environment = self.step(agent, action.value)
                state = environment.next_state
                episode_score += environment.reward
                if environment.done:
                    break
                print("\rEpisode {}: Episode score: {}".format(i_episode, episode_score), end="")

            cumulative_score += episode_score
            agent.step_episode(i_episode)

        average_score = cumulative_score / i_episode
        print('\n-----------Finished evaluation in {:.2f}s with average score {}-----------'.format(time.time() - t_start, average_score))
        return agent, average_score

    def get_agent_performance(self, agent: Agent, n_train_episodes: int = 100, with_eval: bool = False, n_eval_episodes=10, sliding_window_size: int = 100, max_t: int = 1000) -> tuple:
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
        agent, training_scores, i_episode, training_time = self.train(
            agent=agent,
            solved_score=None,
            n_episodes=n_train_episodes,
            max_t=max_t,
            sliding_window_size=sliding_window_size,
            eval_every=None,
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

        if with_eval:
            eval_agent, average_score = self.evaluate(agent=agent, n_episodes=n_eval_episodes, max_t=max_t)
            info.update({'average_eval_score': average_score})
            # Performance is mean over all training episodes
            performance = average_score
        else:
            performance = float(training_scores.get_mean_sliding_scores())

        return performance, info

    def close(self):
        self.env.close()
