from unityagents import UnityEnvironment
import numpy as np
from agents.base import Agent
from collections import deque
from typing import Tuple
import time
import torch
import random
import matplotlib.pyplot as plt
# %matplotlib inline

# Add-on : Set plotting options
plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)
# Add-on : Hide Matplotlib deprecate warnings
import warnings
warnings.filterwarnings("ignore")


class Environment:
    """ Helper class for maintaining the Environment's variables at a given time step """
    def __init__(self, next_state: np.array, reward: np.array, done: np.array):
        self.next_state = next_state
        self.reward = reward
        self.done = done


class Scores:
    """ Helper class for maintaining the scores (rewards) accumulated by an agent """
    def __init__(self, sliding_scores_size: int = 100):
        self.scores = []
        self.sliding_scores = deque(maxlen=sliding_scores_size)

    def add(self, score: float):
        self.scores.append(score)
        self.sliding_scores.append(score)

    def get_mean_sliding_scores(self):
        return np.mean(self.sliding_scores)

    def plot_scores(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(self.scores)), self.scores)
        plt.title('Score (Rewards)')
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.grid(True)
        plt.show()


class UnityEnvironmentSimulator:
    """ Helper class for training an agent in a Unity ML-Agents environment """
    def __init__(self, task_name: str, env: UnityEnvironment, seed: int):
        torch.manual_seed(seed)
        random.seed(seed)
        self.env = env
        self.task_name = task_name

        # Use the default brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        # Get discrete state/action sizes
        self.state_size = self.brain.vector_observation_space_size
        self.action_size = self.brain.vector_action_space_size

    def init_environment(self, mode: str) -> np.array:
        """Reset the environment in training/evaluation mode

        Args:
            mode (str): The mode to use
        Returns:
            state (np.array): The initial environment state
        """
        if mode not in {'train', 'evaluate'}:
            raise ValueError('Mode must be train or evaluate')
        env_info = self.env.reset(train_mode=mode == 'train')[self.brain_name]
        state = env_info.vector_observations[0]
        return state

    def step_environment(self, action: int) -> Environment:
        """Push an action to the environment, receiving the next environment frame

        Args:
            action (int): The action to perform

        Returns:
            environment (Environment): The next frame of the environment
        """
        env_info = self.env.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]
        return Environment(next_state=next_state, reward=reward, done=done)

    def train(self, agent: Agent, solved_score: float, n_episodes=2000, max_t=1000, sliding_scores_size: int = 100, checkpoint_dir: str = 'checkpoints') -> Tuple[Agent, Scores]:
        """Train the agent in the environment

        Args:
            agent (Agent): The agent to undergo training
            solved_score (float): The score required to be obtained (calculated as the mean
                of the sliding window scores) in order to mark the task as solved
            n_episodes (int): The number of episodes to train over
            max_t (int): The maximum number of timesteps allowed in each episode
            sliding_scores_size (int): The number of historical scores to average over
            checkpoint_dir (str): Directory path to save checkpoints

        Returns:
            agent (Agent): The trained agent
            scores (Scores): Scores object containing all historic and sliding-window scores
        """
        t_start = time.time()
        scores = Scores(sliding_scores_size=sliding_scores_size)
        agent.set_mode('train')

        for i_episode in range(1, n_episodes + 1):
            state = self.init_environment(mode='train')
            episode_score = 0
            for t in range(max_t):
                action = agent.act(state)

                environment = self.step_environment(action)

                agent.step(
                    state=state,
                    action=action,
                    reward=environment.reward,
                    next_state=environment.next_state,
                    done=environment.done
                )

                state = environment.next_state
                episode_score += environment.reward
                if environment.done:
                    break

            scores.add(episode_score)
            agent.step_episode(i_episode)

            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, scores.get_mean_sliding_scores()), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, scores.get_mean_sliding_scores()))
            if scores.get_mean_sliding_scores() >= solved_score:
                print("\nTotal Training time = {:.1f} min".format((time.time() - t_start) / 60))
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100, scores.get_mean_sliding_scores()))
                agent.checkpoint(tag=self.task_name, checkpoint_dir=checkpoint_dir)
                break
        return agent, scores

    def evaluate(self, agent: Agent, n_episodes=1, max_t=1000, sliding_scores_size: int = 100, frame_delay=0.1) -> Tuple[Agent, Scores]:
        """ Evaluate a trained agent in an environment

        Used particularly for visualizing a trained agent

        Args:
            agent (Agent): The (trained) agent to evaluate
            n_episodes (int): The number of episodes to train over
            max_t (int): The maximum number of timesteps allowed in each episode
            sliding_scores_size (int): The number of historical scores to average over
            frame_delay (float): the amount of delay (in seconds) between frames.
                Used to visualize the agent in the environment

        Returns:
            agent (Agent): Same as input agent
            scores (Scores): Scores object containing all historic and sliding-window scores
        """
        t_start = time.time()
        scores = Scores(sliding_scores_size=sliding_scores_size)
        agent = agent.eval()
        for i_episode in range(1, n_episodes + 1):
            state = self.init_environment(mode='evaluate')
            episode_score = 0
            for t in range(max_t):
                action = agent.act(state)

                environment = self.step_environment(action)

                agent.step(
                    state=state,
                    action=action,
                    reward=environment.reward,
                    next_state=environment.next_state,
                    done=environment.done
                )

                state = environment.next_state
                episode_score += environment.reward
                if environment.done:
                    break
                if frame_delay:
                    time.sleep(frame_delay)

            scores.add(episode_score)
            agent.step_episode(i_episode)

            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, scores.get_mean_sliding_scores()), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, scores.get_mean_sliding_scores()))
        print('\rFinished evaluation in {:.2f}s'.format(time.time() - t_start))
        return agent, scores
