from collections import Counter

from unityagents import UnityEnvironment
import numpy as np
from agents.base import Agent
from typing import Tuple, Optional, Union, List, Callable
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
            states = np.array(brain_info.vector_observations)
        elif self.observation_type == 'visual':
            states = np.array(brain_info.visual_observations)
        else:
            raise ValueError("Invalid observation_type {}".format(self.observation_type))
        return states

    def reset(self, preprocess_state_fn: Callable, train_mode: bool = True) -> torch.Tensor:
        """Reset the environment in training/evaluation mode

        Args:
            train_mode (str): Reset the environment in train mode
        Returns:
            state (np.array): The initial environment state
        """
        brain_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        states = self._get_state(brain_info)
        states = np.array([preprocess_state_fn(i) for i in states])
        return torch.from_numpy(states).float().to(device)

    def step(self, preprocess_state_fn: Callable, actions: List[Action]) -> Union[Environment, List[Environment]]:
        """Push an action to the environment, receiving the next environment frame

        Args:
            preprocess_state_fn (Callable): Function to perform preprocessing on a state
            actions (Agent or List[Agent]): Agent(s) interacting with the environment

        Returns:
            environment (Environment): The next frame of the environment
        """
        actions = np.vstack([a.value for a in actions])
        brain_info = self.env.step(actions)[self.brain_name]
        next_states = self._get_state(brain_info)
        next_states = np.array([preprocess_state_fn(i) for i in next_states])
        rewards = brain_info.rewards
        dones = brain_info.local_done
        return next_states, rewards, dones

    def train(self, agents: Union[Agent, List[Agent]], solved_score: Optional[float] = None,
              n_episodes=2000, max_t=1000, sliding_window_size: int = 100
              ) -> Tuple[List[Agent], Scores, int, float]:
        """Train the agent in the environment

        Args:
            agents (List[Agent]): The agent(s) to undergo training
            solved_score (float): The score required to be obtained (calculated as the mean
                of the sliding window scores) in order to mark the task as solved
            n_episodes (int): The number of episodes to train over
            max_t (int): The maximum number of timesteps allowed in each episode
            sliding_window_size (int): The number of historical scores to average over

        Returns:
            agent (Agent): The trained agent
            scores (Scores): Scores object containing all historic and sliding-window scores
        """
        for agent in agents:
            agent.set_mode('train')
        self.training_scores = Scores(window_size=sliding_window_size)

        if not isinstance(agents, list):
            agents = [agents]

        t_start = time.time()
        preprocess_function = agents[0].preprocess_state
        for i_episode in range(1, n_episodes + 1):
            states = self.reset(preprocess_state_fn=preprocess_function, train_mode=True)
            episode_scores = np.zeros(len(agents))
            for t in range(max_t):
                actions_list = [agent.get_action(state) for agent, state in zip(agents, states)]
                next_states, rewards, dones = self.step(preprocess_state_fn=preprocess_function, actions=actions_list)
                for i in range(len(agents)):
                    e = Experience(
                        state=states[i], action=actions_list[i].value, reward=rewards[i],
                        next_state=next_states[i], done=dones[i], t_step=t
                    )
                    agents[i].step(e)
                next_states = torch.from_numpy(next_states).float().to(device)
                states = next_states
                episode_scores += rewards
                if t % 20:
                    print('\rTimestep {}\tScore: {:.2f}\tmin: {:.2f}\tmax: {:.2f}'
                          .format(t, np.mean(episode_scores), np.min(episode_scores), np.max(episode_scores)), end="")
                if np.any(dones):
                    break
            # Step the episode
            for agent in agents:
                agent.step_episode(i_episode)

            episode_score = float(np.mean(episode_scores))
            self.training_scores.add(episode_score)

            print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, episode_score, self.training_scores.get_mean_sliding_scores()),
                  end="\n")
            if solved_score and self.training_scores.get_mean_sliding_scores() >= solved_score:
                print("\nTotal Training time = {:.1f} min".format((time.time() - t_start) / 60))
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, self.training_scores.get_mean_sliding_scores()))
                break
        training_time = round(time.time() - t_start)

        return agents, self.training_scores, i_episode, training_time

    def warmup(self, agents: List[Agent], n_episodes: int, max_t: int):
        print("Performing warmup with {} episodes and max_t={}".format(n_episodes, max_t))
        for agent in agents:
            agent.set_mode('eval')
        t1 = time.time()
        for i_episode in range(1, n_episodes + 1):
            preprocess_state_fn = agents[0].preprocess_state
            states = self.reset(preprocess_state_fn=preprocess_state_fn, train_mode=True)  # Train mode --> faster
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

    def get_agent_performance(self, agents: Union[Agent, List[Agent]], n_train_episodes: int = 100, with_eval: bool = False, n_eval_episodes=10, sliding_window_size: int = 100, max_t: int = 1000) -> tuple:
        """
        :param agent: Agent model
        :param n_train_episodes: Number of episodes to train agent over
        :param with_eval:  Whether to obtain the performance metric by averaging agent performance in eval-mode over n_eval_episodes
        :param n_eval_episodes: Number of evaluation episodes to average over
        :param sliding_window_size:
        :param max_t:
        :return:
        """
        if not isinstance(agents, list):
            agents = [agents]
        t1 = time.time()
        # TODO: Allow for early stopping if performance is poor
        agent, training_scores, i_episode, training_time = self.train(
            agents=agents,
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

        if with_eval:
            eval_agent, average_score = self.evaluate(agents=agents, n_episodes=n_eval_episodes, max_t=max_t)
            info.update({'average_eval_score': average_score})
            # Performance is mean over all training episodes
            performance = average_score
        else:
            performance = float(training_scores.get_mean_sliding_scores())

        return performance, info

    def get_rollout(self):
        pass


    def close(self):
        self.env.close()
