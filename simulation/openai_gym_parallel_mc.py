import numpy as np
from typing import Tuple, Optional, Callable
import torch
import random
import matplotlib.pyplot as plt
import gym
plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)
import warnings
warnings.filterwarnings("ignore")
import progressbar as pb
from tools.parallel_gym import ParallelGymEnvironment
from tools.rl_constants import Trajectories
from tools.scores import Scores

RIGHT = 4
LEFT = 5


class ParallelMonteCarloGymEnvironment:
    def __init__(self, task_name: str, env_name: str, parallel_count: int, seed: int, custom_preprocess_trajectories: Optional[Callable] = None):
        torch.manual_seed(seed)
        random.seed(seed)

        self.env = gym.make(env_name)
        self.envs = ParallelGymEnvironment(env_name=env_name, n=parallel_count)
        self.task_name = task_name

        self.custom_preprocess_trajectories = custom_preprocess_trajectories

        self.scores = Scores()

    def _get_trajectories(self, agent, tmax=200, nrand=5, num_stacked_frames: int = 2, no_op_action: int = 0, reset_action: int = 1):
        """ Collect trajectories in parallel """
        # collect trajectories for a parallelized parallelEnv object
        # number of parallel instances
        n = len(self.envs.ps)
        assert nrand > 0

        # initialize returning lists and start the game!
        state_list = []
        reward_list = []
        prob_list = []
        action_list = []

        self.envs.reset()

        # start all parallel agents
        self.envs.step([reset_action] * n)  #

        for _ in range(nrand):
            fr1, _, _, _ = self.envs.step(np.random.choice([RIGHT, LEFT], n))
            next_frames = [self.envs.step([no_op_action] * n)[0] for _ in range(num_stacked_frames - 1)]

        for t in range(tmax):
            # prepare the input
            # preprocess_batch properly converts two frames into
            # shape (n, 2, 80, 80), the proper input for the policy
            # this is required when building CNN with pytorch
            batch_input = self.custom_preprocess_trajectories([fr1] + next_frames)

            # probs will only be used as the pi_old
            # no gradient propagation is needed
            # so we move it to the cpu
            probs = agent.model(batch_input).squeeze().cpu().detach().numpy()

            action = np.where(np.random.rand(n) < probs, RIGHT, LEFT)
            probs = np.where(action == RIGHT, probs, 1.0 - probs)

            # advance the game (0=no action)
            # we take one action and skip game forward
            fr1, re1, is_done, _ = self.envs.step(action)
            next_frames, next_rewards = [], []
            for _ in range(num_stacked_frames - 1):
                next_frame, next_reward, is_done, _ = self.envs.step([no_op_action] * n)
                next_frames.append(next_frame)
                next_rewards.append(next_reward)

            reward = re1 + sum(next_rewards)

            # store the result
            state_list.append(batch_input)
            reward_list.append(reward)
            prob_list.append(probs)
            action_list.append(action)

            # stop if any of the trajectories is done
            # we want all the lists to be retangular
            if is_done.any():
                break

        trajectories = Trajectories(policy_outputs=prob_list, states=state_list, actions=action_list, rewards=reward_list)
        return trajectories

    def train(self, agent, num_episodes: int = 500, tmax=100, nrand=5, num_stacked_frames: int = 2):
        # widget bar to display progress
        widget = ['training loop: ', pb.Percentage(), ' ',
                  pb.Bar(), ' ', pb.ETA()]
        timer = pb.ProgressBar(widgets=widget, maxval=num_episodes).start()

        for e in range(num_episodes):
            # collect trajectories
            trajectories = self._get_trajectories(agent, tmax=tmax, nrand=nrand, num_stacked_frames=num_stacked_frames)
            agent.step(trajectories)
            total_rewards = np.sum(np.stack(trajectories.rewards), axis=0)
            # get the average reward of the parallel environments
            self.scores.add(float(np.mean(total_rewards)))

            # display some progress every 20 iterations
            if (e + 1) % 20 == 0:
                print("Episode: {0:d}, score: {1:f}".format(e + 1, np.mean(total_rewards)))
                print(total_rewards)

            # update progress widget bar
            timer.update(e + 1)

        timer.finish()
