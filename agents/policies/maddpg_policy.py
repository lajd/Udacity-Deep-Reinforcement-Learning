import torch
import numpy as np
from typing import Optional, Tuple
from tools.misc import set_seed
from tools.rl_constants import ExperienceBatch, Action
from tools.parameter_decay import ParameterScheduler
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPGPolicy:
    """ Policy for the MADDPG algorithm"""
    def __init__(
            self,
            noise_factory,
            num_agents,
            action_dim: int,
            gamma: float = 0.99,
            seed: Optional[int] = None,
            action_range: Tuple[float, float] = (-1, 1)
    ):

        if seed:
            self.set_seed(seed)
        self.gamma = gamma
        self.agent_noise = [noise_factory() for _ in range(num_agents)]
        self.num_agents = num_agents
        # self.noise = noise
        self.action_range = action_range
        self.action_dim = action_dim

        self.t_step = 0
        self.n_step = 0

    @staticmethod
    def set_seed(seed: int):
        set_seed(seed)

    def step_episode(self, episode: int):
        """ Perform any end-of-episode updates """
        for i in range(self.num_agents):
            self.agent_noise[i].reset()

    def get_action(self, state: torch.Tensor, online_actor: torch.nn.Module, with_noise: bool = True, training: bool = False) -> np.ndarray:
        """Returns actions for given state as per current policy."""
        assert state.shape[0] == self.num_agents, state.shape[0]

        self.n_step += 1
        epsilon = max((1500 - self.n_step) / 1500, .01)

        online_actor.eval()
        with torch.no_grad():
            actions = online_actor(state)
        online_actor.train()

        if training:
            r = np.random.random()
            if r <= epsilon:
                action = np.random.uniform(-1, 1, (self.num_agents, self.action_dim))
            else:
                action = np.clip(actions.cpu().data.numpy(), -1, 1)  # epsilon greedy policy
        else:
            action = actions.cpu().data.numpy()
        # action = Action(value=action)
        return action

    def get_random_action(self, *args) -> np.ndarray:
        """ Get a random action (used for warmup) """
        uniform_distribution = torch.distributions.uniform.Uniform(*self.action_range)
        sample = uniform_distribution.sample((self.num_agents, self.action_dim))
        return sample
        # return Action(value=sample)

    def compute_actor_errors(self, experience_batch: ExperienceBatch, online_actor, target_actor, target_critic, online_critic) -> tuple:
        """ Compute the error and loss of the actor"""
        batch_size = len(experience_batch)
        action_pr_self = online_actor(experience_batch.states)
        action_pr_other = online_actor(experience_batch.joint_next_states.view(batch_size * 2, -1)[1::2]).detach()

        critic_local_input2 = torch.cat((experience_batch.joint_states, action_pr_other), dim=1)
        actor_errors = -online_critic(critic_local_input2, action_pr_self)
        actor_loss = actor_errors.mean()
        return actor_loss, actor_errors

    def compute_critic_errors(self, experience_batch: ExperienceBatch, online_actor, target_actor, target_critic, online_critic) -> tuple:
        """ Compute the error and loss of the critic"""
        batch_size = len(experience_batch)
        all_next_actions = target_actor(experience_batch.joint_next_states.view(batch_size * 2, -1)).view(batch_size, -1)
        critic_target_input = torch.cat((experience_batch.joint_next_states, all_next_actions.view(batch_size * 2, -1)[1::2]), dim=1).to(
            device)
        with torch.no_grad():
            q_target_next = target_critic(critic_target_input, all_next_actions.view(batch_size * 2, -1)[::2])
        q_targets = experience_batch.rewards + (self.gamma * q_target_next * (1 - experience_batch.dones))

        critic_local_input = torch.cat((experience_batch.joint_states, experience_batch.joint_actions.view(batch_size * 2, -1)[1::2]), dim=1).to(device)
        q_expected = online_critic(critic_local_input, experience_batch.actions)

        # critic loss
        huber_errors = torch.nn.SmoothL1Loss(reduction='none')
        td_errors = huber_errors(q_expected, q_targets.detach())
        critic_loss = td_errors.mean()

        return critic_loss, td_errors
