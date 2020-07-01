import torch
import numpy as np
from typing import Optional, Tuple
from tools.misc import set_seed
from tools.rl_constants import ExperienceBatch, Action
from tools.parameter_decay import ParameterScheduler
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGPolicy:
    """ Policy for the DDPG algorithm"""
    def __init__(
            self,
            noise,
            action_dim: int,
            gamma: float = 0.99,
            seed: Optional[int] = None,
            action_range: Tuple[float, float] = (-1, 1)
    ):

        if seed:
            self.set_seed(seed)
        self.gamma = gamma
        self.noise = noise
        self.action_range = action_range
        self.action_dim = action_dim

    @staticmethod
    def set_seed(seed: int):
        set_seed(seed)

    def step_episode(self, episode: int):
        """ Perform any end-of-episode updates """
        self.noise.reset()

    def get_action(self, state: torch.Tensor, online_actor: torch.nn.Module, with_noise: bool = True):
        """Returns actions for given state as per current policy."""
        online_actor.eval()
        with torch.no_grad():
            action = online_actor(state).cpu().data.numpy()
        online_actor.train()
        if with_noise:
            action += self.noise.sample(action)
        action = np.clip(action, self.action_range[0], self.action_range[1])
        action = Action(value=action)
        return action

    def get_random_action(self, *args):
        """ Get a random action (used for warmup) """
        return Action(value=torch.distributions.uniform.Uniform(*self.action_range).sample(torch.Size((1, self.action_dim))))

    def compute_actor_errors(self, experience_batch: ExperienceBatch, online_actor, target_actor, target_critic, online_critic) -> tuple:
        """ Compute the error and loss of the actor"""
        actions_pred = online_actor(experience_batch.states)
        actor_errors = - online_critic(experience_batch.states, actions_pred)
        actor_loss = actor_errors.mean()
        return actor_loss, actor_errors

    def compute_critic_errors(self, experience_batch: ExperienceBatch, online_actor, target_actor, target_critic, online_critic) -> tuple:
        """ Compute the error and loss of the critic"""
        # Calculate the critic errors/loss
        # Get predicted next-state actions and Q values from target models
        actions_next = target_actor(experience_batch.next_states)
        next_q_targets = target_critic(experience_batch.next_states, actions_next)
        # Compute Q targets for current states (y_i)
        q_targets = experience_batch.rewards + (self.gamma * next_q_targets * (1 - experience_batch.dones))
        # Compute critic loss
        q_expected = online_critic(experience_batch.states, experience_batch.actions)
        td_errors = torch.abs(q_expected - q_targets)
        if experience_batch.is_weights is not None:
            norm_is_weights = experience_batch.get_norm_is_weights()
            td_errors *= norm_is_weights.view_as(td_errors)
        critic_loss = torch.pow(td_errors, 2).mean()
        return critic_loss, td_errors
